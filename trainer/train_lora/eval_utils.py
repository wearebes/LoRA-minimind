from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import jsonlines
import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from eval_llm import build_input_text
from model.model_lora import load_lora
from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import infer_base_weight_dir, init_model, resolve_project_path, setup_seed


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_text_loose(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(text).lower())


def tokenize_for_f1(text: str) -> list[str]:
    normalized = normalize_text_loose(text)
    if not normalized:
        return []
    return re.findall(r"[\u4e00-\u9fff]|[0-9a-zA-Z]+", normalized)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def accuracy_score(prediction: str, reference: str) -> float:
    return float(normalize_text_loose(prediction) == normalize_text_loose(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_for_f1(prediction)
    ref_tokens = tokenize_for_f1(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = sum((pred_counter & ref_counter).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_text_metrics(prediction: str, reference: str) -> dict:
    prediction = "" if prediction is None else str(prediction)
    reference = "" if reference is None else str(reference)
    em = exact_match(prediction, reference)
    accuracy = accuracy_score(prediction, reference)
    f1 = token_f1(prediction, reference)
    return {
        "exact_match": em,
        "accuracy": accuracy,
        "f1": f1,
    }


def compute_dataset_metrics(predictions: list[str], references: list[str], generation_times: list[float], device: str) -> dict:
    predictions = ["" if item is None else str(item) for item in predictions]
    references = ["" if item is None else str(item) for item in references]

    try:
        from bert_score import score as bert_score_score

        _, _, bert_f1 = bert_score_score(predictions, references, lang="zh", verbose=False, device=device)
        bertscore_values = [float(item) for item in bert_f1.tolist()]
    except ImportError:
        bertscore_values = [_tfidf_similarity(pred, ref) for pred, ref in zip(predictions, references)]

    try:
        from rouge_score import rouge_scorer

        rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_l_scores = [rouge.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    except ImportError:
        rouge_l_scores = [_rouge_l_f1(pred, ref) for pred, ref in zip(predictions, references)]

    text_metrics = [compute_text_metrics(pred, ref) for pred, ref in zip(predictions, references)]
    sample_count = len(text_metrics)
    if sample_count == 0:
        raise ValueError("Evaluation dataset is empty")

    return {
        "num_samples": sample_count,
        "rouge_l": float(sum(rouge_l_scores) / sample_count),
        "bertscore_f1": float(sum(bertscore_values) / sample_count),
        "accuracy": float(sum(item["accuracy"] for item in text_metrics) / sample_count),
        "f1": float(sum(item["f1"] for item in text_metrics) / sample_count),
        "em": float(sum(item["exact_match"] for item in text_metrics) / sample_count),
        "avg_generation_time_s": float(sum(generation_times) / sample_count),
    }


def _rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_for_f1(prediction)
    ref_tokens = tokenize_for_f1(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs_length(pred_tokens, ref_tokens)
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for idx, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def _tfidf_similarity(prediction: str, reference: str) -> float:
    texts = [normalize_text_loose(prediction), normalize_text_loose(reference)]
    if not any(texts):
        return 0.0
    try:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5))
        matrix = vectorizer.fit_transform(texts)
        similarity = cosine_similarity(matrix[0], matrix[1])[0][0]
        return float(max(0.0, min(1.0, similarity)))
    except ValueError:
        from difflib import SequenceMatcher

        return float(SequenceMatcher(None, texts[0], texts[1]).ratio())


def extract_eval_sample(sample: dict):
    if "conversations" in sample:
        conversations = sample["conversations"]
        for idx in range(len(conversations) - 1, -1, -1):
            message = conversations[idx]
            if message.get("role") == "assistant":
                prompt_messages = conversations[:idx]
                prompt = next((item.get("content", "") for item in reversed(prompt_messages) if item.get("role") == "user"), "")
                reference = message.get("content") or ""
                return prompt_messages, prompt, reference
        raise ValueError("No assistant reference found in conversations sample")

    prompt = sample.get("user_input", "")
    reference = sample.get("answer_r1") or sample.get("answer") or ""
    return [{"role": "user", "content": prompt}], prompt, reference


def generate_prediction(model, tokenizer, args, conversation, prompt):
    input_text = build_input_text(tokenizer, args.from_weight, conversation, prompt)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len,
    ).to(args.device)

    prompt_tokens = inputs["input_ids"].shape[1]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    start_time = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    prediction = tokenizer.decode(generated_ids[0][prompt_tokens:], skip_special_tokens=True)
    return prediction, elapsed


def load_lora_evaluation_model(args, train_config: dict):
    skip_lora = bool(getattr(args, "skip_lora", False))
    args.lora_name = args.lora_name or train_config.get("model_name")
    args.from_weight = args.from_weight or train_config.get("base_weight", "full_sft")
    args.hidden_size = args.hidden_size if args.hidden_size is not None else train_config.get("hidden_size", 768)
    args.num_hidden_layers = args.num_hidden_layers if args.num_hidden_layers is not None else train_config.get("num_hidden_layers", 16)
    args.use_moe = int(train_config.get("use_moe", 0) if args.use_moe is None else args.use_moe)
    args.lora_top_layers = args.lora_top_layers if args.lora_top_layers is not None else train_config.get("lora_top_layers", 4)
    args.lora_rank = args.lora_rank if args.lora_rank is not None else train_config.get("lora_rank", 8)
    args.max_seq_len = args.max_seq_len if args.max_seq_len is not None else train_config.get("max_seq_len", 340)
    args.batch_size = args.batch_size if args.batch_size is not None else train_config.get("batch_size", 32)
    args.epochs = args.epochs if args.epochs is not None else train_config.get("epochs", 0)
    args.learning_rate = args.learning_rate if args.learning_rate is not None else train_config.get("learning_rate", 0.0)
    if args.lora_name is None:
        args.lora_name = f"lora_{args.lora_top_layers}"

    args.base_weight_dir = str(resolve_project_path(args.base_weight_dir)) if args.base_weight_dir else str(infer_base_weight_dir(args.save_dir))

    setup_seed(args.seed)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        device=args.device,
        tokenizer_path=str(Path(__file__).resolve().parents[2] / "model"),
        save_dir=args.base_weight_dir,
    )
    lora_path = None
    if not skip_lora:
        lora_path = Path(args.lora_path) if getattr(args, "lora_path", None) else Path(train_config.get("artifact_path") or (Path(args.save_dir) / f"{args.lora_name}.pth"))
        load_lora(
            model,
            str(lora_path),
            rank=args.lora_rank,
            top_n_layers=(args.lora_top_layers if args.lora_top_layers > 0 else None),
        )
    return model.eval(), tokenizer, lora_path


def evaluate_dataset(model, tokenizer, args):
    dataset = load_dataset("json", data_files=args.eval_data_path, split="train")
    predictions: list[str] = []
    references: list[str] = []
    generation_times: list[float] = []
    records: list[dict] = []

    for idx, sample in enumerate(dataset):
        conversation, prompt, reference = extract_eval_sample(sample)
        prediction, elapsed = generate_prediction(model, tokenizer, args, conversation, prompt)
        predictions.append(prediction)
        references.append(reference)
        generation_times.append(elapsed)
        sample_metrics = compute_text_metrics(prediction, reference)
        records.append(
            {
                "sample_index": idx,
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction,
                "generation_time_s": elapsed,
                "exact_match": sample_metrics["exact_match"],
                "accuracy": sample_metrics["accuracy"],
                "f1": sample_metrics["f1"],
            }
        )

    if not predictions:
        raise ValueError("Evaluation dataset is empty")

    metrics = compute_dataset_metrics(predictions, references, generation_times, args.device)
    return metrics, records


def write_eval_outputs(output_dir: Path, tag: str, metrics: dict, records: list[dict]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"eval_{tag}_metrics.json"
    predictions_path = output_dir / f"eval_{tag}_predictions.jsonl"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with jsonlines.open(predictions_path, mode="w") as writer:
        writer.write_all(records)
    return metrics_path, predictions_path
