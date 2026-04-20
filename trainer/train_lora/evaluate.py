from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import jsonlines
import torch
from datasets import load_dataset

__package__ = "trainer"
sys.path.append(str(Path(__file__).resolve().parents[2]))

from eval_llm import build_input_text
from model.model_lora import load_lora
from model.model_minimind import MiniMindConfig
from trainer.train_lora.metadata import build_lora_eval_config, dataset_tag, load_training_metadata
from trainer.trainer_utils import Logger, infer_base_weight_dir, init_model, resolve_project_path, setup_seed

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser():
    parser = argparse.ArgumentParser(description="MiniMind LoRA offline evaluation")
    parser.add_argument("--save_dir", type=str, default="out/lora/finance_top4", help="directory containing LoRA outputs")
    parser.add_argument("--eval_data_path", type=str, required=True, help="evaluation dataset path")
    parser.add_argument("--lora_name", type=str, default=None, help="LoRA adapter name")
    parser.add_argument("--from_weight", type=str, default=None, help="base weight name")
    parser.add_argument("--base_weight_dir", type=str, default=None, help="directory containing base model weights")
    parser.add_argument("--hidden_size", type=int, default=None, help="model hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=None, help="number of transformer layers")
    parser.add_argument("--use_moe", default=None, type=int, choices=[0, 1], help="enable MoE or not")
    parser.add_argument("--lora_top_layers", type=int, default=None, help="apply LoRA to the top N transformer layers")
    parser.add_argument("--lora_rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("--max_seq_len", type=int, default=None, help="maximum input length")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="maximum generation length")
    parser.add_argument("--batch_size", type=int, default=None, help="training batch size metadata fallback")
    parser.add_argument("--epochs", type=int, default=None, help="training epoch metadata fallback")
    parser.add_argument("--learning_rate", type=float, default=None, help="training lr metadata fallback")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str, help="device")
    parser.add_argument("--seed", default=2026, type=int, help="random seed")
    parser.add_argument("--use_swanlab", "--use_wandb", dest="use_swanlab", action="store_true", help="enable SwanLab logging")
    parser.add_argument("--swanlab_project", "--wandb_project", dest="swanlab_project", type=str, default="MiniMind-LoRA-Eval", help="SwanLab project name")
    return parser


def apply_train_defaults(args, train_config: dict):
    args.lora_name = args.lora_name or train_config.get("model_name")
    args.from_weight = args.from_weight or train_config.get("base_weight", "full_sft")
    args.hidden_size = args.hidden_size or train_config.get("hidden_size", 768)
    args.num_hidden_layers = args.num_hidden_layers or train_config.get("num_hidden_layers", 16)
    args.use_moe = int(train_config.get("use_moe", 0) if args.use_moe is None else args.use_moe)
    args.lora_top_layers = args.lora_top_layers if args.lora_top_layers is not None else train_config.get("lora_top_layers", 4)
    args.lora_rank = args.lora_rank if args.lora_rank is not None else train_config.get("lora_rank", 8)
    args.max_seq_len = args.max_seq_len or train_config.get("max_seq_len", 340)
    args.batch_size = args.batch_size or train_config.get("batch_size", 32)
    args.epochs = args.epochs or train_config.get("epochs", 0)
    args.learning_rate = args.learning_rate or train_config.get("learning_rate", 0.0)
    if args.lora_name is None:
        args.lora_name = f"lora_{args.lora_top_layers}"


def extract_eval_sample(sample: dict):
    if "conversations" in sample:
        conversations = sample["conversations"]
        for idx in range(len(conversations) - 1, -1, -1):
            message = conversations[idx]
            if message.get("role") == "assistant":
                prompt_messages = conversations[:idx]
                prompt = next((item.get("content", "") for item in reversed(prompt_messages) if item.get("role") == "user"), "")
                reference = message.get("content", "")
                return prompt_messages, prompt, reference
        raise ValueError("No assistant reference found in conversations sample")

    prompt = sample.get("user_input", "")
    reference = sample.get("answer_r1", sample.get("answer", ""))
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


def compute_metrics(predictions, references, generation_times, device):
    try:
        from bert_score import score as bert_score_score
    except ImportError as exc:
        raise ImportError("bert-score is required for LoRA evaluation. Install dependencies from requirements.txt.") from exc
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError("rouge-score is required for LoRA evaluation. Install dependencies from requirements.txt.") from exc

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l_scores = [rouge.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    _, _, bert_f1 = bert_score_score(predictions, references, lang="zh", verbose=False, device=device)
    return {
        "eval/rouge_l": float(sum(rouge_l_scores) / len(rouge_l_scores)),
        "eval/bertscore_f1": float(bert_f1.mean().item()),
        "eval/avg_generation_time_s": float(sum(generation_times) / len(generation_times)),
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.save_dir = str(resolve_project_path(args.save_dir))
    args.eval_data_path = str(resolve_project_path(args.eval_data_path))
    train_config = load_training_metadata(args.save_dir)
    apply_train_defaults(args, train_config)
    args.base_weight_dir = str(resolve_project_path(args.base_weight_dir)) if args.base_weight_dir else str(infer_base_weight_dir(args.save_dir))

    setup_seed(args.seed)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        device=args.device,
        tokenizer_path=str(PROJECT_ROOT / "model"),
        save_dir=args.base_weight_dir,
    )
    lora_path = Path(train_config.get("artifact_path") or (Path(args.save_dir) / f"{args.lora_name}.pth"))
    load_lora(
        model,
        str(lora_path),
        rank=args.lora_rank,
        top_n_layers=(args.lora_top_layers if args.lora_top_layers > 0 else None),
    )
    model.eval()

    dataset = load_dataset("json", data_files=args.eval_data_path, split="train")
    predictions = []
    references = []
    generation_times = []
    records = []

    for idx, sample in enumerate(dataset):
        conversation, prompt, reference = extract_eval_sample(sample)
        prediction, elapsed = generate_prediction(model, tokenizer, args, conversation, prompt)
        predictions.append(prediction)
        references.append(reference)
        generation_times.append(elapsed)
        records.append(
            {
                "sample_index": idx,
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction,
                "generation_time_s": elapsed,
            }
        )

    if not predictions:
        raise ValueError("Evaluation dataset is empty")

    metrics = compute_metrics(predictions, references, generation_times, args.device)
    metrics["num_samples"] = len(records)

    tag = dataset_tag(args.eval_data_path)
    metrics_path = Path(args.save_dir) / f"eval_{tag}_metrics.json"
    predictions_path = Path(args.save_dir) / f"eval_{tag}_predictions.jsonl"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with jsonlines.open(predictions_path, mode="w") as writer:
        writer.write_all(records)

    Logger(f"Saved evaluation metrics to {metrics_path}")
    Logger(f"Saved evaluation predictions to {predictions_path}")

    if args.use_swanlab:
        import swanlab as swanlab_run

        eval_config = build_lora_eval_config(train_config, args)
        eval_run_name = f"MiniMind-LoRA-Eval-{args.lora_name}-{tag}"
        swanlab_run.init(project=args.swanlab_project, name=eval_run_name, config=eval_config)
        swanlab_run.log(metrics)


if __name__ == "__main__":
    main()
