from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

__package__ = "trainer"
sys.path.append(str(Path(__file__).resolve().parents[2]))

from trainer.train_lora.metadata import dataset_tag, load_training_metadata
from trainer.train_lora.eval_utils import evaluate_dataset, load_lora_evaluation_model, write_eval_outputs
from trainer.trainer_utils import Logger, resolve_project_path

warnings.filterwarnings("ignore")


def build_parser():
    parser = argparse.ArgumentParser(description="MiniMind LoRA offline evaluation")
    parser.add_argument("--save_dir", type=str, default="out/lora/finance_top4", help="directory containing LoRA outputs")
    parser.add_argument("--eval_data_path", type=str, default="dataset/lora_dataset/splits/test.jsonl", help="evaluation dataset path")
    parser.add_argument("--output_dir", type=str, default=None, help="directory for evaluation outputs")
    parser.add_argument("--output_tag", type=str, default=None, help="filename tag for evaluation outputs")
    parser.add_argument("--lora_path", type=str, default=None, help="explicit path to the LoRA adapter file")
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
    parser.add_argument("--skip_lora", action="store_true", help="evaluate the base model without loading a LoRA adapter")
    parser.add_argument("--write_summary", action="store_true", help="write a compact summary.json alongside the metrics")
    parser.add_argument("--use_swanlab", "--use_wandb", dest="use_swanlab", action="store_true", help="enable SwanLab logging")
    parser.add_argument("--swanlab_project", "--wandb_project", dest="swanlab_project", type=str, default="MiniMind-LoRA-Eval", help="SwanLab project name")
    return parser


def apply_train_defaults(args, train_config: dict):
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


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.save_dir = str(resolve_project_path(args.save_dir))
    args.eval_data_path = str(resolve_project_path(args.eval_data_path))
    args.output_dir = str(resolve_project_path(args.output_dir)) if args.output_dir else args.save_dir
    train_config = load_training_metadata(args.save_dir)
    apply_train_defaults(args, train_config)
    model, tokenizer, lora_path = load_lora_evaluation_model(args, train_config)
    metrics, records = evaluate_dataset(model, tokenizer, args)

    tag = args.output_tag or dataset_tag(args.eval_data_path)
    metrics_path, predictions_path = write_eval_outputs(Path(args.output_dir), tag, metrics, records)

    if lora_path is None:
        Logger("Loaded base model without LoRA adapter")
    else:
        Logger(f"Loaded LoRA from: {lora_path}")
    Logger(f"Saved evaluation metrics to {metrics_path}")
    Logger(f"Saved evaluation predictions to {predictions_path}")

    if args.write_summary:
        summary_path = Path(args.output_dir) / "summary.json"
        summary = {
            "model_name": args.lora_name if not args.skip_lora else "full_sft_768",
            "base_weight": args.from_weight,
            "eval_dataset": str(args.eval_data_path),
            "metrics": metrics,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        Logger(f"Saved summary to {summary_path}")

    if args.use_swanlab:
        import swanlab as swanlab_run

        eval_run_name = f"MiniMind-LoRA-Eval-{args.lora_name}-{tag}"
        swanlab_run.init(project=args.swanlab_project, name=eval_run_name, config={"stage": "lora_eval", "lora_name": args.lora_name, "eval_dataset": args.eval_data_path})
        swanlab_run.log(metrics)


if __name__ == "__main__":
    main()
