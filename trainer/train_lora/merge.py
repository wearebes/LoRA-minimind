from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.model_lora import load_lora
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.train_lora.metadata import load_training_metadata
from trainer.trainer_utils import infer_base_weight_dir, resolve_project_path


def merge_lora_to_base(model):
    for name, module in model.named_modules():
        if not hasattr(module, "lora"):
            continue
        delta_weight = module.lora.B.weight.data @ module.lora.A.weight.data
        module.weight.data.add_(delta_weight)
        delattr(module, "lora")
    return model


def build_parser():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model")
    parser.add_argument("--save_dir", type=str, default="out/lora/finance_top4")
    parser.add_argument("--lora_name", type=str, default=None)
    parser.add_argument("--base_weight_name", type=str, default=None)
    parser.add_argument("--base_weight_dir", type=str, default=None)
    parser.add_argument("--base_weight_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    metadata = load_training_metadata(args.save_dir)
    args.lora_name = args.lora_name or metadata.get("model_name")
    args.base_weight_name = args.base_weight_name or metadata.get("base_weight", "full_sft")
    args.hidden_size = args.hidden_size or metadata.get("hidden_size", 768)
    args.num_hidden_layers = args.num_hidden_layers or metadata.get("num_hidden_layers", 16)
    if args.lora_name is None:
        raise ValueError("LoRA name is required. Provide --lora_name or train once to generate summary.json.")
    if args.lora_rank is None:
        args.lora_rank = metadata.get("lora_rank", 8)
    lora_top_layers = metadata.get("lora_top_layers")

    base_weight_dir = resolve_project_path(args.base_weight_dir) if args.base_weight_dir else infer_base_weight_dir(args.save_dir)
    base_weight_path = resolve_project_path(args.base_weight_path)
    if base_weight_path is None:
        base_weight_path = Path(base_weight_dir) / f"{args.base_weight_name}_{args.hidden_size}.pth"

    if args.lora_path is None:
        args.lora_path = metadata.get("artifact_path") or Path(args.save_dir) / f"{args.lora_name}.pth"
    lora_path = resolve_project_path(args.lora_path)

    if args.output_path is None:
        args.output_path = Path(base_weight_dir) / f"merged_{args.lora_name}.pth"
    output_path = resolve_project_path(args.output_path)

    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model = MiniMindForCausalLM(config)
    base_state = torch.load(base_weight_path, map_location=args.device)
    model.load_state_dict(base_state, strict=True)

    load_lora(model, str(lora_path), rank=args.lora_rank, top_n_layers=(lora_top_layers if lora_top_layers else None))
    merge_lora_to_base(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved merged model to {output_path}")


if __name__ == "__main__":
    main()
