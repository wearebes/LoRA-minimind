from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from dataset.lora_split import DEFAULT_LORA_SOURCE_FILES, split_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create deterministic LoRA train/val/test splits")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/lora_dataset/splits",
        help="directory for generated split files",
    )
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="test split ratio")
    parser.add_argument(
        "--source-file",
        action="append",
        default=None,
        help="source jsonl file; pass multiple times to override the default finance shards",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_files = tuple(Path(item) for item in args.source_file) if args.source_file else DEFAULT_LORA_SOURCE_FILES
    paths = split_dataset(
        output_dir=Path(args.output_dir),
        source_files=source_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    for name in ("train", "val", "test", "meta"):
        print(f"{name}: {paths[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
