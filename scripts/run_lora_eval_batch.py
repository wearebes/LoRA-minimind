from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
TEST_DATA_PATH = Path(os.environ.get("LORA_TEST_DATA_PATH", PROJECT_ROOT / "dataset" / "lora_dataset" / "splits" / "test.jsonl"))
LORA_ROOT = PROJECT_ROOT / "out" / "lora"
EVAL_ROOT = LORA_ROOT / "eval_runs"
OUTPUT_TAG = "finance_test"
BASE_WEIGHT_DIR = PROJECT_ROOT / "out"
BASE_WEIGHT_NAME = "full_sft"
HIDDEN_SIZE = 768
NUM_HIDDEN_LAYERS = 16
LORA_RANK = 8
def ensure_test_dataset() -> Path:
    if TEST_DATA_PATH.exists():
        return TEST_DATA_PATH

    from dataset.lora_split import DEFAULT_LORA_SOURCE_FILES, split_dataset

    paths = split_dataset(output_dir=TEST_DATA_PATH.parent, source_files=DEFAULT_LORA_SOURCE_FILES)
    return paths["test"]


def discover_lora_weights() -> list[Path]:
    if not LORA_ROOT.exists():
        raise FileNotFoundError(f"Missing LoRA directory: {LORA_ROOT}")
    return sorted(
        path for path in LORA_ROOT.glob("*.pth")
        if path.is_file() and path.name != "summary.json"
    )


def parse_adapter_metadata(adapter_path: Path) -> tuple[str, str, int]:
    stem = adapter_path.stem
    match = re.match(r"^lora_(.+)_top(\d+)$", stem)
    if not match:
        raise ValueError(f"Unrecognized LoRA adapter name: {adapter_path.name}")
    module_part = match.group(1)
    top_layers = int(match.group(2))
    target_modules = ",".join(re.findall(r"(?:q_proj|k_proj|v_proj|o_proj)", module_part))
    return stem, target_modules, top_layers


def run_single_evaluation(adapter_path: Path, test_data_path: Path) -> Path:
    adapter_name, _, top_layers = parse_adapter_metadata(adapter_path)
    output_dir = EVAL_ROOT / adapter_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "trainer.train_lora.evaluate",
        "--save_dir",
        str(output_dir),
        "--output_dir",
        str(output_dir),
        "--output_tag",
        OUTPUT_TAG,
        "--eval_data_path",
        str(test_data_path),
        "--lora_path",
        str(adapter_path),
        "--lora_name",
        adapter_name,
        "--from_weight",
        BASE_WEIGHT_NAME,
        "--base_weight_dir",
        str(BASE_WEIGHT_DIR),
        "--hidden_size",
        str(HIDDEN_SIZE),
        "--num_hidden_layers",
        str(NUM_HIDDEN_LAYERS),
        "--lora_rank",
        str(LORA_RANK),
        "--lora_top_layers",
        str(top_layers),
        "--write_summary",
    ]

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return output_dir


def build_aggregate_csv(output_dirs: list[Path]) -> Path:
    aggregate_path = EVAL_ROOT / "aggregate_results.csv"
    fieldnames = [
        "adapter_name",
        "target_modules",
        "lora_top_layers",
        "num_samples",
        "rouge_l",
        "bertscore_f1",
        "accuracy",
        "f1",
        "em",
        "avg_generation_time_s",
    ]

    rows = []
    for output_dir in output_dirs:
        adapter_name = output_dir.name
        _, target_modules, top_layers = parse_adapter_metadata(LORA_ROOT / f"{adapter_name}.pth")
        metrics_path = output_dir / f"eval_{OUTPUT_TAG}_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "adapter_name": adapter_name,
                "target_modules": target_modules,
                "lora_top_layers": top_layers,
                "num_samples": metrics.get("num_samples"),
                "rouge_l": metrics.get("rouge_l"),
                "bertscore_f1": metrics.get("bertscore_f1"),
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1"),
                "em": metrics.get("em"),
                "avg_generation_time_s": metrics.get("avg_generation_time_s"),
            }
        )

    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return aggregate_path


def main() -> int:
    test_data_path = ensure_test_dataset()
    adapter_paths = discover_lora_weights()
    output_dirs: list[Path] = []

    for adapter_path in adapter_paths:
        try:
            output_dirs.append(run_single_evaluation(adapter_path, test_data_path))
        except subprocess.CalledProcessError as exc:
            print(f"[FAILED] {adapter_path.name}: exit code {exc.returncode}")
        except Exception as exc:
            print(f"[FAILED] {adapter_path.name}: {exc}")

    if output_dirs:
        aggregate_path = build_aggregate_csv(output_dirs)
        print(f"[OK] aggregate results written to {aggregate_path}")
    else:
        print("[WARN] no successful runs to aggregate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
