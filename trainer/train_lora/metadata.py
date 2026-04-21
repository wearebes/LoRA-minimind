from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from trainer.trainer_utils import resolve_project_path

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
TARGET_MODULES_LABEL = "q,k,v,o"


def format_target_modules(modules) -> str:
    if not modules:
        return TARGET_MODULES_LABEL
    return ",".join(str(item).strip() for item in modules if str(item).strip())


def metadata_root(save_dir: str | Path, metadata_dir: str | Path | None = None) -> Path:
    target = metadata_dir if metadata_dir else save_dir
    return Path(resolve_project_path(target))


def summary_path(save_dir: str | Path, metadata_dir: str | Path | None = None) -> Path:
    return metadata_root(save_dir, metadata_dir) / "summary.json"


def experiment_config_path(save_dir: str | Path, metadata_dir: str | Path | None = None) -> Path:
    return metadata_root(save_dir, metadata_dir) / "experiment_config.json"


def best_metrics_path(save_dir: str | Path, metadata_dir: str | Path | None = None) -> Path:
    return metadata_root(save_dir, metadata_dir) / "best_val_metrics.json"


def artifact_path(save_dir: str | Path, model_name: str) -> Path:
    return Path(resolve_project_path(save_dir)) / f"{model_name}.pth"


def build_experiment_config(args) -> dict:
    return {
        "model_name": args.lora_name,
        "base_weight": args.from_weight,
        "target_modules": format_target_modules(getattr(args, "target_modules", TARGET_MODULES)),
        "lora_top_layers": args.lora_top_layers,
        "lora_rank": args.lora_rank,
        "lora_alpha": getattr(args, "lora_alpha", None),
        "lora_dropout": getattr(args, "lora_dropout", 0.0),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "use_moe": getattr(args, "use_moe", 0),
        "data_path": str(resolve_project_path(args.data_path)),
        "eval_data_path": str(resolve_project_path(args.eval_data_path)) if getattr(args, "eval_data_path", None) else None,
        "artifact_path": str(artifact_path(args.save_dir, args.lora_name)),
    }


def build_lora_summary(args) -> dict:
    return {
        "model_name": args.lora_name,
        "stage": "lora_train",
        "base_weight": args.from_weight,
        "target_modules": format_target_modules(getattr(args, "target_modules", TARGET_MODULES)),
        "lora_top_layers": args.lora_top_layers,
        "lora_rank": args.lora_rank,
        "lora_alpha": getattr(args, "lora_alpha", None),
        "lora_dropout": getattr(args, "lora_dropout", 0.0),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "artifact_path": str(artifact_path(args.save_dir, args.lora_name)),
        "best_val_loss": None,
        "best_epoch": None,
        "best_step": None,
        "train_dataset": str(resolve_project_path(args.data_path)),
        "eval_dataset": str(resolve_project_path(args.eval_data_path)) if args.eval_data_path else None,
        "use_swanlab": bool(args.use_swanlab),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def update_summary_with_best(summary: dict, *, val_loss: float | None, epoch: int | None, step: int | None) -> dict:
    updated = dict(summary)
    updated["best_val_loss"] = val_loss
    updated["best_epoch"] = epoch
    updated["best_step"] = step
    return updated


def save_summary(save_dir: str | Path, summary: dict, metadata_dir: str | Path | None = None) -> Path:
    path = summary_path(save_dir, metadata_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_experiment_config(save_dir: str | Path, config: dict, metadata_dir: str | Path | None = None) -> Path:
    path = experiment_config_path(save_dir, metadata_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_best_metrics(
    save_dir: str | Path,
    *,
    val_loss: float | None,
    epoch: int | None,
    step: int | None,
    artifact_path_value: str | Path | None = None,
    metadata_dir: str | Path | None = None,
) -> Path:
    payload = {
        "val_loss": val_loss,
        "epoch": epoch,
        "step": step,
    }
    if artifact_path_value is not None:
        payload["best_lora_path"] = str(artifact_path_value)
    path = best_metrics_path(save_dir, metadata_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary(save_dir: str | Path, metadata_dir: str | Path | None = None) -> dict:
    path = summary_path(save_dir, metadata_dir)
    return _load_json(path)


def load_training_metadata(save_dir: str | Path, metadata_dir: str | Path | None = None) -> dict:
    summary = load_summary(save_dir, metadata_dir)
    if summary:
        return summary

    config = _load_json(experiment_config_path(save_dir, metadata_dir))
    if not config:
        return {}

    best_metrics = _load_json(best_metrics_path(save_dir, metadata_dir))
    model_name = config.get("model_name", "lora")
    hidden_size = config.get("hidden_size", 768)
    legacy_artifact_path = Path(resolve_project_path(save_dir)) / f"{model_name}_{hidden_size}.pth"
    if best_metrics.get("best_lora_path"):
        legacy_artifact_path = Path(best_metrics["best_lora_path"])

    return {
        "model_name": model_name,
        "stage": "lora_train",
        "base_weight": config.get("base_weight", "full_sft"),
        "target_modules": config.get("target_modules", TARGET_MODULES_LABEL),
        "lora_top_layers": config.get("lora_top_layers", 4),
        "lora_rank": config.get("lora_rank", 8),
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout", 0.0),
        "epochs": config.get("epochs", 0),
        "learning_rate": config.get("learning_rate", 0.0),
        "batch_size": config.get("batch_size", 0),
        "max_seq_len": config.get("max_seq_len", 0),
        "hidden_size": config.get("hidden_size", 768),
        "num_hidden_layers": config.get("num_hidden_layers", 16),
        "artifact_path": str(legacy_artifact_path),
        "best_val_loss": best_metrics.get("val_loss"),
        "best_epoch": best_metrics.get("epoch"),
        "best_step": best_metrics.get("step"),
        "train_dataset": None,
        "eval_dataset": None,
        "use_swanlab": None,
        "created_at": None,
    }


def build_lora_eval_config(train_summary: dict, args) -> dict:
    eval_config = dict(train_summary)
    eval_config.update(
        {
            "stage": "lora_eval",
            "eval_dataset": str(resolve_project_path(args.eval_data_path)),
            "weight_name": args.lora_name,
            "base_weight": args.from_weight,
            "target_modules": train_summary.get("target_modules", TARGET_MODULES_LABEL),
            "lora_top_layers": args.lora_top_layers,
            "lora_rank": args.lora_rank,
            "lora_alpha": getattr(args, "lora_alpha", train_summary.get("lora_alpha")),
            "lora_dropout": getattr(args, "lora_dropout", train_summary.get("lora_dropout", 0.0)),
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
        }
    )
    return eval_config


def dataset_tag(data_path: str | Path) -> str:
    stem = Path(data_path).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
