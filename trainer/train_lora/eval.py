from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def unwrap_model(model):
    raw_model = getattr(model, "module", model)
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    return raw_model


def compute_val_loss_from_dataset(model, dataset, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    raw_model = unwrap_model(model)
    raw_model.eval()
    device = next(raw_model.parameters()).device

    total_loss = 0.0
    total_items = 0
    with torch.inference_mode():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = raw_model(input_ids, labels=labels)
            loss = outputs.loss + outputs.aux_loss
            batch_size_actual = input_ids.shape[0]
            total_loss += loss.item() * batch_size_actual
            total_items += batch_size_actual

    return total_loss / total_items if total_items else 0.0


__all__ = ["compute_val_loss_from_dataset"]
