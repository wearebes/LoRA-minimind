def compute_val_loss_from_dataset(*args, **kwargs):
    from .eval import compute_val_loss_from_dataset as _compute_val_loss_from_dataset

    return _compute_val_loss_from_dataset(*args, **kwargs)


__all__ = ["compute_val_loss_from_dataset"]
