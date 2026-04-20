import re
import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=8.0, dropout=0.0):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(float(dropout)) if dropout and float(dropout) > 0 else nn.Identity()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.dropout(self.A(x))) * self.scaling


def _strip_module_prefix(name):
    return name[7:] if name.startswith("module.") else name


def _lookup_state_tensor(state_dict, key):
    if key in state_dict:
        return state_dict[key]
    module_key = f"module.{key}"
    if module_key in state_dict:
        return state_dict[module_key]
    return None


def apply_lora(
    model,
    rank=8,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    top_n_layers=None,
    lora_alpha=None,
    lora_dropout=0.0,
):
    target_modules = {str(item).strip() for item in target_modules if str(item).strip()}
    if not target_modules:
        return 0
    if top_n_layers is not None:
        top_n_layers = int(top_n_layers)
    if lora_alpha is None:
        lora_alpha = rank

    total_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    selected_layer_ids = None
    if top_n_layers is not None and top_n_layers > 0 and total_layers is not None:
        start_layer = max(0, total_layers - top_n_layers)
        selected_layer_ids = set(range(start_layer, total_layers))

    injected = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in target_modules):
            continue

        if selected_layer_ids is not None:
            match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', name)
            if match is None:
                continue
            layer_id = int(match.group(1))
            if layer_id not in selected_layer_ids:
                continue

        if hasattr(module, "lora") and getattr(module, "_lora_wrapped", False):
            continue

        lora = LoRA(
            module.in_features,
            module.out_features,
            rank=rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        ).to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(module, "lora", lora)
        module._lora_wrapped = True
        original_forward = module.forward
        module._original_forward = original_forward

        def forward_with_lora(x, layer1=original_forward, layer2=lora):
            return layer1(x) + layer2(x)

        module.forward = forward_with_lora
        injected += 1

    return injected

# "k_proj", "v_proj", "o_proj"
def load_lora(
    model,
    path,
    rank=8,
    target_modules=("q_proj"),
    top_n_layers=None,
    lora_alpha=None,
    lora_dropout=0.0,
):
    """Load LoRA weights from a state dict saved by save_lora()."""
    state_dict = torch.load(path, map_location="cpu")
    apply_lora(
        model,
        rank=rank,
        target_modules=target_modules,
        top_n_layers=top_n_layers,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    loaded_tensors = 0
    for name, module in model.named_modules():
        if not hasattr(module, "lora"):
            continue

        clean_name = _strip_module_prefix(name)
        a_tensor = _lookup_state_tensor(state_dict, f"{clean_name}.lora.A.weight")
        b_tensor = _lookup_state_tensor(state_dict, f"{clean_name}.lora.B.weight")

        if a_tensor is not None:
            module.lora.A.weight.data.copy_(a_tensor.to(module.lora.A.weight.device))
            loaded_tensors += 1
        if b_tensor is not None:
            module.lora.B.weight.data.copy_(b_tensor.to(module.lora.B.weight.device))
            loaded_tensors += 1

    print(f"Loaded LoRA tensors: {loaded_tensors}/{len(state_dict)}")


def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = _strip_module_prefix(name)
            lora_state = {f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora_to_base(model):
    """Permanently fuse LoRA weights into the base model: W_new = W + B @ A.
    After merging, LoRA adapters are removed and the model behaves as a standard model."""
    raw_model = getattr(model, "_orig_mod", model)
    merged = 0
    for name, module in raw_model.named_modules():
        if not hasattr(module, "lora"):
            continue
        lora = module.lora
        # fuse: W = W + B.weight @ A.weight  (both are [out, in], so transpose A)
        delta = lora.B.weight.data @ lora.A.weight.data  # [out_features, in_features]
        module.weight.data += delta.to(module.weight.device)
        # restore original Linear forward (remove LoRA hook)
        if hasattr(module, "_original_forward"):
            module.forward = module._original_forward
            delattr(module, "_original_forward")
        else:
            module.forward = nn.Linear.forward.__get__(module, type(module))
        if hasattr(module, "_lora_wrapped"):
            delattr(module, "_lora_wrapped")
        delattr(module, "lora")
        merged += 1
    print(f"Merged {merged} LoRA adapters into base model weights.")

