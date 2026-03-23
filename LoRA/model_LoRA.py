import torch
import re
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))

def apply_lora(model, rank=8, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"), top_n_layers=None):
    target_modules = set(target_modules)
    if top_n_layers is not None:
        top_n_layers = int(top_n_layers)

    total_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    selected_layer_ids = None
    if top_n_layers is not None and top_n_layers > 0 and total_layers is not None:
        start_layer = max(0, total_layers - top_n_layers)
        selected_layer_ids = set(range(start_layer, total_layers))

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

        lora = LoRA(module.in_features, module.out_features, rank=rank).to(module.weight.device)
        setattr(module, "lora", lora)
        original_forward = module.forward

        # 显式绑定
        def forward_with_lora(x, layer1=original_forward, layer2=lora):
            return layer1(x) + layer2(x)

        module.forward = forward_with_lora


def load_lora(model, path, rank=8, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"), top_n_layers=None):
    """
    加载LoRA权重
    
    参数:
        model: 基础模型
        path: LoRA权重路径
        rank: LoRA秩 (默认8)
        target_modules: 目标模块列表
        top_n_layers: 只对最后N层应用LoRA
    """
    state_dict = torch.load(path, map_location='cpu')
    
    # 先应用LoRA结构到模型
    apply_lora(model, rank=rank, target_modules=target_modules, top_n_layers=top_n_layers)
    
    # 尝试加载权重
    loaded_keys = set()
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            for key in state_dict.keys():
                # 提取lora参数名 (A.weight 或 B.weight)
                if '.lora.A.' in key:
                    param_name = 'A.weight'
                elif '.lora.B.' in key:
                    param_name = 'B.weight'
                else:
                    continue
                    
                # 尝试匹配模块名
                lora_key = key.split('.lora.')[0]
                if name.endswith(lora_key) or lora_key.endswith(name.split('.')[-1]):
                    module.lora.state_dict()[param_name] = state_dict[key]
                    loaded_keys.add(key)
    
    print(f"LoRA权重已加载: {len(loaded_keys)}/{len(state_dict)} keys")

def save_lora(model, path):
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
