"""
LoRA权重合并脚本
将LoRA权重合并到基座模型中
"""
import argparse
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model_LoRA import load_lora


def merge_lora_to_base(model):
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            original = module.weight.data.clone()
            lora_weight = torch.matmul(module.lora.B.weight.data, module.lora.A.weight.data)
            module.weight.data = original + lora_weight
            delattr(module, 'lora')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=4, help="LoRA层数，默认4")
    args = parser.parse_args()

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # parent of LoRA/
    
    lora_name = f"lora_{args.layers}"
    merged_name = f"merge_{args.layers}"

    config = MiniMindConfig(hidden_size=768, num_hidden_layers=16)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(f"{project_root}/out/full_sft_768.pth"), strict=True)
    
    load_lora(model, f"{project_root}/out/lora/{lora_name}_768.pth")
    merge_lora_to_base(model)
    
    torch.save(model.state_dict(), f"{project_root}/out/{merged_name}_768.pth")
    print(f"已保存: {project_root}/out/{merged_name}_768.pth")
