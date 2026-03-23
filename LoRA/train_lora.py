import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from dataset.lora_dataset import LoRADataset
from model_LoRA import apply_lora, save_lora
from trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None, loss_history=None):
    start_time = time.time()
    if loss_history is None:
        loss_history = []
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 修复：正确的 ETA 计算
            avg_time_per_step = spend_time / (step + 1)
            remaining_steps = iters - step - 1
            eta_seconds = avg_time_per_step * remaining_steps
            eta_min = eta_seconds / 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, eta: {eta_min:.1f}min')
            # 记录loss到历史记录
            loss_history.append({
                'epoch': epoch + 1,
                'step': step,
                'loss': current_loss,
                'logits_loss': current_logits_loss,
                'aux_loss': current_aux_loss,
                'lr': current_lr
            })
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "eta_min": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            # LoRA只保存LoRA权重
            save_lora(model, lora_save_path)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        del input_ids, labels, res, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-2 LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out/lora", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default=None, help="LoRA权重名称，默认自动用lora_层数命名")

    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=16, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    
    parser.add_argument("--lora_top_layers", type=int, default=16,help="仅对最上方连续N层挂LoRA（0表示所有层）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=16, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_dataset/", help="LoRA训练数据路径(支持目录)")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # 自动命名: lora_层数
    if args.lora_name is None:
        args.lora_name = f"lora_{args.lora_top_layers}"
        Logger(f"自动命名: {args.lora_name}")

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用LoRA、冻结非LoRA参数 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device, tokenizer_path='../model', save_dir='../out')
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    apply_lora(model, top_n_layers=(args.lora_top_layers if args.lora_top_layers > 0 else None))
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结非LoRA参数，收集LoRA参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    
    # ========== 6. 定义数据和优化器 ==========
    # 支持目录或单文件加载
    if os.path.isdir(args.data_path):
        # 加载目录下所有jsonl文件
        jsonl_files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.endswith('.jsonl')]
        jsonl_files = sorted(jsonl_files)
        Logger(f"加载数据集目录: {args.data_path}, 找到 {len(jsonl_files)} 个文件")
        # 使用ConcatDataset合并多个数据集
        from torch.utils.data import ConcatDataset
        datasets = []
        for jsonl_file in jsonl_files:
            datasets.append(LoRADataset(jsonl_file, tokenizer, max_length=args.max_seq_len))
            Logger(f"  - {os.path.basename(jsonl_file)}: {len(datasets[-1])} 条")
        train_ds = ConcatDataset(datasets)
    else:
        train_ds = LoRADataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        Logger(f"加载数据集: {args.data_path}, {len(train_ds)} 条")
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练 ==========
    loss_history = []  # 记录所有epoch的loss
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb, loss_history)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb, loss_history)
    
    # ========== 10. 保存loss曲线图 ==========
    if is_main_process() and len(loss_history) > 0:
        try:
            import matplotlib.pyplot as plt
            import json
            import numpy as np
            
            # 保存loss数据为JSON
            loss_json_path = f'{args.save_dir}/loss_history.json'
            with open(loss_json_path, 'w', encoding='utf-8') as f:
                json.dump(loss_history, f, ensure_ascii=False, indent=2)
            Logger(f'Loss数据已保存到: {loss_json_path}')
            
            # 按 epoch 分组，计算每个 epoch 的平均 loss
            epoch_losses = {}
            epoch_logits_losses = {}
            for record in loss_history:
                epoch = record['epoch']
                loss = record['loss']
                logits_loss = record['logits_loss']
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = []
                    epoch_logits_losses[epoch] = []
                epoch_losses[epoch].append(loss)
                epoch_logits_losses[epoch].append(logits_loss)
            
            # 计算每个 epoch 的平均 loss
            epochs = sorted(epoch_losses.keys())
            avg_losses = [np.mean(epoch_losses[epoch]) for epoch in epochs]
            avg_logits_losses = [np.mean(epoch_logits_losses[epoch]) for epoch in epochs]
            # 每个 epoch 最后一个 step 的 loss
            final_losses = [epoch_losses[epoch][-1] for epoch in epochs]
            final_logits_losses = [epoch_logits_losses[epoch][-1] for epoch in epochs]
            
            Logger(f"共 {len(epochs)} 个 epoch, Epoch 1 平均 loss: {avg_losses[0]:.4f}, Epoch {len(epochs)} 平均 loss: {avg_losses[-1]:.4f}")
            
            # 绘制 loss 曲线 - 按 step 显示，同时显示 Total Loss 和 Logits Loss
            plt.figure(figsize=(10, 6))
            steps = [record['step'] for record in loss_history]
            losses = [record['loss'] for record in loss_history]
            logits_losses = [record['logits_loss'] for record in loss_history]
            plt.plot(steps, losses, 'b-', marker='o', markersize=2, label='Total Loss', linewidth=1.0)
            plt.plot(steps, logits_losses, 'r-', marker='s', markersize=2, label='Logits Loss', linewidth=1.0)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve (per Step)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()

            loss_fig_path = f'{args.save_dir}/loss_curve_by_step.png'
            plt.savefig(loss_fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            Logger(f'Loss曲线图(按step)已保存到: {loss_fig_path}')
        except ImportError:
            Logger('警告: 未安装matplotlib,跳过绘图')
    
    # ========== 11. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
