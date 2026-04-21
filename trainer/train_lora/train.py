import os
import sys
import json
import time
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from model.model_lora import apply_lora, load_lora, merge_lora_to_base, save_lora
from trainer.train_lora.eval import compute_val_loss_from_dataset
from trainer.train_lora.metadata import (
    TARGET_MODULES,
    artifact_path,
    build_lora_summary,
    load_summary,
    save_summary,
    update_summary_with_best,
)
from trainer.trainer_utils import (
    PROJECT_ROOT,
    Logger,
    SkipBatchSampler,
    get_lr,
    infer_base_weight_dir,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    resolve_project_path,
    setup_seed,
)
from model.model_minimind import MiniMindConfig
from dataset.lora_dataset import LoRADataset

warnings.filterwarnings('ignore')


def maybe_limit_dataset(dataset, max_samples):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def build_dataset_from_path(jsonl_path, tokenizer, max_length):
    path = Path(jsonl_path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".jsonl")
        if not files:
            raise FileNotFoundError(f"No jsonl files found in {path}")
        datasets = [LoRADataset(str(file_path), tokenizer, max_length=max_length) for file_path in files]
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return LoRADataset(str(path), tokenizer, max_length=max_length)


def build_train_eval_datasets(data_path, tokenizer, max_length, eval_data_path=None, eval_max_samples=None):
    train_source = build_dataset_from_path(data_path, tokenizer, max_length)
    if eval_data_path:
        eval_ds = build_dataset_from_path(eval_data_path, tokenizer, max_length)
        return train_source, maybe_limit_dataset(eval_ds, eval_max_samples)
    return train_source, None


def train_epoch(
    epoch,
    loader,
    iters,
    lora_params,
    tokenizer=None,
    start_step=0,
    wandb=None,
    loss_history=None,
    val_history=None,
    eval_dataset=None,
    best_state=None,
    track_loss_history=True,
    track_val_history=True,
):
    start_time = time.time()
    if loss_history is None:
        loss_history = []
    if val_history is None:
        val_history = []

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
            avg_time_per_step = spend_time / (step + 1)
            remaining_steps = iters - step - 1
            eta_seconds = avg_time_per_step * remaining_steps
            eta_min = eta_seconds / 60
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, eta: {eta_min:.1f}min'
            )
            if track_loss_history:
                loss_history.append(
                    {
                        'epoch': epoch + 1,
                        'step': step,
                        'loss': current_loss,
                        'logits_loss': current_logits_loss,
                        'aux_loss': current_aux_loss,
                        'lr': current_lr,
                    }
                )
            if wandb:
                wandb.log(
                    {
                        'train/loss': current_loss,
                        'train/logits_loss': current_logits_loss,
                        'train/aux_loss': current_aux_loss,
                        'train/learning_rate': current_lr,
                        'train/eta_min': eta_min,
                    }
                )

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            lm_checkpoint(
                lm_config,
                weight=args.lora_name,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='checkpoints',
            )

            if eval_dataset is not None and tokenizer is not None:
                val_loss = compute_val_loss_from_dataset(
                    model=model,
                    dataset=eval_dataset,
                    batch_size=args.batch_size,
                )
                val_record = {
                    'epoch': epoch + 1,
                    'step': step,
                    'val_loss': val_loss,
                }
                if track_val_history:
                    val_history.append(val_record)
                Logger(f'Validation loss at step {step}: {val_loss:.4f}')
                if wandb:
                    wandb.log({'train/val_loss': val_loss, 'train/val_step': step, 'train/val_epoch': epoch + 1})

                if best_state is not None and (best_state['val_loss'] is None or val_loss < best_state['val_loss']):
                    best_state['val_loss'] = val_loss
                    best_state['epoch'] = epoch + 1
                    best_state['step'] = step
                    save_lora(model, str(best_state['artifact_path']))
                    best_state['summary'].update(
                        update_summary_with_best(
                            best_state['summary'],
                            val_loss=val_loss,
                            epoch=epoch + 1,
                            step=step,
                        )
                    )
                    save_summary(args.save_dir, best_state['summary'])

            model.train()

        del input_ids, labels, res, loss


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='768 LoRA Fine-tuning')
    parser.add_argument('--save_dir', type=str, default='out/lora/finance_top4', help='directory for LoRA outputs')
    parser.add_argument('--lora_name', type=str, default=None, help='LoRA adapter name, for example lora_4')
    parser.add_argument('--base_weight_dir', type=str, default=None, help='directory containing base model weights')

    parser.add_argument('--epochs', type=int, default=8, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='training device')
    parser.add_argument('--dtype', type=str, default='bfloat16', help='autocast dtype')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping value')
    parser.add_argument('--log_interval', type=int, default=10, help='logging interval in steps')
    parser.add_argument('--save_interval', type=int, default=200, help='checkpoint save interval in steps')

    parser.add_argument('--lora_top_layers', type=int, default=16, help='apply LoRA to the top N transformer layers')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=None, help='LoRA alpha scaling (defaults to rank when unset)')
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout rate')
    parser.add_argument('--target_modules', type=str, default=','.join(TARGET_MODULES), help='comma-separated LoRA target module suffixes')
    parser.add_argument('--hidden_size', default=768, type=int, help='model hidden size')
    parser.add_argument('--num_hidden_layers', default=16, type=int, help='number of transformer layers')
    parser.add_argument('--max_seq_len', default=340, type=int, help='training sequence length')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='enable MoE or not')
    parser.add_argument('--data_path', type=str, default='dataset/lora_dataset/splits/train.jsonl', help='LoRA training dataset path')
    parser.add_argument('--eval_data_path', type=str, default='dataset/lora_dataset/splits/val.jsonl', help='validation dataset path')
    parser.add_argument('--eval_max_samples', type=int, default=256, help='max validation samples')
    parser.add_argument('--from_weight', default='full_sft', type=str, help='base weight name')
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help='resume from checkpoint or not')
    parser.add_argument('--use_swanlab', '--use_wandb', dest='use_swanlab', action='store_true', help='enable SwanLab logging')
    parser.add_argument('--swanlab_project', '--wandb_project', dest='swanlab_project', type=str, default='MiniMind-LoRA', help='SwanLab project name')
    parser.add_argument('--use_compile', default=0, type=int, choices=[0, 1], help='enable torch.compile or not')
    parser.add_argument('--auto_merge', default=0, type=int, choices=[0, 1], help='merge LoRA into base model after training (1=yes)')
    parser.add_argument('--save_history', default=0, type=int, choices=[0, 1], help='save local JSON histories for debugging')
    parser.add_argument('--save_plots', default=0, type=int, choices=[0, 1], help='save local plots for debugging')
    args = parser.parse_args()
    args.save_dir = str(resolve_project_path(args.save_dir))
    args.data_path = str(resolve_project_path(args.data_path))
    args.eval_data_path = str(resolve_project_path(args.eval_data_path)) if args.eval_data_path else None
    args.base_weight_dir = str(resolve_project_path(args.base_weight_dir)) if args.base_weight_dir else str(infer_base_weight_dir(args.save_dir))

    args.target_modules = tuple(item.strip() for item in args.target_modules.split(',') if item.strip())
    if not args.target_modules:
        args.target_modules = TARGET_MODULES

    if args.lora_name is None:
        # Auto-generate lora_name with format: lora_{targets}_top{top_layers}_rank{rank}
        targets_str = ''.join([m.replace('_proj', '') for m in args.target_modules])
        args.lora_name = f'lora_{targets_str}_top{args.lora_top_layers}_rank{args.lora_rank}'
        Logger(f'LoRA adapter name: {args.lora_name}')
    if not args.target_modules:
        args.target_modules = TARGET_MODULES

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f'cuda:{local_rank}'
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='checkpoints') if args.from_resume == 1 else None
    summary = build_lora_summary(args)
    existing_summary = load_summary(args.save_dir)
    if existing_summary and args.from_resume == 1:
        summary['created_at'] = existing_summary.get('created_at', summary['created_at'])
        summary['best_val_loss'] = existing_summary.get('best_val_loss')
        summary['best_epoch'] = existing_summary.get('best_epoch')
        summary['best_step'] = existing_summary.get('best_step')
    summary_file = save_summary(args.save_dir, summary)
    Logger(f'Summary saved to: {summary_file}')

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    autocast_ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast(dtype=dtype)

    swanlab_run = None
    if args.use_swanlab and is_main_process():
        import swanlab as swanlab_run

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f'MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}'
        swanlab_run.init(project=args.swanlab_project, name=wandb_run_name, id=wandb_id, resume=resume, config=summary)

    model, tokenizer = init_model(
        lm_config,
        args.from_weight,
        device=args.device,
        tokenizer_path=str(PROJECT_ROOT / 'model'),
        save_dir=args.base_weight_dir,
    )
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    apply_lora(
        model,
        rank=args.lora_rank,
        target_modules=args.target_modules,
        top_n_layers=(args.lora_top_layers if args.lora_top_layers > 0 else None),
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f'LLM total params: {total_params / 1e6:.3f} M')
    Logger(f'LoRA params: {lora_params_count / 1e6:.3f} M')
    Logger(f'LoRA ratio: {lora_params_count / total_params * 100:.2f}%')

    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    train_ds, eval_ds = build_train_eval_datasets(
        args.data_path,
        tokenizer,
        args.max_seq_len,
        eval_data_path=args.eval_data_path,
        eval_max_samples=args.eval_max_samples,
    )
    Logger(f'Training dataset: {args.data_path}, {len(train_ds)} samples')
    if eval_ds is not None:
        Logger(f'Validation dataset: {len(eval_ds)} samples')
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {'freqs_cos', 'freqs_sin'}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    loss_history = []
    val_history = []
    track_loss_history = bool(args.save_history) or bool(args.save_plots)
    track_val_history = bool(args.save_history)
    official_artifact_path = artifact_path(args.save_dir, args.lora_name)
    best_state = None
    if eval_ds is not None:
        best_state = {
            'val_loss': summary.get('best_val_loss'),
            'epoch': summary.get('best_epoch'),
            'step': summary.get('best_step'),
            'artifact_path': official_artifact_path,
            'summary': summary,
        }

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skipping to step {start_step + 1}')
            train_epoch(
                epoch,
                loader,
                len(loader) + skip,
                lora_params,
                tokenizer=tokenizer,
                start_step=start_step,
                wandb=swanlab_run,
                loss_history=loss_history,
                val_history=val_history,
                eval_dataset=eval_ds,
                best_state=best_state,
                track_loss_history=track_loss_history,
                track_val_history=track_val_history,
            )
        else:
            train_epoch(
                epoch,
                loader,
                len(loader),
                lora_params,
                tokenizer=tokenizer,
                start_step=0,
                wandb=swanlab_run,
                loss_history=loss_history,
                val_history=val_history,
                eval_dataset=eval_ds,
                best_state=best_state,
                track_loss_history=track_loss_history,
                track_val_history=track_val_history,
            )

    if is_main_process() and eval_ds is None:
        save_lora(model, str(official_artifact_path))
        summary.update(update_summary_with_best(summary, val_loss=None, epoch=args.epochs, step=None))
        save_summary(args.save_dir, summary)

    if is_main_process() and args.save_history == 1 and len(loss_history) > 0:
        try:
            loss_json_path = f'{args.save_dir}/loss_history.json'
            with open(loss_json_path, 'w', encoding='utf-8') as f:
                json.dump(loss_history, f, ensure_ascii=False, indent=2)
            Logger(f'Loss history saved to: {loss_json_path}')
        except Exception as exc:
            Logger(f'Failed to save loss history: {exc}')

    if is_main_process() and args.save_plots == 1 and len(loss_history) > 0:
        try:
            import matplotlib.pyplot as plt
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
            Logger(f'Loss curve saved to: {loss_fig_path}')
        except ImportError:
            Logger('matplotlib is not installed, skip loss curve export')

    if is_main_process() and args.save_history == 1 and len(val_history) > 0:
        val_json_path = Path(args.save_dir) / 'val_history.json'
        val_json_path.write_text(json.dumps(val_history, ensure_ascii=False, indent=2), encoding='utf-8')
        Logger(f'Validation history saved to: {val_json_path}')

    if is_main_process() and best_state and best_state.get('val_loss') is not None:
        Logger(
            f"Best validation loss: {best_state['val_loss']:.4f} "
            f"(epoch {best_state['epoch']}, step {best_state['step']})"
        )

    if dist.is_initialized():
        dist.destroy_process_group()

    # Auto-merge LoRA into base model after training
    if args.auto_merge == 1 and is_main_process():
        Logger('Auto-merge enabled, merging LoRA into base model...')
        lora_path = official_artifact_path
        base_weight_dir = Path(args.base_weight_dir)
        merged_output = base_weight_dir / f'merged_{args.lora_name}.pth'

        _merge_model, _ = init_model(
            lm_config,
            args.from_weight,
            device='cpu',
            tokenizer_path=str(PROJECT_ROOT / 'model'),
            save_dir=str(base_weight_dir),
        )
        load_lora(_merge_model, str(lora_path), rank=args.lora_rank, top_n_layers=(args.lora_top_layers if args.lora_top_layers > 0 else None))
        merge_lora_to_base(_merge_model)
        torch.save(_merge_model.state_dict(), merged_output)
        Logger(f'Merged model saved to {merged_output}')
