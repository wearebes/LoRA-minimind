# MiniMind Pipeline

当前推荐流水线：

`pretrain -> full_sft -> lora -> merge -> eval_llm -> serve_openai_api -> web_demo`

所有命令都从项目根目录运行。

## 目录结构

```text
minimind/
  model/
    model_minimind.py
    model_lora.py
  trainer/
    train_pretrain.py
    train_full_sft.py
    train_lora/
      train.py
      eval.py
      evaluate.py
      merge.py
    trainer_utils.py
  scripts/
    serve_openai_api.py
    convert_model.py
    web_demo.py
  dataset/
    smoke/
      pretrain.jsonl
      sft.jsonl
      lora.jsonl
  out/
  checkpoints/
```

## Smoke 验证流水线

### 1. 预训练

```bash
python trainer/train_pretrain.py --hidden_size 128 --num_hidden_layers 2 --epochs 1 --batch_size 2 --learning_rate 1e-4 --max_seq_len 64 --num_workers 0 --accumulation_steps 1 --log_interval 1 --save_interval 1 --data_path dataset/smoke/pretrain.jsonl --save_dir out/smoke --save_weight pretrain_smoke --from_weight none --device cuda:0
```

输出：

- `out/smoke/pretrain_smoke_128.pth`

### 2. 全量 SFT

```bash
python trainer/train_full_sft.py --hidden_size 128 --num_hidden_layers 2 --epochs 1 --batch_size 2 --learning_rate 1e-4 --max_seq_len 64 --num_workers 0 --accumulation_steps 1 --log_interval 1 --save_interval 1 --data_path dataset/smoke/sft.jsonl --save_dir out/smoke --save_weight full_sft_smoke --from_weight pretrain_smoke --device cuda:0
```

输出：

- `out/smoke/full_sft_smoke_128.pth`

### 3. LoRA 微调

```bash
python trainer/train_lora/train.py --hidden_size 128 --num_hidden_layers 2 --epochs 1 --batch_size 2 --learning_rate 1e-4 --max_seq_len 64 --num_workers 0 --accumulation_steps 1 --log_interval 1 --save_interval 1 --data_path dataset/smoke/lora.jsonl --from_weight full_sft_smoke --save_dir out/smoke/lora_smoke --lora_name lora_smoke --lora_top_layers 2 --device cuda:0
```

输出：

- `out/smoke/lora_smoke/lora_smoke.pth`
- `out/smoke/lora_smoke/summary.json`

### 4. 合并 LoRA

```bash
python trainer/train_lora/merge.py --base_weight_name full_sft_smoke --save_dir out/smoke/lora_smoke --output_path out/smoke/merged_lora_smoke.pth --device cuda:0
```

也可以改成模块方式：

```bash
python -m trainer.train_pretrain --hidden_size 128 --num_hidden_layers 2 --data_path dataset/smoke/pretrain.jsonl --save_dir out/smoke --save_weight pretrain_smoke --from_weight none --device cuda:0
python -m trainer.train_full_sft --hidden_size 128 --num_hidden_layers 2 --data_path dataset/smoke/sft.jsonl --save_dir out/smoke --save_weight full_sft_smoke --from_weight pretrain_smoke --device cuda:0
python -m trainer.train_lora.train --hidden_size 128 --num_hidden_layers 2 --data_path dataset/smoke/lora.jsonl --from_weight full_sft_smoke --save_dir out/smoke/lora_smoke --lora_name lora_smoke --lora_top_layers 2 --device cuda:0
python -m trainer.train_lora.merge --base_weight_name full_sft_smoke --save_dir out/smoke/lora_smoke --output_path out/smoke/merged_lora_smoke.pth --device cuda:0
```

输出：

- `out/smoke/merged_lora_smoke.pth`

## 推理和服务

### 5. 单次推理

```bash
python eval_llm.py --input_mode one_shot --prompt "请用一句话介绍你自己。" --weight full_sft_smoke --save_dir out/smoke --hidden_size 128 --num_hidden_layers 2 --device cuda:0
python eval_llm.py --input_mode one_shot --prompt "请用一句话介绍你自己。" --weight merged_lora_smoke --save_dir out/smoke --hidden_size 128 --num_hidden_layers 2 --device cuda:0
```

`eval_llm.py` 也支持交互模式：

```bash
python eval_llm.py --weight full_sft --save_dir out --hidden_size 768 --num_hidden_layers 16
```

### 6. OpenAI 兼容 API

```bash
python scripts/serve_openai_api.py --host 127.0.0.1 --port 8998 --weight merged_lora_smoke --save_dir out/smoke --hidden_size 128 --num_hidden_layers 2 --device cuda:0
```

接口地址：

- `GET /healthz`
- `POST /v1/chat/completions`

示例：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8998/v1", api_key="none")
resp = client.chat.completions.create(
    model="minimind",
    messages=[{"role": "user", "content": "请用一句话介绍你自己。"}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

### 7. Web Demo

```bash
streamlit run scripts/web_demo.py
```

默认行为：

- API 模式默认地址：`http://127.0.0.1:8998/v1`
- 本地模型模式会自动扫描 `huggingface_models/*/config.json`

## SwanLab 使用

当前训练脚本接入的是 SwanLab。

推荐参数名：

- `--use_swanlab`
- `--swanlab_project`

旧参数别名仍然兼容：

- `--use_wandb`
- `--wandb_project`

首次在一台机器上使用时，先登录：

```bash
swanlab login
```

或者：

```bash
swanlab login -k <api-key>
```

登录成功后，API Key 会保存在本地。正常情况下，后续训练不需要再次登录，直接执行训练命令即可。

需要重新登录的常见情况：

- 新机器
- 更换账号
- 删除本地凭证
- 覆盖已保存的 API Key

无交互环境可以用环境变量：

```bash
export SWANLAB_API_KEY=<api-key>
```

三个训练入口的启用方式一致：

```bash
python -m trainer.train_pretrain --use_swanlab --swanlab_project MiniMind-Pretrain --device cuda:0
python -m trainer.train_full_sft --use_swanlab --swanlab_project MiniMind-Full-SFT --device cuda:0
python -m trainer.train_lora --use_swanlab --swanlab_project MiniMind-LoRA --lora_rank 8 --device cuda:0
```

LoRA 常用示例：

```bash
swanlab login
python -m trainer.train_lora \
  --use_swanlab \
  --swanlab_project MiniMind-LoRA \
  --lora_rank 8 \
  --device cuda:0
```

如果已经登录过，通常只需要后半段训练命令。

最简 LoRA 训练写法：

```bash
python -m trainer.train_lora --use_swanlab
```

LoRA 评测写法：

```bash
python -m trainer.train_lora eval \
  --eval_data_path dataset/lora_dataset/splits/test.jsonl \
  --save_dir out/lora/finance_top4 \
  --use_swanlab \
  --swanlab_project MiniMind-LoRA-Eval \
  --device cuda:0
```

训练 run 和评测 run 分开记录：

- 训练 run：`train/*`
- 评测 run：`eval/*`
- 本地默认只保留 `<lora_name>.pth` 和 `summary.json`
- 如需额外落本地历史或曲线图，显式传 `--save_history 1` / `--save_plots 1`

## 权重格式转换

`.pth -> HuggingFace`：

```bash
python scripts/convert_model.py --mode torch2minimind --torch_path out/full_sft_768.pth --transformers_path huggingface_models/minimind-converted --tokenizer_path model --hidden_size 768 --num_hidden_layers 16
```

`HuggingFace -> .pth`：

```bash
python scripts/convert_model.py --mode hf2torch --transformers_path huggingface_models/minimind-converted --torch_path out/minimind-converted_768.pth
```

## 说明

- 训练输出默认写入 `out/`
- 断点状态默认写入 `checkpoints/`
- `docs/dev_logs/` 保留历史记录，不代表当前运行方式
## WSL One-Command LoRA Sweep

Use the root sweep script to run split check -> 18 LoRA trainings -> batch evaluation in one shot:

```bash
bash run_experiments.sh
```

The script activates the `minimind` conda env, uses the fixed split files below, trains 18 adapters into `out/lora/*.pth`, then evaluates them on the test split and writes aggregate results:

- `dataset/lora_dataset/splits/train.jsonl`
- `dataset/lora_dataset/splits/val.jsonl`
- `dataset/lora_dataset/splits/test.jsonl`
- `out/lora/train_runs/<adapter_name>/summary.json`
- `out/lora/train_runs/<adapter_name>/best_val_metrics.json`
- `out/lora/eval_runs/<adapter_name>/eval_finance_test_metrics.json`
- `out/lora/eval_runs/<adapter_name>/eval_finance_test_predictions.jsonl`
- `out/lora/eval_runs/aggregate_results.csv`

Common overrides can be set with environment variables before running:

```bash
CONDA_ENV=minimind SAVE_DIR=out/lora EPOCHS=8 BATCH_SIZE=32 LEARNING_RATE=1e-4 bash run_experiments.sh
```
