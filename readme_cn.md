# 步骤1. 基础模型定义

## 1.1 模型概述

| 项目       | 值                                         |
| ---------- | ------------------------------------------ |
| 模型系列   | MiniMind2                                  |
| 模型规模   | ~0.1B (官方标注为104M)                     |
| 架构       | Decoder-only Causal Language Model         |
| 训练范式   | 自回归下一个token预测                      |
| 位置编码   | RoPE                                       |
| 归一化     | RMSNorm                                    |
| 激活函数   | SiLU                                       |
| 注意力机制 | Multi-head attention with grouped KV heads |
| 权重绑定   | `lm_head.weight = embed_tokens.weight`     |
| 预期用途   | LoRA / SFT微调的基础模型                   |

---

## 1.2 核心配置

| 字段                      |         值 |
| ------------------------- | ---------: |
| `hidden_size`             |        768 |
| `num_hidden_layers`       |         16 |
| `num_attention_heads`     |          8 |
| `num_key_value_heads`     |          2 |
| `head_dim`                |         96 |
| `intermediate_size`       |       2048 |
| `vocab_size`              |       6400 |
| `max_position_embeddings` |       8192 |
| `hidden_act`              |     `silu` |
| `rope_theta`              |  1000000.0 |
| `rms_norm_eps`            |       1e-5 |
| `attention_dropout`       |        0.0 |
| `attention_bias`          |      false |
| `mlp_bias`                |      false |
| `tie_word_embeddings`     |      false |
| `use_cache`               |       true |
| `torch_dtype`             | `bfloat16` |
| `bos_token_id`            |          1 |
| `eos_token_id`            |          2 |

---

## 1.3 各层结构

| 模块        | 形状        | 含义                           |
| ----------- | ----------- | ------------------------------ |
| `q_proj`    | 768 -> 768  | query projection               |
| `k_proj`    | 768 -> 192  | key projection                 |
| `v_proj`    | 768 -> 192  | value projection               |
| `o_proj`    | 768 -> 768  | attention output projection    |
| `gate_proj` | 768 -> 2048 | gated MLP branch               |
| `up_proj`   | 768 -> 2048 | expansion branch               |
| `down_proj` | 2048 -> 768 | projection back to hidden size |

---

## 1.4 衍生结构参数

| 参数                         | 公式                                |     值 |
| ---------------------------- | ----------------------------------- | -----: |
| Query head dimension         | `hidden_size / num_attention_heads` |     96 |
| Total Q output dim           | `8 x 96`                            |    768 |
| Total KV output dim          | `2 x 96`                            |    192 |
| FFN expansion ratio          | `2048 / 768`                        | 2.6667 |
| Number of transformer blocks | `num_hidden_layers`                 |     16 |

---

## 1.5 架构设计笔记

| 方面        | 观察结果                  | 对LoRA/SFT的设计启示                 |
| ----------- | ------------------------- | ------------------------------------ |
| 模型规模小  | 仅~0.1B                   | 容量有限，避免过于复杂的微调设计     |
| Hidden size | 768                       | 中等特征宽度                         |
| 深度        | 16层                      | 足够进行层级适应实验                 |
| KV头少于Q头 | `num_key_value_heads = 2` | 分组KV设计影响注意力参数形状         |
| FFN大小     | 2048                      | FFN不太宽，领域注入能力有限          |
| 上下文长度  | 8192                      | 可进行长上下文评估，但不是首要优先级 |
| 数据类型    | bfloat16                  | 训练/推理时尽量使用bf16              |

---

## 1.6 一句话模型定义

**本项目使用的基础模型**: MiniMind2 (~104M), decoder-only causal LM, `hidden_size=768`, `layers=16`, `heads=8`, `kv_heads=2`, `ffn_dim=2048`, `vocab_size=6400`, `max_position_embeddings=8192`。


# LoRA微调

## 训练

```bash
python trainer/train_lora/train.py
```

## 评测

```bash
python trainer/train_lora/evaluate.py
```

## 合并LoRA到基础模型

```bash
python trainer/train_lora/merge.py
```

## 批量实验

```bash
bash run_experiments.sh
```

## 绘制评测结果图表

```bash
python scripts/plot_eval_metric_lines.py
```

---
