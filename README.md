# Step 1. Base Model Definition

## 1.1 Model Summary

| Item                          | Value                                      |
| ----------------------------- | ------------------------------------------ |
| Model family                  | MiniMind2                                  |
| Model scale                   | ~0.1B (officially listed as 104M)          |
| Architecture                  | Decoder-only Causal Language Model         |
| Training paradigm             | Autoregressive next-token prediction       |
| Positional encoding           | RoPE                                       |
| Normalization                 | RMSNorm                                    |
| Activation                    | SiLU                                       |
| Attention style               | Multi-head attention with grouped KV heads |
| Weight tying                  | `lm_head.weight = embed_tokens.weight`     |
| Intended role in this project | Base model for LoRA / SFT adaptation       |

---

## 1.2 Core Configuration

| Field                     |      Value |
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

## 1.3 Per-Layer Structure

| Module      | Shape       | Meaning                        |
| ----------- | ----------- | ------------------------------ |
| `q_proj`    | 768 -> 768  | query projection               |
| `k_proj`    | 768 -> 192  | key projection                 |
| `v_proj`    | 768 -> 192  | value projection               |
| `o_proj`    | 768 -> 768  | attention output projection    |
| `gate_proj` | 768 -> 2048 | gated MLP branch               |
| `up_proj`   | 768 -> 2048 | expansion branch               |
| `down_proj` | 2048 -> 768 | projection back to hidden size |

---

## 1.4 Derived Structural Quantities

| Quantity                     | Formula                             |  Value |
| ---------------------------- | ----------------------------------- | -----: |
| Query head dimension         | `hidden_size / num_attention_heads` |     96 |
| Total Q output dim           | `8 x 96`                            |    768 |
| Total KV output dim          | `2 x 96`                            |    192 |
| FFN expansion ratio          | `2048 / 768`                        | 2.6667 |
| Number of transformer blocks | `num_hidden_layers`                 |     16 |

---

## 1.5 Architectural Notes for Later Training Design

| Aspect                      | Observation               | Implication for later LoRA/SFT                                     |
| --------------------------- | ------------------------- | ------------------------------------------------------------------ |
| Small model scale           | only ~0.1B                | capacity is limited; avoid over-complex fine-tuning design         |
| Hidden size                 | 768                       | moderate feature width                                             |
| Depth                       | 16 layers                 | enough for layer-wise adaptation experiments                       |
| KV heads fewer than Q heads | `num_key_value_heads = 2` | grouped-KV design affects attention parameter shapes               |
| FFN size                    | 2048                      | FFN is not extremely wide, so domain injection capacity is limited |
| Context length              | 8192                      | long-context evaluation is possible, but not the first priority    |
| dtype                       | bfloat16                  | training/inference should preferably stay in bf16 when supported   |

---

## 1.6 One-Line Baseline Definition

**Base model used in this project**: MiniMind2 (~104M), decoder-only causal LM, `hidden_size=768`, `layers=16`, `heads=8`, `kv_heads=2`, `ffn_dim=2048`, `vocab_size=6400`, `max_position_embeddings=8192`.


# lora traning
We perform a layer-wise LoRA ablation with trainable depth
\[
N \in \{2,4,8,16\}
\]
where only the top-\(N\) transformer layers are adapted.

Base model: full_sft_768

Fine-tuning settings:
- Finance, Top-4
- Finance, Top-8
- Finance, Top-16
- Math+Finance, Top-4
- Math+Finance, Top-8
- Math+Finance, Top-16

Evaluation metrics:
- Accuracy
- F1
- Exact Match (EM)