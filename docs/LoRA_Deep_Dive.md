# LoRA 深度解析：从原理到实践

> 本文档面向初学者，用通俗易懂的语言详细解释 LoRA 的核心思想、实现细节和完整训练流程。

---

## 1. 背景：为什么需要 LoRA？

### 1.1 大模型微调的困境

想象一下，你有一个训练好的大型语言模型（比如 MiniMind），它已经具备了强大的语言理解和生成能力。现在你想让它：

- 学会某个特定领域的知识（如医疗、法律）
- 遵循特定的指令风格（如对话更友好、更简洁）
- 具备某个特殊技能（如角色扮演、代码生成）

传统做法是**全量微调（Full Fine-tuning）**：对模型的所有参数进行训练。

**问题**：
- MiniMind-512 有约 **3亿参数**，全量微调需要大量显存和计算资源
- 训练完成后，需要保存**完整的模型权重**（约 1GB+）
- 如果有多个任务，需要保存多份完整权重

### 1.2 解决方案对比

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **全量微调** | 训练所有参数 | 效果最好 | 显存占用大、参数量大 |
| **LoRA** | 冻结原权重，添加小矩阵 | 只训练3%参数、效果接近全量 | 效果略低于全量 |
| **Prompt Tuning** | 只训练提示词 | 几乎不占显存 | 效果依赖模型规模 |
| **Adapter** | 插入小型网络 | 效果稳定 | 需要修改模型结构 |

---

## 2. LoRA 核心思想

### 2.1 关键洞察

LoRA 的核心发现是：**预训练模型虽然参数很多，但实际有用的"知识"往往可以用低维空间表示**。

就像一幅高清图片，虽然有上百万像素，但我们可以用压缩算法把它压缩成很小的文件，同时保持主要信息。

### 2.2 "旁路学习"机制

```
原模型权重 W（冻结不训练）
     ↓
     ├──→ 原输出：W · x
     │
     └──→ LoRA旁路（可训练）：
         输入 x → A(降维) → B(升维) → 输出
```

**最终输出 = 原输出 + LoRA调整**

### 2.3 一句话解释

> **LoRA = 在大模型旁边"挂"一个很小的学习器，只训练它，大模型本身保持不变。**

---

## 3. 数学原理

### 3.1 矩阵分解

假设原有权重矩阵 W ∈ ℝ^(d×k)：
- d = 512（输出维度）
- k = 512（输入维度）

LoRA 用两个小矩阵近似这个变换：
- **A**: d × r 的矩阵（512 × 8）
- **B**: r × k 的矩阵（8 × 512）

```
原变换: y = W · x           (需要训练 512×512 = 262,144 个参数)
LoRA:    y = W·x + B·A·x    (只需训练 512×8 + 8×512 = 8,192 个参数)
```

r 称为 **rank（秩）**，通常取 4、8、16、32 等小值。

### 3.2 参数量对比

| 模型规模 | 原参数 | LoRA参数 (r=8) | 占比 |
|---------|--------|---------------|------|
| MiniMind-512 | 300M | ~10M | ~3% |
| MiniMind-768 | 1.1B | ~20M | ~2% |

### 3.3 为什么能work？

线性代数告诉我们：对于一个大的矩阵 W，它的"有效信息"往往可以用更小的秩来近似。

这背后的理论支撑是：
- **过参数化模型**：预训练模型的参数冗余很大
- **低秩性**：任务相关的知识通常位于低维子空间
- **渐进学习**：LoRA 逐渐学习对原模型的"调整方向"

---

## 4. 代码详解

### 4.1 LoRA 模块实现

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # 秩，控制低秩矩阵大小
        
        # A矩阵：输入→低维，高斯初始化
        self.A = nn.Linear(in_features, rank, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        
        # B矩阵：低维→输出，零初始化
        self.B = nn.Linear(rank, out_features, bias=False)
        self.B.weight.data.zero_()
    
    def forward(self, x):
        # 核心计算：B(A(x))
        return self.B(self.A(x))
```

**为什么要这样初始化？**

- **A 用高斯分布**：给 LoRA 一个随机起点，让它能探索各种方向
- **B 用零初始化**：确保训练刚开始时，LoRA 输出为 0，只走原模型路径，随着训练逐渐"学会"调整

### 4.2 应用到模型

```python
def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        # 只对方阵（输入输出维度相同）添加LoRA
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建LoRA模块
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank)
            
            # 保存原forward
            original_forward = module.forward
            
            # 替换为：原输出 + LoRA输出
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            
            module.forward = forward_with_lora
```

**应用范围**：
- 在 MiniMind 中，通常对 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj` 全部添加 LoRA

### 4.3 冻结与训练

```python
# 冻结非LoRA参数，只训练LoRA参数
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False  # 冻结
```

---

## 5. 完整训练流程

### 5.1 流程概览

```
┌─────────────────────────────────────────────────────────┐
│                     训练流程                              │
├─────────────────────────────────────────────────────────┤
│  1. 加载预训练权重（如 full_sft_512.pth）                │
│  2. 初始化 MiniMind 模型                                  │
│  3. apply_lora() 添加 LoRA 旁路                          │
│  4. 冻结非 LoRA 参数                                      │
│  5. 加载指令微调数据集                                    │
│  6. 训练：只更新 LoRA 参数                                │
│  7. 保存：只保存 LoRA 权重                                │
└─────────────────────────────────────────────────────────┘
```

### 5.2 训练代码解析

```python
# ========== 1. 加载基础模型 ==========
model, tokenizer = init_model(lm_config, "full_sft")

# ========== 2. 应用LoRA ==========
apply_lora(model, rank=8)

# ========== 3. 冻结参数 ==========
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False

# ========== 4. 创建优化器（只优化LoRA） ==========
optimizer = optim.AdamW(lora_params, lr=1e-4)

# ========== 5. 训练循环 ==========
for epoch in range(epochs):
    for batch in dataloader:
        # 前向传播
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        # 反向传播（只更新LoRA参数）
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 5.3 关键配置

| 参数 | 说明 | 常用值 |
|------|------|--------|
| `rank` | LoRA 矩阵的秩 | 8, 16, 32 |
| `learning_rate` | 学习率 | 1e-4, 5e-5 |
| `epochs` | 训练轮数 | 3-10 |
| `batch_size` | 批次大小 | 8, 16, 32 |

---

## 6. 权重保存与加载

### 6.1 只保存 LoRA 权重

```python
def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 保存格式：layer_name.lora.A.weight
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

### 6.2 加载时合并

推理时，有两种方式：

**方式1：加载合并（推理快）**
```python
# 加载基础模型
model = load_pretrained("full_sft_512")
# 加载LoRA权重并合并到模型
apply_lora(model)
load_lora(model, "lora_identity.pth")
# 合并后可以删除lora分支
```

**方式2：动态加载（灵活）**
```python
# 每次推理时动态添加LoRA
model = load_pretrained("full_sft_512")
apply_lora(model)
load_lora(model, "lora_medical.pth")  # 加载医疗LoRA
# 推理...

# 切换到另一个LoRA
load_lora(model, "lora_legal.pth")  # 加载法律LoRA
# 推理...
```

---

## 7. 推理实现

### 7.1 基础模型推理

```python
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
from model.model_lora import apply_lora, load_lora

# 1. 加载基础模型
config = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
model = MiniMindForCausalLM(config)
model.load_state_dict(torch.load("full_sft_512.pth"))

# 2. 应用LoRA
apply_lora(model, rank=8)

# 3. 加载LoRA权重
load_lora(model, "lora_identity.pth")

# 4. 推理
inputs = tokenizer("你好，请介绍一下自己", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## 8. 实践技巧

### 8.1 Rank 选择

| Rank | 参数量 | 效果 | 适用场景 |
|------|--------|------|----------|
| 4 | 最少 | 一般 | 显存极其受限 |
| 8 | 较少 | 较好 | 常规实验 |
| 16 | 中等 | 很好 | 追求效果 |
| 32 | 较多 | 接近全量 | 特殊任务 |

**建议**：从 rank=8 开始尝试，效果不好再增大。

### 8.2 哪些层加 LoRA？

通常对所有线性层添加：
- Attention 的 `q_proj`, `k_proj`, `v_proj`, `o_proj`
- FeedForward 的 `gate_proj`, `up_proj`, `down_proj`

也可以只对 QKV 添加，效果稍差但参数量更少。

### 8.3 训练数据量

- **少样本**（<1000条）：效果可能不如全量微调
- **中样本**（1000-10000条）：LoRA 效果很好
- **多样本**（>10000条）：接近全量微调效果

---

## 9. LoRA vs 其他方法

### 9.1 效果对比

| 方法 | 显存占用 | 训练时间 | 推理开销 | 效果 |
|------|----------|----------|----------|------|
| 全量微调 | 高 | 长 | 无 | 100% |
| LoRA | 低 | 短 | 略高 | 95-99% |
| Adapter | 中 | 中 | 中 | 90-95% |
| Prompt Tuning | 极低 | 极短 | 无 | 80-90% |

### 9.2 适用场景

- **LoRA**：资源有限、想要快速实验、多任务切换
- **全量微调**：数据量大、追求最高效果
- **Adapter**：需要保持原模型结构

---

## 10. 总结

### 核心要点

1. **不修改原模型**：冻结预训练权重，保持原能力
2. **低秩近似**：用两个小矩阵的乘积代替大矩阵
3. **参数高效**：只需训练 2-5% 的参数
4. **效果显著**：达到全量微调 95%+ 的效果

### 工作流程

```
预训练模型 → 添加LoRA旁路 → 冻结原参数 → 训练LoRA → 保存LoRA权重
                                                        ↓
                                            加载推理（基础模型 + LoRA权重）
```

### 优势

- 🎯 **显存友好**：只需约 3% 的训练显存
- 🚀 **训练快速**：训练速度提升 2-3 倍
- 💾 **存储节省**：每个任务只需保存几十MB
- 🔄 **灵活切换**：同一个基础模型可以快速切换不同LoRA

---

## 11. 验证微调效果

完成 LoRA 训练后，需要验证模型是否学到了新能力。本节介绍如何测试和评估微调效果。

### 11.1 方式一：交互式对话测试

使用项目提供的 `eval_llm.py` 脚本进行交互式对话测试：

```bash
# 基本用法
python eval_llm.py \
    --load_from model \
    --weight full_sft \
    --lora_weight lora_identity \
    --hidden_size 512 \
    --num_hidden_layers 8
```

**参数说明**：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--load_from` | 加载来源 | `model` |
| `--weight` | 基础权重名 | `full_sft` |
| `--lora_weight` | LoRA权重名 | `lora_identity` |
| `--hidden_size` | 隐藏层维度 | `512` |
| `--num_hidden_layers` | 层数 | `8` |
| `--temperature` | 生成温度 | `0.85` |
| `--top_p` | nucleus采样 | `0.85` |
| `--max_new_tokens` | 最大生成长度 | `8192` |

**交互模式选择**：
- 输入 `0`：自动测试预设问题
- 输入 `1`：手动输入问题

**预设测试问题**（自动模式）：
```
你有什么特长？
为什么天空是蓝色的
请用Python写一个计算斐波那契数列的函数
解释一下"光合作用"的基本过程
如果明天下雨，我应该如何出门
比较一下猫和狗作为宠物的优缺点
解释什么是机器学习
推荐一些中国的美食
```

### 11.2 方式二：对比测试

为了验证 LoRA 的效果，建议进行对比测试：

**1. 对比基础模型 vs LoRA 模型**

```python
# 测试脚本示例
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
from model.model_lora import apply_lora, load_lora
import torch

# 基础模型测试
base_model = MiniMindForCausalLM(MiniMindConfig(hidden_size=512, num_hidden_layers=8))
base_model.load_state_dict(torch.load("full_sft_512.pth"))
# ... 推理测试

# LoRA 模型测试
lora_model = MiniMindForCausalLM(MiniMindConfig(hidden_size=512, num_hidden_layers=8))
lora_model.load_state_dict(torch.load("full_sft_512.pth"))
apply_lora(lora_model)
load_lora(lora_model, "lora_identity_512.pth")
# ... 推理测试
```

**2. 关注指标**

| 评估维度 | 说明 |
|---------|------|
| **回答正确性** | 模型是否能正确回答问题 |
| **指令遵循** | 是否按照要求的格式/风格回答 |
| **知识准确性** | 事实性知识是否正确 |
| **生成流畅性** | 语句是否通顺自然 |
| **推理能力** | 逻辑推理是否合理 |

### 11.3 方式三：自动化评估

可以编写自动化测试脚本，对 LoRA 模型进行批量评估：

```python
import torch
from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
from model.model_lora import apply_lora, load_lora

def load_model_with_lora(lora_path):
    config = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load("full_sft_512.pth"))
    apply_lora(model)
    load_lora(model, lora_path)
    return model.eval()

# 测试问题集
test_questions = [
    ("你好", "需要包含问候"),
    ("你是谁", "需要介绍自己"),
    ("1+1等于多少", "需要回答2"),
]

# 评估函数
def evaluate_model(model, tokenizer, questions):
    results = []
    for question, expectation in questions:
        inputs = tokenizer(question, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            "question": question,
            "response": response,
            "expectation": expectation
        })
    return results

# 运行评估
model = load_model_with_lora("lora_identity_512.pth")
results = evaluate_model(model, tokenizer, test_questions)
for r in results:
    print(f"问题: {r['question']}")
    print(f"回答: {r['response']}")
    print(f"期望: {r['expectation']}")
    print("---")
```

### 11.4 方式四：查看训练 Loss 曲线

训练过程中可以使用 Weights & Biases (wandb) 监控训练曲线：

```bash
# 训练时启用 wandb
python -m trainer.train_lora \
    --use_wandb \
    --wandb_project MiniMind-LoRA \
    --data_path ../dataset/lora_identity.jsonl
```

**观察要点**：
- loss 是否持续下降
- 是否出现过拟合（训练 loss 下降但验证 loss 上升）
- 收敛速度是否合理

### 11.5 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| **回答没有变化** | LoRA 权重未正确加载 | 检查 `--lora_weight` 参数和权重路径 |
| **回答质量下降** | 学习率过高 | 降低学习率，如 1e-5 |
| **输出重复** | temperature 过低 | 提高 temperature 到 0.7-0.9 |
| **显存不足** | batch_size 过大 | 减小 batch_size |
| **训练不收敛** | 数据质量问题 | 检查数据格式和清洗数据 |

### 11.6 快速验证命令

```bash
# 1. 验证 LoRA 权重是否存在
ls -la out/lora/

# 2. 快速推理测试（自动模式）
python eval_llm.py --lora_weight lora_identity --hidden_size 512

# 3. 手动对话测试
python eval_llm.py --lora_weight lora_identity --hidden_size 512
# 输入 1 进入手动模式，然后输入问题
```

---

## 附录：相关资源

- 原始论文：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- 本项目代码：`model/model_lora.py`、`trainer/train_lora.py`、`eval_llm.py`
