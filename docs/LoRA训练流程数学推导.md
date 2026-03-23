# LoRA训练流程完整数学推导

## 目录

1. [概述](#一概述)
2. [数据维度与符号约定](#二数据维度与符号约定)
3. [模型结构与前向传播](#三模型结构与前向传播)
4. [损失函数](#四损失函数)
5. [梯度累积](#五梯度累积)
6. [反向传播与参数更新](#六反向传播与参数更新)
7. [学习率调度](#七学习率调度)
8. [混合精度训练](#八混合精度训练)
9. [检查点保存与加载](#九检查点保存与加载)
10. [一个Epoch的完整流程](#十一个epoch的完整流程)
11. [代码对照](#十一代码对照)
12. [常见问题](#十二常见问题)

---

## 一、概述

本文档详细讲解MiniMind项目中LoRA（Low-Rank Adaptation）微调训练的完整数学原理。

**LoRA的核心思想：** 不直接训练原始模型的全部参数，而是在某些层旁边添加低秩矩阵来捕获任务特定的知识。这样可以：
- 只训练约2%的参数
- 保持原始模型权重冻结
- 快速切换不同的适配器

---

## 二、数据维度与符号约定

### 2.1 常用符号

| 符号 | 含义                           | 示例值   |
| ---- | ------------------------------ | -------- |
| B    | 批量大小（batch size）         | 4        |
| L    | 序列长度（sequence length）    | 100      |
| d    | 隐藏层维度（hidden dimension） | 512      |
| V    | 词表大小（vocabulary size）    | 32000    |
| K    | 梯度累积步数                   | 3        |
| H    | 注意力头数                     | 8        |
| h    | 注意力头维度                   | d/H = 64 |

### 2.2 数据形状

```
input_ids : (B, L)     # 整数矩阵，每个元素是一个token ID
labels    : (B, L)     # 整数矩阵，每个元素是下一个token的ID
```

**示例：**
```
input_ids = [[101, 2054, 2003, 1996, 3961, 102, 0, 0, ...],
             [101, 1997, 2019, 2116, 2005, 1996, 25351, ...],
             ...]
labels    = [[2054, 2003, 1996, 3961, 102, 0, 0, 0, ...],
             [1997, 2019, 2116, 2005, 1996, 25351, 102, ...],
             ...]
```

---

## 三、模型结构与前向传播

### 3.1 LoRA的数学原理

LoRA的核心是在Transformer的某些层（通常是QKV和FFN）旁边添加两个低秩矩阵：

```
原始权重: W ∈ ℝ^{d×d}
LoRA添加: W_A ∈ ℝ^{r×d}, W_B ∈ ℝ^{d×r}, 其中 r << d

前向计算:
h_out = W · h_in + W_B · (W_A · h_in)
       = W · h_in + ΔW · h_in
其中 ΔW = W_B · W_A, rank(ΔW) = r
```

**r（秩）的选择：**
- 常用值：4, 8, 16, 32, 64
- r=8 意味着：W_A ∈ ℝ^{8×512}, W_B ∈ ℝ^{512×8}
- 参数量：8×512 + 512×8 = 8192
- 而原始W：512×512 = 262144
- **参数量减少：96.9%**

### 3.2 完整前向传播流程

```
输入: input_ids ∈ ℤ^{B×L}
     ↓
Embedding层
     ↓
h_0 ∈ ℝ^{B×L×d}
     ↓
重复L次Transformer层:
  ┌────────────────────────────────────┐
  │ 1. 注意力机制 + LoRA               │
  │    Q = W_Q · h + W_B·W_A·W_Q·h    │
  │    K = W_K · h + W_B·W_A·W_K·h    │
  │    V = W_V · h + W_B·W_A·W_V·h    │
  │    ↓                               │
  │    Attention(Q,K,V)               │
  │    ↓                               │
  │    + 残差连接 + LayerNorm          │
  │                                    │
  │ 2. FFN + LoRA                     │
  │    h = FFN(h) + LoRA(h)           │
  │    ↓                               │
  │    + 残差连接 + LayerNorm          │
  └────────────────────────────────────┘
     ↓
h_L ∈ ℝ^{B×L×d}
     ↓
Language Modeling Head (Linear)
     ↓
logits ∈ ℝ^{B×L×V}
```

### 3.3 数学公式

**注意力计算：**
```
Q = XW_Q + ΔW_QX  = (W_Q + W_BW_AW_Q)X
K = XW_K + ΔW_KX  = (W_K + W_BW_AW_K)X  
V = XW_V + ΔW_VX  = (W_V + W_BW_AW_V)X

其中 X = h_{l-1}

注意力分数:
Attention(Q,K,V) = softmax(QK^T / √d) · V

简化表示:
attn = Softmax(QK^T / √d) · V
```

**最终输出：**
```
logits = Linear(h_L) = h_L · W_O^T + b
       ∈ ℝ^{B×L×V}
```

---

## 四、损失函数

### 4.1 语言模型交叉熵损失

这是语言建模的标准损失函数，目标是预测下一个token：

```
L_main = - Σ_{i=1}^{B} Σ_{j=1}^{L} y_{i,j} · log( P( token_{i,j+1} | token_{i,≤j} ) )
```

**逐项解释：**
- y_{i,j}：第i个样本第j个位置的**真实标签**（下一个token的one-hot编码）
- P：模型预测的概率分布

**用PyTorch表示：**
```python
loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
L_main = loss_fct(logits.view(-1, V), labels.view(-1))
```

**数学等价形式：**
```
L_main = - Σ_{i=1}^{B} Σ_{j=1}^{L} log( softmax(logits_{i,j})[labels_{i,j}] )
```

### 4.2 辅助损失（Auxiliary Loss）

LoRA通常会添加一个辅助损失来正则化LoRA参数：

```
L_aux = λ · ( ||W_A||_F² + ||W_B||_F² )
```

或者使用更复杂的正则化（如l2正则、 sparsity正则）。

**F范数的定义：**
```
||W_A||_F = √( Σ_i Σ_j (W_A[i,j])² )
```

### 4.3 总损失

```
L_total = L_main + L_aux
```

**注意：** 在代码中，我们还会对这个总损失进行梯度累积处理：

```
L_scaled = L_total / K
```

---

## 五、梯度累积

### 5.1 为什么需要梯度累积

**问题：** 大batch训练效果更好，但GPU显存有限

**解决方案：** 把多个小batch的梯度累积起来，模拟大batch

**公式：**
```
实际batch_size = batch_size × accumulation_steps
                = B × K
```

### 5.2 梯度累积的数学原理

假设我们有K个小batch，每个batch计算出的梯度为∇θ_1, ∇θ_2, ..., ∇θ_K

**方法1：先平均再累积**
```
L_effective = (L_1 + L_2 + ... + L_K) / K
∇θ = ∇L_effective
```

**方法2：先累积再平均（代码中使用的）**
```
每个step: L_i' = L_i / K
累积梯度: ∇θ_accumulated = Σ_{i=1}^{K} ∇L_i'
```

**这两种方法在数学上是等价的！**

### 5.3 累积过程

```
Step 1: 
  - 计算 loss_1 / K
  - backward() → grad_1 = ∂(loss_1/K)/∂θ
  - 梯度保留在参数中（不更新）

Step 2:
  - 计算 loss_2 / K  
  - backward() → grad_2 = ∂(loss_2/K)/∂θ
  - grad = grad_1 + grad_2（PyTorch自动累积）

Step 3:
  - 计算 loss_3 / K
  - backward() → grad_3 = ∂(loss_3/K)/∂θ
  - grad = grad_1 + grad_2 + grad_3
  - 执行参数更新！
```

**数学表示：**
```
θ_new = θ_old - lr × (∇θ_1 + ∇θ_2 + ∇θ_3) / K
```

---

## 六、反向传播与参数更新

### 6.1 反向传播

**链式法则：**
```
∂L/∂θ = ∂L/∂logits × ∂logits/∂h × ∂h/∂θ
```

对于LoRA，我们只更新W_A和W_B的参数：
```
∂L/∂W_A = ∂L/∂ΔW × ∂ΔW/∂W_A
        = ∂L/∂(W_B·W_A) · W_B^T
        
∂L/∂W_B = ∂L/∂ΔW × ∂ΔW/∂W_B
        = W_A^T · ∂L/∂(W_B·W_A)
```

### 6.2 梯度裁剪

防止梯度爆炸：
```
g = ∇θ  # 当前梯度
g_norm = ||g||_2 = √(Σ g_i²)

if g_norm > max_norm:
    g_clipped = g × (max_norm / g_norm)
else:
    g_clipped = g
```

**数学公式：**
```
g_clipped = g × min(1, max_norm / ||g||_2)
```

### 6.3 AdamW优化器

AdamW = Adam + Weight Decay

**算法流程：**
```
输入: 学习率 lr, 权重衰减 wd, 动量参数 β₁, β₂, 梯度 g

# 偏置校正
m_hat = m / (1 - β₁^t)
v_hat = v / (1 - β₂^t)

# 参数更新
θ = θ - lr × m_hat / (√v_hat + ε) - wd × θ
```

**逐项解释：**

1. **动量估计（类似物理中的惯性）：**
   ```
   m_t = β₁ × m_{t-1} + (1 - β₁) × g
   ```
   - β₁通常设为0.9
   - 作用：使更新更加平滑，减少震荡

2. **方差估计（自适应性学习率）：**
   ```
   v_t = β₂ × v_{t-1} + (1 - β₂) × g²
   ```
   - β₂通常设为0.999
   - 作用：对频繁更新的参数用小学习率，稀疏参数用大学习率

3. **偏置校正（消除初始化偏差）：**
   ```
   m_hat = m_t / (1 - β₁^t)
   v_hat = v_t / (1 - β₂^t)
   ```
   - t是迭代次数
   - 作用：校正初期m和v接近0的问题

4. **最终更新：**
   ```
   θ_{t+1} = θ_t - lr × m_hat / (√v_hat + ε) + wd × θ_t
          = θ_t - lr × m_hat / (√v_hat + ε) - lr × wd × θ_t
   ```
   - ε通常设为1e-8，防止除零

### 6.4 完整的参数更新步骤

```python
# 1. 梯度缩放（用于混合精度）
scaled_loss = loss / K
scaled_loss.backward()  # grad自动累积

# 2. 达到累积步数时
if (step + 1) % K == 0:
    # 取消梯度缩放
    scaler.unscale_(optimizer)
    
    # 梯度裁剪
    g = get_parameter_grads(optimizer.param_groups)
    g_norm = torch.norm(g)
    if g_norm > max_norm:
        g = g * (max_norm / g_norm)
    
    # 参数更新
    scaler.step(optimizer)
    
    # 更新scaler状态
    scaler.update()
    
    # 清零梯度
    optimizer.zero_grad(set_to_none=True)
```

---

## 七、学习率调度

### 7.1 余弦退火（Cosine Annealing）

代码中使用的学习率调度器：
```
lr(t) = lr_max × [0.1 + 0.45 × (1 + cos(π × t / T_total))]
```

**参数说明：**
- t：当前总步数 = epoch × steps_per_epoch + step
- T_total：总步数 = epochs × steps_per_epoch
- lr_max：最大学习率（如1e-4）

**学习率曲线：**
```
lr
↑
lr_max×0.55                     ╭─────────────────╮
                              ╱                   ╲
lr_max×0.1                 ╱                       ╲
                         ╱                         ╲
________________________╱___________________________╲→
0                 t/T_total                  1

- 开始时：lr = 0.55 × lr_max
- 中间：lr平滑下降
- 结束：lr = 0.1 × lr_max
```

### 7.2 Warmup（学习率预热）

通常在训练开始时，会先逐渐增大学习率：
```
if t < warmup_steps:
    lr = lr_max × (t / warmup_steps)
else:
    lr = lr_max × [0.1 + 0.45 × (1 + cos(π × (t-warmup_steps) / (T_total-warmup_steps)))]
```

---

## 八、混合精度训练

### 8.1 为什么需要混合精度

| 精度        | 显存占用      | 计算速度 |
| ----------- | ------------- | -------- |
| FP32 (32位) | 4 bytes/param | 1×       |
| FP16 (16位) | 2 bytes/param | 2-3×     |
| BF16 (16位) | 2 bytes/param | 2-3×     |

**FP16的问题：** 精度范围较小，容易下溢（underflow）

```
FP32:     1.2e-45 ~ 3.4e38
FP16:     6.0e-5 ~ 65504
BF16:     1.0e-45 ~ 3.4e38
```

### 8.2 GradScaler的原理

**问题：** 当梯度很小时（如1e-7），FP16会变成0（下溢）

**解决方案：** 放大梯度→反向传播→缩小回原始scale

```
# 前向传播（自动）
with autocast():
    output = model(input)
    loss = loss_fn(output, target)

# 反向传播（放大梯度）
scaled_loss = loss / G
scaled_loss.backward()  # 梯度实际上放大了G倍

# 参数更新前（缩小梯度）
optimizer.step()  # scaler自动处理缩放
scaler.update()  # 调整缩放因子G
```

**缩放因子G的动态调整：**
```
if has_inf_or_nan梯度:
    G = G / 2  # 缩小scale，下次重试
else:
    G = G * 2  # 尝试放大scale
```

---

## 九、检查点保存与加载

### 9.1 保存的内容

```python
# 只保存LoRA权重（轻量）
save_lora(model, path)

# 保存完整训练状态
checkpoint = {
    'epoch': epoch,
    'step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'lr_scheduler_state': ...,
}
```

### 9.2 加载断点续训

```python
# 检查是否有断点
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scaler.load_state_dict(checkpoint['scaler_state_dict'])

start_epoch = checkpoint['epoch']
start_step = checkpoint['step']
```

---

## 十、一个Epoch的完整流程

### 10.1 伪代码

```
输入: 
  - epoch: 当前轮次
  - loader: 数据加载器 (共T个batch)
  - K: 梯度累积步数
  - max_norm: 梯度裁剪阈值

初始化:
  start_time = 当前时间
  accumulated_grad = 0

主循环: for step in range(T):
  
  # ======== Step 1: 数据加载 ========
  input_ids, labels = next(loader)  # 形状: (B, L)
  input_ids = input_ids.to(device)
  labels = labels.to(device)
  
  # ======== Step 2: 学习率更新 ========
  t = epoch × T + step
  lr = lr_max × [0.1 + 0.45 × (1 + cos(π×t/T_total))]
  for param_group in optimizer.param_groups:
      param_group['lr'] = lr
  
  # ======== Step 3: 前向传播 ========
  with autocast():
      logits = model(input_ids)
      loss_main = CrossEntropyLoss(logits, labels)
      loss_aux = λ × (||W_A||² + ||W_B||²)
      loss = (loss_main + loss_aux) / K
  
  # ======== Step 4: 反向传播 ========
  scaler.scale(loss).backward()
  
  # ======== Step 5: 梯度累积与更新 ========
  if (step + 1) % K == 0:
      # 取消梯度缩放
      scaler.unscale_(optimizer)
      
      # 梯度裁剪
      g_norm = ||g||
      if g_norm > max_norm:
          g = g × (max_norm / g_norm)
      
      # 参数更新
      scaler.step(optimizer)
      scaler.update()
      
      # 清零梯度
      optimizer.zero_grad(set_to_none=True)
  
  # ======== Step 6: 日志记录 ========
  if step % log_interval == 0:
      loss_display = loss.item() × K  # 恢复真实损失
      eta = 预计剩余时间
      print(f"step={step}, loss={loss_display:.4f}, lr={lr:.8f}")
  
  # ======== Step 7: 保存检查点 ========
  if step % save_interval == 0:
      save_lora(model, lora_path)
      save_checkpoint(model, optimizer, epoch, step, ckpt_path)
  
  # ======== Step 8: 清理内存 ========
  del input_ids, labels, loss, logits

一个Epoch结束！
```

### 10.2 数学流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         单个Step的完整流程                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DataLoader: (input_ids, labels)                                   │
│       │                                                             │
│       ↓                                                             │
│  移动到GPU: input_ids→cuda, labels→cuda                           │
│       │                                                             │
│       ↓                                                             │
│  计算学习率: lr(t) = lr_max × [0.1 + 0.45×(1+cos(πt/T))]           │
│       │                                                             │
│       ↓                                                             │
│  前向传播: logits = Model(input_ids; Θ_frozen, Θ_lora)            │
│       │                                                             │
│       ↓                                                             │
│  计算损失: L = (L_main + L_aux) / K                                │
│       │                                                             │
│       ↓                                                             │
│  反向传播: grad = ∂L/∂Θ_lora  (梯度自动累积到参数)                  │
│       │                                                             │
│       ↓                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ if (step + 1) % K == 0:                                      │   │
│  │     # 梯度裁剪                                                │   │
│  │     grad = grad × min(1, max_norm/||grad||)                 │   │
│  │                                                             │   │
│  │     # 参数更新 (AdamW)                                       │   │
│  │     m = β₁×m + (1-β₁)×grad                                  │   │
│  │     v = β₂×v + (1-β₂)×grad²                                 │   │
│  │     θ = θ - lr×m/(√v+ε) - wd×θ                              │   │
│  │                                                             │   │
│  │     # 清零梯度                                               │   │
│  │     grad = 0                                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ↓                                                             │
│  记录日志 / 保存检查点 / 清理内存                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 十一、代码对照

### 11.1 原始代码

```python
def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 数据到GPU
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        
        # 学习率更新
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 前向传播
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # 日志记录
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            # ...
        
        # 检查点保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            save_lora(model, lora_save_path)
            lm_checkpoint(...)
            model.train()
        
        # 内存清理
        del input_ids, labels, res, loss
```

### 11.2 与数学公式对应

| 代码                                   | 对应数学公式                                | 说明           |
| -------------------------------------- | ------------------------------------------- | -------------- |
| `lr = get_lr(...)`                     | lr(t) = lr_max × [0.1 + 0.45×(1+cos(πt/T))] | 余弦退火学习率 |
| `loss = (res.loss + res.aux_loss) / K` | L = (L_main + L_aux) / K                    | 梯度累积缩放   |
| `scaler.scale(loss).backward()`        | grad += ∂L/∂θ                               | 缩放后反向传播 |
| `scaler.unscale_(optimizer)`           | -                                           | 取消缩放       |
| `clip_grad_norm_(..., 1.0)`            | g = g × min(1, N/                           |                | g |  | ) | 梯度裁剪 |
| `scaler.step(optimizer)`               | θ = θ - lr×m/(√v+ε) - wd×θ                  | AdamW更新      |

---

## 十二、常见问题

### Q1: 梯度累积和batch size是什么关系？

**答：** 有效batch size = batch_size × accumulation_steps

例如：
- batch_size = 4
- accumulation_steps = 3
- 有效batch size = 12

### Q2: 为什么需要梯度裁剪？

**答：** 防止梯度爆炸。当loss突然增大时，梯度可能变得非常大，导致参数更新完全偏离方向。梯度裁剪将梯度限制在一个合理范围内。

### Q3: LoRA的r值越大越好吗？

**答：** 不是。r越大，参数量越大，训练越慢，但表达能力更强。一般从r=8开始尝试。

### Q4: 为什么使用余弦退火学习率？

**答：** 
1. 初期用较大学习率快速收敛
2. 后期用较小学习率精细调优
3. 余弦曲线平滑，比step decay更稳定

### Q5: 混合精度训练会不会影响模型精度？

**答：** 不会。FP16/BF16对于深度学习训练足够精确，GradScaler解决了下溢问题，最终效果与FP32相当。

### Q6: save_lora和lm_checkpoint有什么区别？

**答：**
- save_lora：只保存LoRA的W_A和W_B矩阵（可插拔切换）
- lm_checkpoint：保存完整训练状态（断点续训用）

---

## 总结

训练一个LoRA epoch的核心公式：

| 步骤   | 公式                                        |
| ------ | ------------------------------------------- |
| 输入   | x ∈ ℤ^{B×L}                                 |
| 前向   | logits = Model(x; Θ_frozen, Θ_lora)         |
| 损失   | L = (CrossEntropy(logits, y) + λ            |  | W_A |  | ² + λ |  | W_B |  | ²) / K |
| 梯度   | g = ∂L/∂Θ_lora (累积K次)                    |
| 裁剪   | g_clipped = g × min(1, N/                   |  | g   |  | )     |
| 更新   | Θ_lora ← Θ_lora - lr × AdamW(g_clipped)     |
| 学习率 | lr(t) = lr_max × [0.1 + 0.45×(1+cos(πt/T))] |

希望这份文档能帮助你完全理解LoRA训练流程！