# LLM 两个核心问题高度总结

## 目录
1. [问题一：为什么训练数据是 `input_ids/labels`，而不是 768 维 embedding？](#问题一为什么训练数据是-input_idslabels而不是-768-维-embedding)
2. [问题二：多层 Transformer 里的 Q/K/V 是共享的吗？](#问题二多层-transformer-里的-qkv-是共享的吗)
3. [一张总图：从 token 到 loss 的完整链路](#一张总图从-token-到-loss-的完整链路)
4. [最小例子：把两个问题放到同一个训练流程里看](#最小例子把两个问题放到同一个训练流程里看)
5. [一句话速记版](#一句话速记版)

---

# 问题一：为什么训练数据是 `input_ids/labels`，而不是 768 维 embedding？

## 1. 本质结论

**训练数据在进入模型之前，仍然是离散 token 的编号（token IDs），不是 embedding 向量。**

也就是说，数据集通常长这样：

```text
input_ids : (B, L)
labels    : (B, L)
```

其中每个元素都是一个整数，表示词表中的某个 token 编号。

而 embedding 是模型内部做的第一步变换：

\[
\text{token id} \rightarrow \text{embedding vector} \in \mathbb{R}^{768}
\]

所以：

- `input_ids` 是**离散索引**
- embedding 是**模型内部的连续表示**
- `labels` 仍然是**离散索引**，因为训练目标是“预测下一个 token 是哪个编号”

---

## 2. 数学主线

设：

- 词表大小为 \(V\)
- embedding 维度为 \(d = 768\)
- embedding 矩阵为

\[
E \in \mathbb{R}^{V \times d}
\]

对于一个 token id \(x\in\{0,1,\dots,V-1\}\)，其 embedding 为

\[
\mathrm{Emb}(x)=E[x] \in \mathbb{R}^{d}
\]

也就是说，**本质上是从 embedding 矩阵的第 \(x\) 行取出一个长度为 768 的向量**。

对于一个 batch 的输入：

\[
\text{input\_ids} \in \mathbb{Z}^{B\times L}
\]

经过 embedding lookup 之后：

\[
H^{(0)} \in \mathbb{R}^{B\times L\times d}
\]

即

\[
(B,L) \xrightarrow{\text{Embedding}} (B,L,768)
\]

---

## 3. 为什么说它“本质上等价于 one-hot 查表”？

对于 token \(x\)，可以构造一个 one-hot 向量 \(e_x\in\mathbb{R}^V\)：

\[
(e_x)_i=
\begin{cases}
1, & i=x \\
0, & i\neq x
\end{cases}
\]

那么 embedding 可以写成

\[
e_x^T E = E[x]
\]

所以：

\[
\boxed{\text{Embedding lookup} \equiv \text{one-hot} \times E}
\]

但在工程实现里，**不会真的先生成一个几万维的 one-hot 向量**，因为那样太浪费内存。实际做法是直接按索引取矩阵的某一行。

---

## 4. 为什么 `labels` 不是 768 维向量？

因为训练目标不是“拟合一个 embedding 向量”，而是：

> 在整个词表 \(V\) 个 token 中，判断下一个 token 是哪一个。

所以模型最后输出的是：

\[
\text{logits} \in \mathbb{R}^{B\times L\times V}
\]

即对每个位置给出一个长度为 \(V\) 的分数向量。

然后与真实标签比较：

\[
\text{label}_{i,j} \in \{0,1,\dots,V-1\}
\]

也就是说，label 是**类别编号**，不是 768 维向量。

使用交叉熵损失：

\[
\mathcal{L}_{i,j} = -\log p_{i,j}[y_{i,j}]
\]

其中：

- \(p_{i,j}\) 是 softmax 后得到的长度为 \(V\) 的概率分布
- \(y_{i,j}\) 是真实 token 的编号

所以训练目标空间是：

\[
\{0,1,\dots,V-1\}
\]

而不是：

\[
\mathbb{R}^{768}
\]

---

## 5. 例子 1：最小 token 序列例子

假设一句话 tokenized 后是：

```text
[101, 2054, 2003, 1996, 102]
```

在自回归语言模型中，通常构造：

```text
input_ids = [101, 2054, 2003, 1996]
labels    = [2054, 2003, 1996, 102]
```

含义是：

- 给模型 `101`，让它预测 `2054`
- 给模型 `101,2054`，让它预测 `2003`
- 给模型 `101,2054,2003`，让它预测 `1996`
- 给模型 `101,2054,2003,1996`，让它预测 `102`

所以 `labels` 始终是“下一个 token 的 ID”。

---

## 6. 例子 2：加入 embedding 之后的维度变化

若：

- batch size \(B=2\)
- sequence length \(L=4\)
- embedding dim \(d=768\)

则：

```text
input_ids.shape = (2, 4)
```

例如：

```text
input_ids =
[
  [101, 2054, 2003, 1996],
  [101, 1997, 2019, 2116]
]
```

经过 embedding：

```text
hidden_states.shape = (2, 4, 768)
```

即每一个整数 token，都变成一个长度为 768 的向量。

---

# 问题二：多层 Transformer 里的 Q/K/V 是共享的吗？

## 1. 本质结论

**标准 Transformer 中，不同层的 Q/K/V 权重通常是彼此独立的，不共享。**

也就是说，如果一个模型有 16 层，那么第 1 层和第 2 层使用的是不同的 Q/K/V 投影矩阵：

\[
W_q^{(1)} \neq W_q^{(2)},\quad
W_k^{(1)} \neq W_k^{(2)},\quad
W_v^{(1)} \neq W_v^{(2)}
\]

同理一直到第 16 层。

---

## 2. 数学主线

设第 \(l\) 层输入为

\[
H^{(l)} \in \mathbb{R}^{B\times T\times d}
\]

该层的 Q/K/V 由三个线性映射得到：

\[
Q^{(l)} = H^{(l)} W_q^{(l)}
\]

\[
K^{(l)} = H^{(l)} W_k^{(l)}
\]

\[
V^{(l)} = H^{(l)} W_v^{(l)}
\]

其中

\[
W_q^{(l)},W_k^{(l)},W_v^{(l)} \in \mathbb{R}^{d\times d}
\]

如果模型有 16 层，则存在 16 套不同的参数：

\[
\{W_q^{(l)},W_k^{(l)},W_v^{(l)}\}_{l=1}^{16}
\]

---

## 3. 为什么通常不共享？

因为每一层都希望学习**不同层次的表示**。

直观上：

- 底层可能更偏向局部模式、词法信息
- 中层可能更偏向句法关系
- 高层可能更偏向语义整合、任务相关信息

如果所有层共用同一套 Q/K/V 参数，那么模型相当于反复应用同一个变换，表达能力会受限。

因此标准设计更常见的是：

\[
\boxed{\text{每层独立学习自己的 Q/K/V 投影}}
\]

---

## 4. 同一层内部，不同 head 共享吗？

通常也**不共享**。

多头注意力中，设 head 数为 \(h\)，则每个 head 都对应自己的投影子空间。即使在实现上常把它们拼成一个大矩阵，本质上不同 head 对应的是不同参数块。

所以：

- **不同层之间：不共享**
- **同一层不同 head：通常也不共享**

---

## 5. 例子 3：16 层模型的参数结构

假设模型为：

- 16 层 Transformer
- hidden size \(d=768\)

则第 1 层有：

\[
W_q^{(1)}, W_k^{(1)}, W_v^{(1)}
\]

第 2 层有：

\[
W_q^{(2)}, W_k^{(2)}, W_v^{(2)}
\]

... ...

第 16 层有：

\[
W_q^{(16)}, W_k^{(16)}, W_v^{(16)}
\]

这些通常都是**不同的参数张量**。

---

## 6. 一个反例：有些特殊模型会共享

有些轻量化模型（例如 ALBERT 一类设计）会做跨层参数共享，以减少参数量。

这种情况属于：

\[
W_q^{(1)} = W_q^{(2)} = \cdots = W_q^{(L)}
\]

但这**不是标准大多数 LLM 的默认设置**。

所以你在学习一般 LLM（如常规 decoder-only Transformer）时，默认理解为：

> **每层独立，不共享。**

---

## 7. LoRA 和 Q/K/V 的关系

LoRA 并不是改变 Q/K/V 的“是否共享”，而是在某些线性层上加一个低秩增量：

\[
W_q^{(l)} \leftarrow W_q^{(l)} + A_q^{(l)} B_q^{(l)}
\]

其中：

\[
A_q^{(l)} \in \mathbb{R}^{d\times r},
\quad
B_q^{(l)} \in \mathbb{R}^{r\times d}
\]

同样，这个增量通常也是**每层独立**的，而不是全层共享。

所以 LoRA 的典型情况仍然是：

- 第 1 层有自己的 LoRA 参数
- 第 2 层有自己的 LoRA 参数
- ...
- 第 16 层有自己的 LoRA 参数

---

# 一张总图：从 token 到 loss 的完整链路

现在把两个问题合并起来看。

## 1. 输入还是 token ID

\[
\text{input\_ids} \in \mathbb{Z}^{B\times L}
\]

例如：

```text
input_ids =
[
  [101, 2054, 2003, 1996],
  [101, 1997, 2019, 2116]
]
```

---

## 2. embedding 层：离散索引变成连续向量

\[
(B,L) \xrightarrow{\text{Embedding}} (B,L,768)
\]

即：

\[
H^{(0)} = E[\text{input\_ids}]
\]

---

## 3. 经过多层 Transformer

对于第 \(l\) 层：

\[
Q^{(l)} = H^{(l)}W_q^{(l)},
\quad
K^{(l)} = H^{(l)}W_k^{(l)},
\quad
V^{(l)} = H^{(l)}W_v^{(l)}
\]

其中每一层的 \(W_q^{(l)},W_k^{(l)},W_v^{(l)}\) 一般都独立。

层与层之间维度通常保持：

\[
(B,L,768) \to (B,L,768)
\]

---

## 4. 输出层投影到词表

最后隐藏状态为：

\[
H^{(L)} \in \mathbb{R}^{B\times L\times 768}
\]

再乘输出矩阵：

\[
W_o \in \mathbb{R}^{768\times V}
\]

得到：

\[
\text{logits} \in \mathbb{R}^{B\times L\times V}
\]

---

## 5. 与 labels 做交叉熵

labels 仍然是：

\[
\text{labels} \in \mathbb{Z}^{B\times L}
\]

每个位置是一个正确 token 的索引。

所以整个过程是：

\[
(B,L)
\xrightarrow{\text{Embedding}}
(B,L,768)
\xrightarrow{\text{16 层 Transformer}}
(B,L,768)
\xrightarrow{\text{LM Head}}
(B,L,V)
\xrightarrow{\text{Cross Entropy with labels}}
\text{loss}
\]

---

# 最小例子：把两个问题放到同一个训练流程里看

假设：

- 词表大小 \(V=50000\)
- hidden size \(d=768\)
- 层数 16
- batch size \(B=1\)
- 序列长度 \(L=4\)

输入：

```text
input_ids = [10, 25, 7, 88]
labels    = [25, 7, 88, 103]
```

## 第一步：embedding

```text
[10, 25, 7, 88]
```

变成

```text
[
  e_10,
  e_25,
  e_7,
  e_88
]
```

其中每个 \(e_i\in\mathbb{R}^{768}\)。

因此张量形状从：

```text
(1, 4)
```

变成：

```text
(1, 4, 768)
```

## 第二步：第 1 层 attention

第 1 层使用自己的：

- \(W_q^{(1)}\)
- \(W_k^{(1)}\)
- \(W_v^{(1)}\)

计算出第 1 层的 attention 输出。

## 第三步：第 2 层 attention

第 2 层重新使用自己的：

- \(W_q^{(2)}\)
- \(W_k^{(2)}\)
- \(W_v^{(2)}\)

注意这里不是继续复用第 1 层的参数，而是用一套新参数。

## 第四步：一直到第 16 层

最终得到：

```text
hidden_states.shape = (1, 4, 768)
```

## 第五步：投影到词表

输出：

```text
logits.shape = (1, 4, 50000)
```

对每个位置，模型都给出 50000 个 token 的分数。

## 第六步：与 label 比较

真实 label 是：

```text
[25, 7, 88, 103]
```

所以模型不是去拟合某个 768 维 embedding，而是在每个位置做一个“50000 分类问题”。

---

# 一句话速记版

## 问题一速记

> 训练数据最开始是 token ID，不是 embedding；embedding 是模型内部查表后得到的 768 维向量；labels 仍然是 token ID，因为训练目标是“预测下一个 token 的类别”。

## 问题二速记

> 标准 Transformer 中，每一层都有自己独立的 Q/K/V 参数，不共享；LoRA 也是在这些层各自的线性层上加低秩增量，通常仍然是每层独立。

---

# 最终统一理解

你可以把整个 LLM 看成下面这条主线：

\[
\boxed{
\text{token IDs}
\rightarrow
\text{embedding vectors}
\rightarrow
\text{multi-layer Transformer (each layer has its own Q/K/V)}
\rightarrow
\text{vocab logits}
\rightarrow
\text{cross-entropy against token-ID labels}
}
\]

也就是说：

1. **输入空间**是离散 token ID；
2. **内部计算空间**是 768 维连续向量空间；
3. **输出目标空间**仍然是离散词表类别空间；
4. **多层 Transformer 的参数一般逐层独立，不共享。**
