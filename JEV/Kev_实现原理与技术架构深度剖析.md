# Kev 决策模型实现原理与技术架构深度剖析：从 Block-Causal Mask 到 Pointer Head 的全链路复刻

---

## 目录
1. [一、 概述与核心定位](#一-概述与核心定位)
   - 1.1 什么是 Kev？
   - 1.2 为什么传统自回归 LLM 难以胜任精准决策？
   - 1.3 Kev 的破局思路：Prefill-Only 决策模型
2. [二、 Kev 总体架构全景](#二-kev-总体架构全景)
   - 2.1 整体架构拓扑图
   - 2.2 核心设计哲学：结构化控制与纯粹表示学习
3. [三、 核心技术原理与代码级实现深度拆解](#三-核心技术原理与代码级实现深度拆解)
   - 3.1 输入流编码与控制标记防伪（Tokenization & Delimiter Sanitization）
   - 3.2 分支注意力与状态共享（Block-Causal Masking 机制）
   - 3.3 混合注意力架构（Qwen3.5 DeltaNet）的破局之道：Row-Form 物理分支
   - 3.4 指针读出网络（Pointer Readout Head）的设计与概率求解
   - 3.5 三大决策原语的统一映射（Choice / Noul / Score）
4. [四、 训练策略与损失函数设计](#四-训练策略与损失函数设计)
   - 4.1 训练数据构造与无缝推理对齐
   - 4.2 损失函数组合（Cross-Entropy / Permutation KL / RPS 有序惩罚）
   - 4.3 LoRA 适配器靶向注入（针对 Attention 与 Linear Attention 算子）
5. [五、 推理服务引擎与高性能工程优化](#五-推理服务引擎与高性能工程优化)
   - 5.1 零生成开销与 Prefill-Only 运行流水线
   - 5.2 LoRA 权重离线融合（fp32 Merge Before Cast）
   - 5.3 状态前缀 KV Cache（State Prefix LRU）
   - 5.4 跨平台推理适配：MPS (Shape Bucket) 与 CUDA (FLA)
   - 5.5 后处理概率校准与温度缩放（Temperature Scaling）
6. [六、 工业实战：AICO 技能选择（Skill Selection）基准评测](#六-工业实战aico-技能选择skill-selection基准评测)
   - 6.1 场景背景与 20 题业务基准构建
   - 6.2 Kev-4B vs. 官方 JEV vs. 通用大模型实测对比
7. [七、 总结与架构演进展望](#七-总结与架构演进展望)

---

## 一、 概述与核心定位

### 1.1 什么是 Kev？

**Kev** 是由 Jared Palmer（前 Vercel 核心成员、Formik/Turborepo 架构师）主导开源的**轻量级、自托管决策模型家族（Decision Models）**。Kev 的诞生直接对标并逆向复刻了商业决策模型 **TypeSafe Jev（System One）** 的内部架构，模型开源规格涵盖 **0.8B、4B、9B**（基于 Qwen3.5 底座）以及上一代 **0.6B、4B、8B**（基于 Qwen3 底座）。

在代码工程中，Kev 实现了以下三个维度的无缝平替：
* **协议平替**：100% 兼容 TypeSafe 的 System One REST API（`POST /v1/systemone`），现有业务代码可无缝使用官方 `typesafe-sdk`；
* **能力平替**：原生支持 `Choice`（多选分类/路由）、`Noul`（二元判定）、`Score`（连续/有序等级评分）三大决策原语；
* **自主可控**：开放全套预训练权重、微调训练代码、评估套件与 Web Playground，支持在企业私有云与本地 Apple Silicon / CUDA 环境下私有化部署。

```
                       ┌───────────────────────────────┐
                       │     TypeSafe Jev (商业闭源)     │
                       │   • 闭源托管 API              │
                       │   • 缺乏私有微调通道           │
                       │   • 产生商用 token/调用费用     │
                       └───────────────┬───────────────┘
                                       │ 架构解密 & 逆向重构
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Kev (完全开源复刻)                              │
│  • 架构重构: Prefill-Only + Block-Causal Mask + Pointer Readout Head        │
│  • 开放生态: 权重开放 (0.8B / 4B / 9B) + 训练代码 + 本地 FastAPI 服务         │
│  • 原生兼容: 协议与 typesafe-sdk 零修改对接，支持本地微调私有分类规则         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 为什么传统自回归 LLM 难以胜任精准决策？

在 Agent 工具调用、意图路由、风险拦截等控制流场景中，软件系统真正需要的并不是自由文本，而是一个**确定性的离散选项标签**或**严格校准的概率值**。使用传统自回归生成式大模型（如 GPT-4, Qwen-27B-Instruct 等）来做分类与决策，在工程上面临四大致命痛点：

1. **自回归解码的延迟瓶颈（Latency Overhead）**：
   传统 LLM 必须按 Token 逐个自回归解码输出（如输出 `{"decision": "query_param"}`）。生成数十个 Token 通常需要数百毫秒至数秒，对于作为系统底层路由器的高频决策任务而言，延迟开销过大。
2. **输出结构脆弱与解析失败（Parsing Fragility）**：
   即使开启 JSON Schema 或约束解码，模型依然存在生成非法键名、附加多余解释、甚至拒绝回答的概率，导致上游业务代码抛出解析异常。
3. **Softmax 假概率与过度自信（Overconfidence）**：
   从生成式大模型的最后一个 Token Logits 取出的 Softmax 概率通常极不平准（例如输出 99% 的置信度，真实正确率仅 70%）。缺乏统计学校准的概率无法作为自动化业务熔断的可靠阈值。
4. **选项换序敏感性（Permutation Bias）**：
   选项的物理排列顺序（A/B/C/D）会对大模型的选择造成高达 10%~30% 的预测漂移。

### 1.3 Kev 的破局思路：Prefill-Only 决策模型

Kev 彻底改变了模型推理的计算范式：
* **不生成一个 Token（Never generates a token）**：彻底剔除大模型的词表预测头（`lm_head`），模型前向计算仅执行 Prefill（预填充）阶段；
* **单次前向传播直达结果**：仅需一次 Backbone Forward，即可通过特殊的特征读出头直接输出数学上严格归一化的概率分布；
* **前缀共享与并行评估**：针对一份文本（State），可以在同一次 Forward 中并行挂载评估十几个独立问题，而问题之间保持严格的注意力隔离。

---

## 二、 Kev 总体架构全景

### 2.1 整体架构拓扑图

下图清晰展示了一个包含 1 个状态（State）和 2 个决策问题（Question 1 为 Choice，Question 2 为 Noul）的请求在 Kev 中的端到端数据流向与计算拓扑：

```
========================================================================================================
                                     Kev 端到端模型架构与计算流向
========================================================================================================

【输入文本】
State (材料文本) : "Shoes arrived two weeks late and in the wrong size. Also two charges on my card."
Q1 (Choice 路由) : "Which team?" -> [returns: "Exchanges/refunds", shipping: "Delays", billing: "Charges"]
Q2 (Noul 判定)   : "Is this urgent?"
--------------------------------------------------------------------------------------------------------
                                                ▼  
【Token 序列化 & 控制标记包装】 (model.py: encode)
[SPECIAL_0] + State_Tokens 
  + [Q] + Q1_Instr + [OPT] returns [C] + [OPT] shipping [C] + [OPT] billing [C] + [DECIDE]
  + [Q] + Q2_Instr + [OPT] true [C]    + [OPT] false [C]                        + [DECIDE]
--------------------------------------------------------------------------------------------------------
                                                ▼  
【Position IDs 拓扑编排】 (位置重置机制)
State:     0,  1,  2, ..., Ls-1
Branch 1:  Ls, Ls+1, ..., Ls + len(br1) - 1
Branch 2:  Ls, Ls+1, ..., Ls + len(br2) - 1   <-- 关键: 各分支位置索引自 State 末尾独立累加!
--------------------------------------------------------------------------------------------------------
                                                ▼  
【4D Block-Causal Attention Mask】 (model.py: branch_mask)
                   State     Q1-Tokens   Q2-Tokens
        State   [  Causal  |     0     |     0     ]   <- State 仅见自身 (因果)
        Q1      [ Full_Vis |  Causal   |     0     ]   <- Q1 见完整 State + 自身 (不可见 Q2!)
        Q2      [ Full_Vis |     0     |  Causal   ]   <- Q2 见完整 State + 自身 (不可见 Q1!)
--------------------------------------------------------------------------------------------------------
                                                ▼  
【Transformer 骨干网络 (Qwen3 / Qwen3.5 + LoRA)】
前向计算，抽取各 Token 最终 Hidden State: H ∈ ℝ^(L × d)
--------------------------------------------------------------------------------------------------------
                                                ▼  
【Pointer Head 指针读出】 (model.py: PointerHead)
针对 Q1:
  h_decide = H[idx_<decide>_Q1]  ∈ ℝ^d
  h_opts   = [H[idx_</opt>_ret], H[idx_</opt>_ship], H[idx_</opt>_bill]] ∈ ℝ^(3 × d)
  
  Logits = (W_k · h_opts) @ (W_q · h_decide) / sqrt(dp)
  Probabilities = Softmax(Logits)  ===>  {returns: 0.47, shipping: 0.28, billing: 0.25}
--------------------------------------------------------------------------------------------------------
                                                ▼  
【API 响应组装】 (api.py)
输出类型安全的结构体，包含预测值 (Choice)、后验概率分布 (Probabilities) 与统计置信度 (Confidence)
========================================================================================================
```

### 2.2 核心设计哲学：结构化控制与纯粹表示学习

1. **零词表依赖**：输出完全脱离语言模型的 Token 词表。即使一个类别是新奇的业务字符串（如 `__no_skill__`），模型也是在选项描述符的隐层语义上做相似度度量，不需要词表中存在该 Token。
2. **多题并行计算开销恒定**：State 无论多么冗长（如 2000 Token 的工单上下文），在注意力机制中只需要计算一次 Key/Value，挂载 1 个问题和挂载 10 个问题所带来的算力增量极其微小。
3. **真概率输出**：直接输出浮点后验概率分布，赋予外部调用方灵活设定业务置信度阈值的能力。

---

## 三、 核心技术原理与代码级实现深度拆解

### 3.1 输入流编码与控制标记防伪（Tokenization & Delimiter Sanitization）

在决策模型中，控制标记（Delimiter Tokens）起到界定语义边界的决定性作用：
* `<state>`：材料开始
* `<q>`：问题开始
* `<opt>`：候选选项开始
* `</opt>`：候选选项结束（**极其关键，此处汇聚该选项的所有语义**）
* `<decide>`：决策指令点（**全问题最终汇聚点**）

#### 巧妙复用现有控制标记，避免破坏 Embedding 层
在主流框架中，新增 Special Token 通常需要扩充词表（`resize_token_embeddings`），这会导致未初始化的 Embedding 权重、破坏预训练特征分布，并在 PyTorch MPS / CUDA 内存对齐上带来潜在显存泄露。
Kev 在 [`kev/kev/model.py`](file:///Users/zhangfan/project/jev/kev/kev/model.py#L10) 中采用了一种极其实用的工程策略——**借用 Qwen 预训练底座中极少被用到的已有特殊 Token**：
```python
# 复用已有的 FIM / Box Token，无需扩充 Embedding 行
SPECIAL = [
    "<|fim_prefix|>",  # 映射为 State 起始标记
    "<|fim_middle|>",  # 映射为 Question 起始标记 (<q>)
    "<|box_start|>",   # 映射为 Option 起始标记 (<opt>)
    "<|box_end|>",     # 映射为 Option 结束标记 (</opt>)
    "<|fim_suffix|>",  # 映射为 Decide 触发标记 (<decide>)
]
```
通过 LoRA 适配器训练，网络在微调过程中自然学会了这些特殊标记在决策语境下的路由含义，完全不需要修改基座词表。

#### 严密的安全防御：防止 Prompt 注入伪造边界
如果用户输入的文本恶意包含 `<|box_end|>` 或 `<|fim_suffix|>`，传统的简单拼接会导致选项边界被攻击者强行闭合。Kev 在 [`user_tokens`](file:///Users/zhangfan/project/jev/kev/kev/model.py#L21-L24) 中实现了防御性字符清洗：
```python
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")

def user_tokens(tok, text):
    """
    清洗用户输入的文本，将任何类似 <|name|> 的控制序列改写为全角形态 <¦name¦>，
    彻底阻断用户文本冒充内部控制标记的可能性。
    """
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids
```

#### 编码数据结构与位置重置机制
在 [`encode`](file:///Users/zhangfan/project/jev/kev/kev/model.py#L30-L68) 函数中，Kev 负责将单条样本打包成扁平化的 1D 数组，并生成严密的段标识与重置位置编码：
```python
# ids:  所有 Token ID 拼成的长序列
# seg:  段标签: 0 代表 State, 1..Q 代表第 1 到第 Q 个 Question
# pos:  Position IDs，这是实现分支隔离的关键之一！
#       State 占据 0 .. Ls-1
#       所有 Question 分支的 position 全部重置，统一从 Ls 开始递增！
```
这种设计在数学上让每个 Question 分支在相对位置上与 State 保持完全一致的几何距离，从位置编码层面抹除了分支之间的先后偏好。

---

### 3.2 分支注意力与状态共享（Block-Causal Masking 机制）

对于基于纯注意力（Pure Self-Attention）的架构（如 Qwen3 系列），Kev 通过一个精心设计的 4D Additive Attention Mask 实现了**状态共享 + 分支隔离**。

其核心实现位于 [`branch_mask_batch`](file:///Users/zhangfan/project/jev/kev/kev/model.py#L75-L101)：
```python
def branch_mask_batch(segs, device, dtype=torch.float32, opts=None, length=None):
    # s[b, :] 存储了各个 token 属于哪个段 (0 为 state, 1..Q 为问题编号)
    # 因果下三角矩阵：保证因果注意力 (j <= i)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    
    # 状态前缀可见性规则：
    # 当 key 属于 state (seg[j] == 0) 或者 key 与 query 属于同一个问题 (seg[j] == seg[i]) 时允许可见
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    
    # 掩码合成
    allow = causal[None] & same & valid_key
    ...
    # 将不可见的位置填充为极小浮点值 (finfo.min)，经过 Softmax 后权重精确为 0
    return torch.zeros(len(segs), L, L, dtype=dtype, device=device).masked_fill(~allow, torch.finfo(dtype).min)[:, None]
```

#### 注意力掩码矩阵的可视化表象：
设输入序列由 $S$（State）、$Q_1$（问题1）、$Q_2$（问题2）组成：
$$
\mathbf{M}_{ij} = \begin{cases} 
0, & \text{if } j \le i \text{ and } (\text{seg}[j] = 0 \text{ or } \text{seg}[j] = \text{seg}[i]) \\
-\infty, & \text{otherwise}
\end{cases}
$$

| Query \ Key | State (S) | Question 1 ($Q_1$) | Question 2 ($Q_2$) |
| :--- | :---: | :---: | :---: |
| **State ($S$)** | **下三角因果可见** | 🚫 完全遮蔽 ($-\infty$) | 🚫 完全遮蔽 ($-\infty$) |
| **Question 1 ($Q_1$)** | **全可见** (共享上下文) | **下三角因果可见** | 🚫 完全遮蔽 ($-\infty$) |
| **Question 2 ($Q_2$)** | **全可见** (共享上下文) | 🚫 完全遮蔽 ($-\infty$) | **下三角因果可见** |

在此掩码控制下：
1. **$Q_1$ 与 $Q_2$ 互不知晓**：计算 $Q_1$ 时没有任何信息能够流向 $Q_2$，计算 $Q_2$ 时也绝不受 $Q_1$ 选项内容的影响，实现了**完美的跨问题隔离（Question Isolation）**；
2. **状态计算复用**：State 处于最前端，其隐层特征在 Transformer 内部只需要经历一次计算，被所有问题分支充分共享。

---

### 3.3 混合注意力架构（Qwen3.5 DeltaNet）的破局之道：Row-Form 物理分支

当 Kev 将基座升级至更强大的 **Qwen3.5** 时，遭遇了一项严峻的底层数学冲突：

> **核心困境**：Qwen3.5 是混合注意力架构（Hybrid Backbone），交替堆叠了标准的 Full Attention 层与 **Gated DeltaNet（线性递归/状态空间模型 SSM）** 层。
> Gated DeltaNet 层本质上是**时域递归状态更新（Recurrent Hidden State）**，它不依赖 2D 注意力权重矩阵，因此**无法感知并遵守 4D Block-Causal Mask**！如果强行使用 Packed 序列拼合，后方问题的 DeltaNet 内部状态会被前方问题污染。

	#### Kev 的工程解法：`rows_of` 与物理批处理（Row-Form）
	为了让 Qwen3.5 具备严格隔离性，Kev 在 [`DecisionModel.forward_rows_batch`](file:///Users/zhangfan/project/jev/kev/kev/model.py#L192-L200) 中设计了 **Row-Form 方案**：
	
	1. **逻辑解包**：调用 [`rows_of`](file:///Users/zhangfan/project/jev/kev/kev/model.py#L104-L119)，将一条包含 $Q$ 个问题的请求在物理上拆分为 $Q$ 个独立的“单行序列”：
	   $$\text{Row}_k = [\text{State Tokens}] + [\text{Tokens of Question } k]$$
	2. **位置保持**：每个 $\text{Row}_k$ 中 Question 的 Position IDs 依然从 $L_s$ 开始编号。
	3. **物理隔离 Batch 前向**：将这 $Q$ 个独立行组装成一个标准 Batch（形状为 $[Q, L_s + L_{\text{branch}}, d]$），交由 Qwen3.5 执行标准因果前向传播。
	4. **数学等价性**：由于每行在物理内存中完全独立，彻底杜绝了 DeltaNet 状态跨问题泄漏。在纯 Attention 模型上，Kev 严格证明了 Row-Form 与 Packed Mask 在 fp32 精度下的输出误差 $< 4 \times 10^{-6}$（见 `tests/test_v3.py`）。

---

### 3.4 指针读出网络（Pointer Readout Head）的设计与概率求解

在传统大模型中，输出是通过词表矩阵 $\mathbf{W}_{\text{vocab}} \in \mathbb{R}^{d \times V}$ 计算的。而在 Kev 中，这一庞大结构被一个极简而精悍的 **[`PointerHead`](file:///Users/zhangfan/project/jev/kev/kev/model.py#L121-L130)** 取代：

```python
class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        super().__init__()
        # 将骨干网络的隐藏维度 d (例如 4B 模型的 2560) 投影到更低维的指针空间 dp (256)
        self.q = nn.Linear(d, dp)
        self.k = nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide, h_opts):
        # h_decide: [d]       <-- 来自 <decide> 标记的隐藏状态向量
        # h_opts:   [K, d]    <-- 来自 K 个 </opt> 标记的隐藏状态向量集合
        # 输出:     logits [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale
```

#### 为什么取这两个 Token 的 Hidden State？
* **$h_{\text{decide}}$（决策查询向量 $\mathbf{q}$）**：`<decide>` 是问题分支的最后一个 Token。在因果注意力机制下，由于其处于末尾，它**回溯关注了题干指令以及所有候选选项的全貌**，汇聚了全局决策上下文；
* **$h_{\text{opts}}[k]$（候选键向量 $\mathbf{k}_k$）**：位于第 $k$ 个选项的闭合标记 `</opt>` 处。它刚刚完整处理完该选项的文本描述，是该选项语义的最精确聚合表征；
* **点积打分与 Softmax**：
  $$\text{Logits}_k = \frac{(\mathbf{W}_k \cdot h_{\text{opts}}[k])^T (\mathbf{W}_q \cdot h_{\text{decide}})}{\sqrt{d_p}}$$
  $$P(\text{Option}_k) = \frac{\exp(\text{Logits}_k)}{\sum_{j=1}^K \exp(\text{Logits}_j)}$$

这种指针点积机制不仅参数量极小（仅两个 $d \times 256$ 的线性层），而且将选项分类问题直接退化为一个跨空间的相关度度量，极大地提高了泛化能力。

---

### 3.5 三大决策原语的统一映射（Choice / Noul / Score）

Kev 的底层指针头只做一件事：**针对一组候选向量输出概率分布**。在上层 API 逻辑（[`kev/api.py`](file:///Users/zhangfan/project/jev/kev/kev/api.py)）中，三大决策原语被优雅地统一映射到底层指针原语上：

```
                              ┌────────────────────────────────────────┐
                              │           底层 PointerHead             │
                              │   输出: K 个选项的原始概率 P_0..P_{K-1}  │
                              └──────────────────┬─────────────────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   ▼                             ▼                             ▼
       【Choice: 多选一分类】           【Noul: 二元是非判定】            【Score: 有序梯度打分】
       • 构造 K 个 </opt> 分支           • 转化为 2 个选项的 Choice:     • 构造等级梯度 </opt> 阶梯
       • 选出最大概率选项                  - opt 0: "true" 判定           - opt 0..K-1: 对应各个等级
       • 计算置信度指标                    - opt 1: "false" 判定         • 计算数学期望得分:
                                         • 输出真值概率:                   E = ∑ i × P(i)
                                           P(noul) = P_0                 • 评估分布离散方差计算置信度
```

#### 1. `Choice` 原语：置信度数学公式
对于 $K$ 个选项的 Choice 问题，除了输出命中项和各选项概率，Kev 还会计算一个标量置信度（Confidence）。当 $K > 1$ 时：
$$\text{Confidence} = \frac{p_{\max} - \frac{1}{K}}{1 - \frac{1}{K}}$$
* 当所有选项完全均分（$p_{\max} = 1/K$）时，置信度为 $0$（完全迷茫）；
* 当单一选项确定性命中（$p_{\max} = 1.0$）时，置信度为 $1.0$（绝对确信）。

#### 2. `Noul` 原语：二元决策
用户提问一个判断句（如 `"Is this ticket urgent?"`）。Kev 内部自动将其包装为两个选项：`true` 和 `false`。最终输出 `response.nouls["urgency"].noul = P(true)`，返回值天然落在 $[0, 1]$ 闭区间内。

#### 3. `Score` 原语：期望值与有序连续评分
许多业务需求是对程度打分（例如用户愤怒等级 0: 平静, 1: 恼怒, 2: 极度愤怒）。如果强行采用离散分类，容易丢失有序信息。
Kev 将其建模为一个阶梯积分过程，最终输出连续期望得分：
$$\text{Score} = \sum_{i=0}^{K-1} i \cdot P(i)$$
调用方可以直接获得诸如 `1.44` 这样具有精细刻度的连续评价值。

---

## 四、 训练策略与损失函数设计

### 4.1 训练数据构造与无缝推理对齐

Kev 的训练数据构建在 [`kev/kev/data.py`](file:///Users/zhangfan/project/jev/kev/kev/data.py) 中。作者团队将十余个公开数据集（Banking77、BoolQ、AG News、MNLI、SST-5、Yelp 等）以及合成策略规则转化为了统一格式。

每一条训练样本在落盘时，直接序列化为标准的 TypeSafe API Request 结构体（JSONL），并标注目标 `label`：
```json
{
  "state": "I see two charges on my card for the same transaction.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which department should handle this ticket?",
      "criteria": {
        "billing": "Charges, refunds, and duplicate fees",
        "shipping": "Tracking and parcel delays"
      },
      "label": "billing"
    }
  }
}
```
通过 [`api.to_record()`](file:///Users/zhangfan/project/jev/kev/kev/api.py) 模块，**模型在训练时看到的分词、Token 序列拓扑与在生产环境中接收到 HTTP 请求后的排布完全一模一样**，实现了真正的“训练即推理，零推理格式漂移”。

### 4.2 损失函数组合

Kev 的综合损失函数包含了任务分类交叉熵与两个专门针对决策稳定性设计的辅助正则项（位于 [`kev/kev/train.py`](file:///Users/zhangfan/project/jev/kev/kev/train.py)）：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{perm}} \mathcal{L}_{\text{perm\_KL}} + \lambda_{\text{ord}} \mathcal{L}_{\text{RPS}}$$

#### 1. 主交叉熵损失（$\mathcal{L}_{\text{CE}}$）
针对每个决策问题的目标真实选项计算标准负对数似然（Negative Log-Likelihood）。

#### 2. 置换一致性损失（$\mathcal{L}_{\text{perm\_KL}}$）
为了解决大模型普遍存在的“选项顺序敏感”顽疾，Kev 在训练时对同一个请求随机打乱其 Choice 选项的排列顺序，得到两个不同的前向概率分布 $P_{\text{orig}}$ 与 $P_{\text{shuffled}}$。损失函数强制约束两者在映射回同一选项时的 KL 散度趋近于零：
$$\mathcal{L}_{\text{perm\_KL}} = D_{\text{KL}}(P_{\text{orig}} \parallel P_{\text{shuffled}})$$

#### 3. 有序等级排序损失（$\mathcal{L}_{\text{RPS}}$ / Ranked Probability Score）
对于 `Score` 打分题型，预测为相邻等级（如将真实 1 级误判为 2 级）的惩罚应该远小于跨等级误判（误判为 5 级）。Kev 引入严格的排序概率得分损失（RPS），惩罚累积分布函数（CDF）之间的二阶范数偏离。

### 4.3 LoRA 适配器靶向注入

Kev 采用参数高效微调（PEFT / LoRA），Rank 固定为 16，Alpha 为 32。
针对不同的基座类型，Kev 在 [`model.py:151-157`](file:///Users/zhangfan/project/jev/kev/kev/model.py#L151-L157) 中自动识别并动态挂载对应的权重投影层：
* **纯注意力基座（Qwen3）**：注入标准 Attention 算子（`q_proj`, `k_proj`, `v_proj`, `o_proj`）以及 MLP 前馈网络（`gate_proj`, `up_proj`, `down_proj`）；
* **混合注意力基座（Qwen3.5）**：不仅覆盖上述算子，更深入注入了 Gated DeltaNet 的时序混合算子：
  `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`。
这使得 LoRA 适配器能够同时掌握跨注意力层与长程状态记忆层中的决策路由模式。

---

## 五、 推理服务引擎与高性能工程优化

Kev 的服务模块内置于 [`kev/kev/serve.py`](file:///Users/zhangfan/project/jev/kev/kev/serve.py)，使用 FastAPI 驱动。为了在极低资源开销下实现极致的推理响应，Kev 内置了多项高度专业化的底层工程优化。

### 5.1 零生成开销与 Prefill-Only 运行流水线
在标准自回归生成中，由于 Decode 步无法完全并行化，受限于显存带宽（Memory Bandwidth Bound）。
Kev 只有 Prefill 阶段，属于典型的**计算密集型（Compute Bound）**任务。在现代张量加速器（Tensor Cores / Apple AMX）上，矩阵相乘利用率接近峰值。一个由 5 个问题组成的复杂请求，在单张 H100 上推理耗时仅需 **几十毫秒**。

### 5.2 LoRA 权重离线融合（fp32 Merge Before Cast）
在服务启动装载模型时，Kev 默认执行 LoRA 权重的物理融合：
$$W_{\text{fused}} = W_{\text{base}} + \frac{\alpha}{r} (B \cdot A)$$
Kev 在单精度（fp32）下完成矩阵点加后再向下转型为 bf16。融合后的模型退化为一个标准的单体模型，在推理执行期间**完全消除了额外的 LoRA 分支计算与显存跳跃开销**。

### 5.3 状态前缀 KV Cache（State Prefix LRU）
在很多智能客服或 Agent 监控场景中，同一个长文档（如一段系统报错或长篇业务手册）往往会被连续调用多次来回答不同的决策问题。
Kev 在推理服务中内置了一个 LRU State Prefix Cache：
* 当新请求到达时，系统计算其 `state` 的 Token 哈希；
* 若命中缓存，模型**直接复用该 State 已经预计算完毕的 Key/Value 隐层状态**，GPU 只需要前向计算各 Question 对应的那几十个 Token；
* 在 772-Token 的长文档压测中，前缀缓存将 Kev-4B 在 Mac 本地的响应耗时从 **861 ms 骤降至 242 ms**（提升 3.5 倍）。

### 5.4 跨平台推理适配：MPS (Shape Bucket) 与 CUDA (FLA)

```
                            ┌─────────────────────────────────┐
                            │      硬件平台自动化适配路由       │
                            └────────────────┬────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          【Apple Silicon (MPS)】                            【NVIDIA (CUDA)】
    • 痛点: Metal 动态 Shape 导致驱动层反复编译      • 采用 Flash-Linear-Attention (FLA)
    • 解法: KEV_SHAPE_BUCKET=64 (按 64 步长对齐 Padding)    • 启用 SDPA 高性能融合算子
    • 收益: 消除 JIT 编译毛刺，降低 P99 抖动           • 收益: H100 单卡并发吞吐大幅跃升
```

1. **Apple Silicon (MPS)**：
   在 Mac 上，由于缺少针对 DeltaNet 的手写 Metal 算子，动态长度序列会导致 Metal 驱动层反复发生微内核（Micro-kernel）编译重试，产生巨大的时间开销。Kev 通过环境变量 `KEV_SHAPE_BUCKET=64`，将所有序列按 64 的倍数进行右侧安全填充，锁定了静态计算图，消除了耗时抖动。
2. **NVIDIA CUDA**：
   集成了 `flash-linear-attention`（FLA）与 Triton 优化算子，在 GPU 硬件上以极致速度展开线性状态更新。

### 5.5 后处理概率校准与温度缩放（Temperature Scaling）
为了让输出的概率分布具备极佳的置信度可靠性（较低的 ECE 与 Brier Score），Kev 提供了开箱即用的温度标定：
* 环境变量 `KEV_TEMPERATURE=2.0`：使用在分布外测试集上拟合的最优温度缩放因子对 Logits 进行平滑。在保证最终分类命中完全不变的前提下，将 Kev-9B 的过度自信错误率（预测置信度 $\ge 0.9$ 但实际选错的比例）从 **8.7% 砍半至 4.4%**，完全媲美商业 Jev 服务的 3.7%。
* 环境变量 `KEV_DATE_FACTS=1`：针对语言模型对绝对日历时间差运算较差的短板，自动在 State 后追加日期时间跨度事实断言，在无须重新微调的前提下将截止日期判定准确率从 0.80 提升至 0.90。

---

## 六、 工业实战：AICO 技能选择（Skill Selection）基准评测

在本工程（`/Users/zhangfan/project/jev`）的落地实践中，Kev 被深度应用于智能体系统的首轮技能路由器——**AICO Skill Selection**。

### 6.1 场景背景与 20 题业务基准构建
在真实智能体生产调用流中，当用户提出一个运维或配置请求时，系统必须第一时间内做出低延迟、高准确的技能分发：
* **候选集合**：
  * `query_param`：配置参数查询
  * `query_pm`：性能指标查询
  * `query_alarm`：设备与链路告警检索
  * `calculate_gain`：增益与衰减推演计算
  * `__no_skill__`：业务无法处理，兜底流转
* **评测架构**：项目内构建了独立的自动化对比评测套件 [`skill_selection_20/`](file:///Users/zhangfan/project/jev/skill_selection_20)，通过规范的离线准备、回放请求与报表生成，衡量不同技术方案的表现：
  * 本地评测脚本：[`run_benchmark.py`](file:///Users/zhangfan/project/jev/skill_selection_20/run_benchmark.py)（直连本地 `kev.serve` 实例）
  * 官方对照脚本：[`run_official_benchmark.py`](file:///Users/zhangfan/project/jev/skill_selection_20/run_official_benchmark.py)（直连 TypeSafe 官方云端服务）

### 6.2 Kev-4B vs. 官方 JEV vs. 通用大模型实测对比

基于 20 道真实业务题目的实测对比结果如下表所示：

| 评估维度 | 原业务大模型 (`Qwen-27B-TP4`) | 官方云端 JEV (`jev-latest`) | 本地部署 Kev (`Kev-4B` bf16) |
| :--- | :---: | :---: | :---: |
| **模型参数量** | 27 Billion (需多卡并行) | 闭源黑盒 | **4 Billion (单卡/单台 Mac 可跑)** |
| **单次请求平均延迟** | 3.5s ~ 8.0s (含参数自回归解码) | 约 1.400s (含跨境网络往返) | **0.49s ~ 0.78s (本地局域网)** |
| **业务判定一致率** | 基准源（100%） | **20 / 20 (100% 一致)** | **20 / 20 (100% 一致)** |
| **输出不确定性/幻觉** | 存在 JSON 语法破坏风险 | 零幻觉，标准 Float 概率 | **零幻觉，标准 Float 概率** |
| **数据安全性与成本** | 本地推理成本极高 | 数据出境，产生 API 计费 | **完全内网闭环，零边际调用成本** |

实践表明：在特定领域的决策路由任务上，经过架构改造的 4B 规模 Kev 模型**完全达到了 27B 级大模型与商业级 Jev 服务的决策判断水准**，同时将延迟压缩至亚秒级，并彻底根除了生成式模型的格式幻觉问题。

---

## 七、 总结与架构演进展望

Kev 展现了在后大模型时代（Post-LLM Era）一种极具启发性的技术路线：**并非所有智能化需求都需要自回归生成自然语言**。

通过将 Transformer 算力约束在纯表示空间（Representation Learning）内，结合 **Block-Causal 掩码拓扑**、**Prefix KV Cache 共享** 与 **Pointer Head 点积读出机制**，Kev 成功以最小的算力代价实现了对商业 Jev 体系的完整逆向复刻。它不仅为 Agent 架构师提供了可靠的高性能控制流基础设施，也为解决大模型的过度自信、换序敏感与高延迟瓶颈指明了一条清晰的工程演进道路。
