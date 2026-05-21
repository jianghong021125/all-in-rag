# Test 1：“easy-rl-chapter1” 

## Experiment Setup

- 文档: `easy-rl-chapter1.md`
- 原文长度: `23172` 字符
- 解释器: `/Users/jianghong/anaconda3/envs/all-in-rag/bin/python`
- QA 集来源: `curated`
- QA 数量: `5`
- 运行时间: `2026-05-21T10:16:31`
- 默认语义模型: `BAAI/bge-small-zh-v1.5`
- 所有策略均使用当前后端文件中的默认配置，并对 `easy-rl-chapter1.md` 重新实验。

## Cross-Comparison Metrics Table

| 方法 | 状态 | 块数 | 平均长度 | 中位长度 | 最短 | 最长 | 覆盖率 | 方差得分 | Header得分 | 检索得分 | Hit@3 | MRR | 平均重叠 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CharacterTextSplitter | 成功 | 76 | 323.289 | 341.000 | 93 | 608 | 1.060 | 0.787 | 0.234 | 1.000 | 1.000 | 1.000 | 0.661 |
| RecursiveCharacterTextSplitter | 成功 | 83 | 299.783 | 318.000 | 93 | 391 | 1.074 | 0.800 | 0.230 | 1.000 | 1.000 | 1.000 | 0.694 |
| SemanticChunker | 成功 | 2 | 11584.500 | 11584.500 | 7134 | 16035 | 1.000 | 0.722 | 1.000 | 1.000 | 1.000 | 1.000 | 0.033 |
| Semantic + Recursive | 成功 | 85 | 293.741 | 315.000 | 93 | 391 | 1.078 | 0.797 | 0.233 | 1.000 | 1.000 | 1.000 | 0.690 |
| MarkdownHeaderTextSplitter | 成功 | 21 | 1092.190 | 917.000 | 225 | 3300 | 0.990 | 0.553 | 0.000 | 0.950 | 1.000 | 0.900 | - |
| MarkdownHeader + Recursive | 成功 | 80 | 310.712 | 319.000 | 84 | 399 | 1.073 | 0.822 | 1.000 | 1.000 | 1.000 | 1.000 | 0.430 |
| Parent-Child Chunking | 成功 | 102 | 283.716 | 297.000 | 23 | 391 | 1.249 | 0.777 | 0.244 | 0.950 | 1.000 | 0.900 | 0.676 |

## Pairwise Overlap Matrix

| 方法 | CharacterTextSplitter | RecursiveCharacterTextSplitter | SemanticChunker | Semantic + Recursive | MarkdownHeaderTextSplitter | MarkdownHeader + Recursive | Parent-Child Chunking |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CharacterTextSplitter | 1.000 | 0.949 | 0.034 | 0.933 | - | 0.478 | 0.911 |
| RecursiveCharacterTextSplitter | 0.949 | 1.000 | 0.034 | 0.983 | - | 0.547 | 0.957 |
| SemanticChunker | 0.034 | 0.034 | 1.000 | 0.034 | - | 0.031 | 0.034 |
| Semantic + Recursive | 0.933 | 0.983 | 0.034 | 1.000 | - | 0.557 | 0.942 |
| MarkdownHeaderTextSplitter | - | - | - | - | 1.000 | - | - |
| MarkdownHeader + Recursive | 0.478 | 0.547 | 0.031 | 0.557 | - | 1.000 | 0.537 |
| Parent-Child Chunking | 0.911 | 0.957 | 0.034 | 0.942 | - | 0.537 | 1.000 |

## How To Read The Metrics

- `块数`、`平均长度`、`中位长度`、`最短/最长` 用来判断分块粒度和大小分布。
- `覆盖率` 大于 `1` 通常意味着 overlap 带来了重复内容；越大表示索引冗余越明显。
- `方差得分` 越高，表示 chunk 长度越均匀，通常更利于稳定检索。
- `Header得分` 越高，表示 Markdown 章节结构保留得越好。
- `检索得分` 是 `Hit@K` 和 `MRR` 的平均值，用来衡量 chunk 是否方便被检索回来。
- `平均重叠` 用来比较该方法和其他方法的边界相似度，高不一定更好，只表示切法更接近其他策略。

## Strategy-by-Strategy Interpretation

### CharacterTextSplitter

- 这是最基础的长度型 baseline。本轮得到 `76` 个块，平均长度 `323.289`。
- 检索得分为 `1.000`，说明对当前 QA 集已经足够覆盖；但 Header 得分只有 `0.234`，表明它基本不感知 Markdown 结构。
- 平均重叠为 `0.661`，说明它和多数细粒度策略仍然比较接近，适合作为对照基线。

### RecursiveCharacterTextSplitter

- 本轮得到 `83` 个块，平均长度 `299.783`，方差得分 `0.800` 高于普通字符切分。
- 检索得分仍为 `1.000`，平均重叠达到 `0.694`，说明它与大多数有效细粒度方法的边界相当接近。
- 如果只想要一个简单可靠、无需结构假设的默认分块方案，它依然是很强的通用 baseline。

### SemanticChunker

- 默认配置下只切出 `2` 个超大块，平均每块 `11584.500` 字符，明显属于粗粒度主题切分。
- Header 得分和检索得分都为 `1.000` / `1.000`，但这更多是因为块太大，标题和答案内容都被整体包含进去了。
- 它和其他方法的平均重叠只有 `0.033`，说明切分风格与其他细粒度方法差异很大。

### Semantic + Recursive

- 先做语义分块，再对过长语义块做递归细分，最终得到 `85` 个块，平均长度 `293.741`。
- 它与 `RecursiveCharacterTextSplitter` 的 overlap 为 `0.983`，说明在当前参数下，后续递归切分几乎主导了最终边界。
- 当前检索得分为 `1.000`，但 Header 得分 `0.233` 仍然不高，结构优势还没有被明显释放。

### MarkdownHeaderTextSplitter

- 只按标题切分后得到 `21` 个块，平均长度 `1092.190`，最长块达到 `3300`，长度差异明显。
- 检索得分为 `0.950`，说明能命中正确内容，但 `MRR=0.900` 表明不是每次都把最佳块排在第 1。
- 当前 Header 得分显示为 `0.000`，更可能反映 span/metadata 对齐方式的限制，而不是它真的不保留标题结构。

### MarkdownHeader + Recursive

- 这是本轮综合最强的策略之一：得到 `80` 个块，平均长度 `310.712`，方差得分 `0.822` 为全场最高。
- Header 得分 `1.000`、检索得分 `1.000`，说明它既保住了章节结构，又维持了适合检索的块大小。
- 平均重叠只有 `0.430`，说明它不是简单复刻字符切分，而是沿着标题结构形成了更有文档意识的边界。

### Parent-Child Chunking

- 本轮得到 `102` 个子块，是本轮最多的；覆盖率达到 `1.249`，说明重叠和冗余也最明显。
- 检索得分为 `0.950`，Hit@3 仍是 `1.000`，但 `MRR=0.900` 说明正确块能找回，却不总是排在最前。
- 这轮评测只在 child chunk 上做检索，没有真正测试 parent-child 联合检索，因此它的潜力可能还没有被完全体现。

## Overall Takeaways

- 从当前默认配置看，`MarkdownHeader + Recursive` 是最值得优先继续调参的方案：它同时拿到了 `1.000` 的 Header 得分和 `1.000` 的检索得分。
- 如果需要一个简单稳定的 baseline，`RecursiveCharacterTextSplitter` 依然是最稳妥的起点，它的平均重叠 `0.694` 也说明它和多数细粒度方案较一致。
- `SemanticChunker` 当前默认参数下只切出 `2` 个超大块，更适合主题级粗分，而不是直接作为细粒度检索块。
- `Parent-Child Chunking` 还需要配套真正的父子层级检索流程，当前分数偏保守。
- `MarkdownHeaderTextSplitter` 的结构相关分数值得后续单独校验，因为现有实现可能低估了它对标题结构的保留效果。
