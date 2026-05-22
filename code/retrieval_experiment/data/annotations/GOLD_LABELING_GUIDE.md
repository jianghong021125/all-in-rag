# Gold Labeling Guide

这一轮 retrieval eval 建议采用两阶段 gold。

## 第一阶段：先做文档级 gold

先给每个问题标 `gold_doc_ids`，也就是“最少需要命中的文档集合”。

这样做的好处是：

- 标注门槛低
- 能先跑通 `dense / sparse / hybrid` 的 baseline 筛选
- 更适合作为第 1 组实验的起点

### 文档级 gold 的规则

1. `gold_doc_ids` 只放“最小充分证据文档”，不要把所有沾边文档都放进去。
2. 单跳问题通常只有 1 个 `gold_doc_id`。
3. 多跳问题可以有多个 `gold_doc_ids`。
4. 模糊意图型问题允许给 1 到 2 个更贴近用户真实需求的主文档。

## 第二阶段：再做 chunk 级 gold

等 `chunk_manifest.json` 生成后，再补 `gold_chunk_ids`。

### chunk 级 gold 的规则

1. 只选择能直接支持回答的最小充分证据块。
2. 优先选择信息最集中、最直接的块。
3. 不要因为同一段文字被重复切分，就把所有重叠块都标进去。
4. 如果问题需要跨段综合，可以保留多个 `gold_chunk_ids`。

## 推荐工作流

1. 先运行：

```bash
conda run -n all-in-rag python all-in-rag/code/retrieval_experiment/retrieval_eval.py prepare-chunks
```

2. 再运行：

```bash
conda run -n all-in-rag python all-in-rag/code/retrieval_experiment/prepare_gold_templates.py
```

3. 打开生成的 `gold_chunk_labeling_workspace.json`
4. 对每个问题，从 `candidate_chunks_from_gold_docs` 中手动填 `gold_chunk_ids`

## 字段解释

- `question_id`：问题唯一标识
- `question_type`：问题类型
- `gold_doc_ids`：第一阶段主标注字段
- `gold_chunk_ids`：第二阶段主标注字段
- `gold_section_hints`：帮助快速定位的章节提示
- `candidate_chunks_from_gold_docs`：仅用于标注辅助，不是最终 gold

## 一个实用建议

如果你目前还不熟悉 gold 构建，最稳的方法是：

- 先把 `gold_doc_ids` 做准确
- 先用文档级指标筛基础检索方案
- 后面只给重点问题或最终实验集补 `gold_chunk_ids`

这样能先把实验推进起来，而不是一开始就卡在细粒度标注上。
