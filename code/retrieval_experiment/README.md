# Retrieval Experiment

这一目录用于搭建第 1 组实验的最小 retrieval eval 脚手架，先固定语料、固定 chunk、固定 QA，再比较 `dense / sparse / hybrid` 的检索表现。

## 当前范围

- 固定文档集合：`data/docs/chapter1-3` 下的 Markdown 文档副本
- 固定 chunk 策略：`MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter`
- 固定 QA：`data/qa/curated_qa_small.json`
- 固定 gold 的第一阶段：`gold_doc_ids`
- 固定 gold 的第二阶段：`gold_chunk_ids`，通过 chunk manifest 再补齐
- 当前最稳可跑：`sparse`
- 可扩展：`dense / hybrid`

## 目录结构

```text
retrieval_experiment/
├── config.py
├── prepare_gold_templates.py
├── retrieval_eval.py
├── README.md
└── data
    ├── annotations
    │   └── GOLD_LABELING_GUIDE.md
    ├── docs
    │   ├── chapter1
    │   ├── chapter2
    │   └── chapter3
    └── qa
        └── curated_qa_small.json
```

## 固定 chunk 配置

配置参考 [05_chunker_cross_comparison.py](/Users/jianghong/Desktop/Summer-SupplyChainData/all-in-rag/code/C2/05_chunker_cross_comparison.py:1)：

- `headers_to_split_on = [#, ##, ###, ####]`
- `strip_headers = False`
- `chunk_size = 400`
- `chunk_overlap = 100`
- `separators = ["\\n\\n", "\\n", "。", "，", " ", ""]`

## 快速开始

先生成 chunk manifest：

```bash
conda run -n all-in-rag python all-in-rag/code/retrieval_experiment/retrieval_eval.py prepare-chunks
```

先跑最稳的 `sparse` baseline：

```bash
conda run -n all-in-rag python all-in-rag/code/retrieval_experiment/retrieval_eval.py evaluate --mode sparse --top-k 5
```

如果后面要给 `gold_chunk_ids` 打标，先生成标注工作区：

```bash
conda run -n all-in-rag python all-in-rag/code/retrieval_experiment/prepare_gold_templates.py
```

## Gold 标注建议

第 1 轮建议先只标 `gold_doc_ids`，因为：

- 文档级 gold 更容易做
- 足够支持第 1 组实验先筛选 `dense / sparse / hybrid`
- 后面要做更细的 chunk-level 评估时，再补 `gold_chunk_ids`

建议流程：

1. 先确认每个问题最少需要命中的文档集合，也就是 `gold_doc_ids`
2. 跑 `prepare-chunks` 生成 `chunk_manifest.json`
3. 跑 `prepare_gold_templates.py` 生成按 gold 文档过滤后的 chunk 标注工作区
4. 再手动填写每题最小充分证据块的 `gold_chunk_ids`

详细规则见 [GOLD_LABELING_GUIDE.md](/Users/jianghong/Desktop/Summer-SupplyChainData/all-in-rag/code/retrieval_experiment/data/annotations/GOLD_LABELING_GUIDE.md:1)。

## 当前环境局限

- 当前机器 `GPU=False`，只能做 CPU 驱动实验
- `sparse` 路线最稳
- `dense` 和 `hybrid` 依赖可用的本地或联网嵌入模型
- 如果本地 Hugging Face 模型解析不完整，`dense` 初始化可能失败；这时先用 `sparse` 打通评测流程即可

## 后续怎么扩

这个脚手架优先覆盖第 1 组的阶段 A。后面可以直接在此基础上继续接：

- `query rewrite only`
- `parent-child only`
- `reranker only = RRF`
- 前两名基础方案上的组合实验
