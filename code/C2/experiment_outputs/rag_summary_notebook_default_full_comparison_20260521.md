# Chunker Cross Comparison Experiment

- Date: 2026-05-21T10:36:40
- Interpreter: `/Users/jianghong/anaconda3/envs/all-in-rag/bin/python`
- Document: `/Users/jianghong/Desktop/Summer-SupplyChainData/all-in-rag/data/C2/md/rag-summary-notebook.md`
- Source length: `15121`
- QA source: `heuristic`
- QA count: `5`

## Cross-Comparison Metrics Table

| 方法 | 状态 | 块数 | 平均长度 | 中位长度 | 最短 | 最长 | 覆盖率 | 方差得分 | Header得分 | 检索得分 | Hit@3 | MRR | 平均重叠 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CharacterTextSplitter | 成功 | 48 | 318.708 | 283.500 | 3 | 1056 | 1.012 | 0.639 | 0.862 | 0.800 | 0.800 | 0.800 | 0.664 |
| RecursiveCharacterTextSplitter | 成功 | 57 | 275.368 | 284.000 | 3 | 395 | 1.038 | 0.751 | 0.756 | 0.800 | 0.800 | 0.800 | 0.712 |
| SemanticChunker | 成功 | 2 | 7560.000 | 7560.000 | 1314 | 13806 | 1.000 | 0.548 | 1.000 | 0.950 | 1.000 | 0.900 | 0.045 |
| Semantic + Recursive | 成功 | 58 | 270.603 | 283.500 | 3 | 395 | 1.038 | 0.741 | 0.753 | 0.750 | 0.800 | 0.700 | 0.711 |
| MarkdownHeaderTextSplitter | 成功 | 55 | 273.145 | 255.000 | 52 | 1062 | 0.994 | 0.585 | 1.000 | 0.750 | 0.800 | 0.700 | 0.595 |
| MarkdownHeader + Recursive | 成功 | 66 | 233.621 | 253.000 | 52 | 399 | 1.020 | 0.703 | 1.000 | 0.750 | 0.800 | 0.700 | 0.619 |
| Parent-Child Chunking | 成功 | 71 | 260.183 | 277.000 | 3 | 398 | 1.222 | 0.725 | 0.802 | 0.733 | 0.800 | 0.667 | 0.716 |

## Pairwise Overlap Matrix

| 方法 | CharacterTextSplitter | RecursiveCharacterTextSplitter | SemanticChunker | Semantic + Recursive | MarkdownHeaderTextSplitter | MarkdownHeader + Recursive | Parent-Child Chunking |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CharacterTextSplitter | 1.000 | 0.864 | 0.078 | 0.857 | 0.697 | 0.631 | 0.859 |
| RecursiveCharacterTextSplitter | 0.864 | 1.000 | 0.047 | 0.992 | 0.654 | 0.732 | 0.983 |
| SemanticChunker | 0.078 | 0.047 | 1.000 | 0.046 | 0.030 | 0.023 | 0.048 |
| Semantic + Recursive | 0.857 | 0.992 | 0.046 | 1.000 | 0.655 | 0.733 | 0.981 |
| MarkdownHeaderTextSplitter | 0.697 | 0.654 | 0.030 | 0.655 | 1.000 | 0.850 | 0.680 |
| MarkdownHeader + Recursive | 0.631 | 0.732 | 0.023 | 0.733 | 0.850 | 1.000 | 0.745 |
| Parent-Child Chunking | 0.859 | 0.983 | 0.048 | 0.981 | 0.680 | 0.745 | 1.000 |

## QA Examples Used

- 请检索与“RAG Summary Notebook”相关的内容。
- 请检索与“1. 什么是 RAG”相关的内容。
- 请检索与“2. 端到端工作流”相关的内容。
- 请检索与“2.1 高层流程”相关的内容。
- 请检索与“2.2 核心逻辑”相关的内容。

## Notes

- CharacterTextSplitter: note=`-` error=`-`
- RecursiveCharacterTextSplitter: note=`-` error=`-`
- SemanticChunker: note=`-` error=`-`
- Semantic + Recursive: note=`-` error=`-`
- MarkdownHeaderTextSplitter: note=`-` error=`-`
- MarkdownHeader + Recursive: note=`-` error=`-`
- Parent-Child Chunking: note=`parent_chunks=18` error=`-`
