# Chunker Cross Comparison Experiment

- Date: 2026-05-21T10:16:31
- Interpreter: `/Users/jianghong/anaconda3/envs/all-in-rag/bin/python`
- Document: `/Users/jianghong/Desktop/Summer-SupplyChainData/all-in-rag/data/C2/md/easy-rl-chapter1.md`
- Source length: `23172`
- QA source: `curated`
- QA count: `5`

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

## Notes

- CharacterTextSplitter: note=`-` error=`-`
- RecursiveCharacterTextSplitter: note=`-` error=`-`
- SemanticChunker: note=`-` error=`-`
- Semantic + Recursive: note=`-` error=`-`
- MarkdownHeaderTextSplitter: note=`-` error=`-`
- MarkdownHeader + Recursive: note=`-` error=`-`
- Parent-Child Chunking: note=`parent_chunks=26` error=`-`
