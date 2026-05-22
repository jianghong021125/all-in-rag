from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
QA_PATH = DATA_DIR / "qa" / "curated_qa_small.json"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
OUTPUTS_DIR = BASE_DIR / "outputs"

COMMON_SEPARATORS = ["\n\n", "\n", "。", "，", " ", ""]
MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

DEFAULT_CHUNK_STRATEGY = "markdown_recursive"
PARENT_CHILD_CHUNK_STRATEGY = "parent_child"

DEFAULT_CHUNK_CONFIG = {
    "strip_headers": False,
    "chunk_size": 400,
    "chunk_overlap": 100,
}

PARENT_CHILD_CHUNK_CONFIG = {
    "parent_chunk_size": 1200,
    "parent_chunk_overlap": 300,
    "child_chunk_size": 400,
    "child_chunk_overlap": 100,
}

# Backward-compatible alias for the default experiment chunker.
CHUNK_CONFIG = DEFAULT_CHUNK_CONFIG

# Fixed order of headers for consistent chunking and evaluation.
HEADER_ORDER = ["Header 1", "Header 2", "Header 3", "Header 4"]
DEFAULT_TOP_K = 5 # final retrieved top-k, which will be tuned in group 4 variants
DEFAULT_CANDIDATE_K = 10
DEFAULT_RRF_K = 60 # rrf k, which will be tuned in group 4 variants
DEFAULT_DENSE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_REWRITE_MODEL_NAME = "deepseek-chat"

# Default parameters for parent-child retrieval variants. (Group 1)
DEFAULT_PARENT_CHILD_K = 10

# Default parameters for Group 4 retrieval variants, can be tuned. (Group 4)
DEFAULT_GROUP4_DENSE_CANDIDATE_K = 20
DEFAULT_GROUP4_SPARSE_CANDIDATE_K = 20
DEFAULT_GROUP4_ALPHA = 0.5
PARENT_CHILD_VARIANTS = {
    "parent_child",
    "query_rewrite_parent_child",
    "rrf_parent_child",
    "all_in",
}
GROUP4_VARIANTS = [
    "union_rrf",
    "weighted_score_fusion",
    "sparse_first_dense_rerank",
    "dense_first_sparse_rerank",
    "intersection_rrf",
]

CHUNK_MANIFEST_PATH = ANNOTATIONS_DIR / "chunk_manifest.json"
DOC_MANIFEST_PATH = ANNOTATIONS_DIR / "doc_manifest.json"
PARENT_CHILD_CHUNK_MANIFEST_PATH = ANNOTATIONS_DIR / "chunk_manifest_parent_child.json"
PARENT_CHILD_DOC_MANIFEST_PATH = ANNOTATIONS_DIR / "doc_manifest_parent_child.json"
GOLD_WORKSPACE_PATH = ANNOTATIONS_DIR / "gold_chunk_labeling_workspace.json"
PARENT_CHILD_GOLD_WORKSPACE_PATH = ANNOTATIONS_DIR / "gold_chunk_labeling_workspace_parent_child.json"
