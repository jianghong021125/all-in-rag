from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import (
    ANNOTATIONS_DIR,
    CHUNK_CONFIG,
    CHUNK_MANIFEST_PATH,
    COMMON_SEPARATORS,
    DEFAULT_CHUNK_CONFIG,
    DEFAULT_GROUP4_ALPHA,
    DEFAULT_GROUP4_DENSE_CANDIDATE_K,
    DEFAULT_GROUP4_SPARSE_CANDIDATE_K,
    DEFAULT_CHUNK_STRATEGY,
    DEFAULT_CANDIDATE_K,
    DEFAULT_DENSE_MODEL_NAME,
    DEFAULT_PARENT_CHILD_K,
    DEFAULT_REWRITE_MODEL_NAME,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    DOC_MANIFEST_PATH,
    DOCS_DIR,
    GROUP4_VARIANTS,
    HEADER_ORDER,
    MARKDOWN_HEADERS_TO_SPLIT_ON,
    OUTPUTS_DIR,
    PARENT_CHILD_CHUNK_CONFIG,
    PARENT_CHILD_CHUNK_MANIFEST_PATH,
    PARENT_CHILD_CHUNK_STRATEGY,
    PARENT_CHILD_DOC_MANIFEST_PATH,
    PARENT_CHILD_VARIANTS,
    QA_PATH,
)

try:
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
except ImportError:
    MarkdownHeaderTextSplitter = None
    RecursiveCharacterTextSplitter = None

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
except ImportError:
    StrOutputParser = None
    PromptTemplate = None

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:
    ChatDeepSeek = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.retrievers import BM25Retriever
    from langchain_community.vectorstores import FAISS
except ImportError:
    HuggingFaceEmbeddings = None
    BM25Retriever = None
    FAISS = None


@dataclass
class SearchResult:
    docs: list[Document]
    evidence_chunk_ids: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusBundle:
    chunk_strategy: str
    child_docs: list[Document]
    parent_docs_by_id: dict[str, Document]
    chunk_manifest: list[dict[str, Any]]
    doc_manifest: list[dict[str, Any]]


@dataclass(frozen=True)
class RunSpec:
    mode: str
    variant: str


@dataclass(frozen=True)
class Group4Preset:
    preset_id: str
    description: str
    dense_candidate_k: int
    sparse_candidate_k: int
    rrf_k: int
    alpha: float
    top_k: int


GROUP4_PARAMETER_PRESETS = [
    Group4Preset(
        preset_id="p01_balanced_default",
        description="Balanced baseline around current defaults.",
        dense_candidate_k=20,
        sparse_candidate_k=20,
        rrf_k=60,
        alpha=0.50,
        top_k=5,
    ),
    Group4Preset(
        preset_id="p02_dense_biased",
        description="Keep candidate sizes fixed, bias weighted fusion toward dense scores.",
        dense_candidate_k=20,
        sparse_candidate_k=20,
        rrf_k=60,
        alpha=0.70,
        top_k=5,
    ),
    Group4Preset(
        preset_id="p03_sparse_biased",
        description="Keep candidate sizes fixed, bias weighted fusion toward sparse scores.",
        dense_candidate_k=20,
        sparse_candidate_k=20,
        rrf_k=60,
        alpha=0.30,
        top_k=5,
    ),
    Group4Preset(
        preset_id="p04_precision_small_pool",
        description="Small candidate pools and smaller final top-k for precision-oriented runs.",
        dense_candidate_k=10,
        sparse_candidate_k=10,
        rrf_k=30,
        alpha=0.50,
        top_k=3,
    ),
    Group4Preset(
        preset_id="p05_recall_large_pool",
        description="Large candidate pools and larger final top-k for recall-oriented runs.",
        dense_candidate_k=30,
        sparse_candidate_k=30,
        rrf_k=100,
        alpha=0.50,
        top_k=8,
    ),
    Group4Preset(
        preset_id="p06_dense_heavy_pool",
        description="Wider dense candidate pool, narrower sparse candidate pool, mild dense preference.",
        dense_candidate_k=30,
        sparse_candidate_k=15,
        rrf_k=60,
        alpha=0.70,
        top_k=5,
    ),
    Group4Preset(
        preset_id="p07_sparse_heavy_pool",
        description="Wider sparse candidate pool, narrower dense candidate pool, mild sparse preference.",
        dense_candidate_k=15,
        sparse_candidate_k=30,
        rrf_k=60,
        alpha=0.35,
        top_k=5,
    ),
    Group4Preset(
        preset_id="p08_aggressive_rrf",
        description="Lower RRF smoothing to make top-ranked items contribute more sharply.",
        dense_candidate_k=20,
        sparse_candidate_k=20,
        rrf_k=20,
        alpha=0.50,
        top_k=5,
    ),
    Group4Preset(
        preset_id="p09_smooth_rrf",
        description="Higher RRF smoothing to reduce the impact of local rank spikes.",
        dense_candidate_k=20,
        sparse_candidate_k=20,
        rrf_k=100,
        alpha=0.50,
        top_k=5,
    ),
    Group4Preset(
        preset_id="p10_large_pool_balanced",
        description="Large balanced pools with slight dense preference and standard final top-k.",
        dense_candidate_k=30,
        sparse_candidate_k=30,
        rrf_k=60,
        alpha=0.55,
        top_k=5,
    ),
]


def ensure_dirs() -> None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_qa_items() -> list[dict[str, Any]]:
    return json.loads(QA_PATH.read_text(encoding="utf-8"))


def relative_doc_id(path: Path) -> str:
    return path.relative_to(DOCS_DIR).with_suffix("").as_posix()


def source_path_for_manifest(path: Path) -> str:
    return path.relative_to(Path(__file__).resolve().parent).as_posix()


def normalize_preview(text: str, width: int = 120) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:width]


def build_header_path(metadata: dict[str, Any]) -> str:
    values = [metadata[key] for key in HEADER_ORDER if metadata.get(key)]
    return " > ".join(values)


def load_raw_markdown_documents() -> list[dict[str, Any]]:
    items = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        items.append(
            {
                "path": path,
                "doc_id": relative_doc_id(path),
                "source_path": source_path_for_manifest(path),
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return items


def build_header_fields(metadata: dict[str, Any]) -> dict[str, str]:
    return {key: metadata.get(key, "") for key in HEADER_ORDER}


def build_markdown_recursive_bundle() -> CorpusBundle:
    if MarkdownHeaderTextSplitter is None or RecursiveCharacterTextSplitter is None:
        raise ImportError("缺少 langchain_text_splitters，无法执行 Markdown + Recursive 分块。")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=DEFAULT_CHUNK_CONFIG["strip_headers"],
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=COMMON_SEPARATORS,
        chunk_size=DEFAULT_CHUNK_CONFIG["chunk_size"],
        chunk_overlap=DEFAULT_CHUNK_CONFIG["chunk_overlap"],
    )

    child_docs: list[Document] = []
    parent_docs_by_id: dict[str, Document] = {}
    chunk_manifest: list[dict[str, Any]] = []
    doc_manifest: list[dict[str, Any]] = []

    for raw_doc in load_raw_markdown_documents():
        header_docs = header_splitter.split_text(raw_doc["text"])
        chunk_start = len(child_docs)

        for section_index, header_doc in enumerate(header_docs):
            parent_section_id = f"{raw_doc['doc_id']}::parent_{section_index:03d}"
            parent_metadata = dict(header_doc.metadata)
            parent_metadata.update(
                {
                    "doc_id": raw_doc["doc_id"],
                    "source_path": raw_doc["source_path"],
                    "parent_section_id": parent_section_id,
                    "section_index": section_index,
                    "header_path": build_header_path(header_doc.metadata),
                    "supporting_chunk_ids": [],
                }
            )
            parent_docs_by_id[parent_section_id] = Document(
                page_content=header_doc.page_content,
                metadata=parent_metadata,
            )

            split_children = recursive_splitter.split_documents([header_doc])
            for child_doc in split_children:
                chunk_index = len(child_docs) - chunk_start
                chunk_id = f"{raw_doc['doc_id']}::chunk_{chunk_index:03d}"
                metadata = dict(child_doc.metadata)
                metadata.update(
                    {
                        "doc_id": raw_doc["doc_id"],
                        "source_path": raw_doc["source_path"],
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "section_index": section_index,
                        "header_path": build_header_path(child_doc.metadata),
                        "parent_section_id": parent_section_id,
                    }
                )
                parent_docs_by_id[parent_section_id].metadata["supporting_chunk_ids"].append(chunk_id)

                child_docs.append(Document(page_content=child_doc.page_content, metadata=metadata))
                chunk_manifest.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": raw_doc["doc_id"],
                        "source_path": raw_doc["source_path"],
                        "chunk_index": chunk_index,
                        "section_index": section_index,
                        "parent_section_id": parent_section_id,
                        "header_path": metadata["header_path"],
                        "headers": build_header_fields(metadata),
                        "text": child_doc.page_content,
                        "preview": normalize_preview(child_doc.page_content),
                    }
                )

        doc_manifest.append(
            {
                "doc_id": raw_doc["doc_id"],
                "source_path": raw_doc["source_path"],
                "char_length": len(raw_doc["text"]),
                "chunk_count": len(child_docs) - chunk_start,
            }
        )

    return CorpusBundle(
        chunk_strategy=DEFAULT_CHUNK_STRATEGY,
        child_docs=child_docs,
        parent_docs_by_id=parent_docs_by_id,
        chunk_manifest=chunk_manifest,
        doc_manifest=doc_manifest,
    )


def build_parent_child_bundle() -> CorpusBundle:
    if RecursiveCharacterTextSplitter is None:
        raise ImportError("缺少 RecursiveCharacterTextSplitter，无法执行 parent-child 分块。")

    parent_splitter = RecursiveCharacterTextSplitter(
        separators=COMMON_SEPARATORS,
        chunk_size=PARENT_CHILD_CHUNK_CONFIG["parent_chunk_size"],
        chunk_overlap=PARENT_CHILD_CHUNK_CONFIG["parent_chunk_overlap"],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        separators=COMMON_SEPARATORS,
        chunk_size=PARENT_CHILD_CHUNK_CONFIG["child_chunk_size"],
        chunk_overlap=PARENT_CHILD_CHUNK_CONFIG["child_chunk_overlap"],
    )

    child_docs: list[Document] = []
    parent_docs_by_id: dict[str, Document] = {}
    chunk_manifest: list[dict[str, Any]] = []
    doc_manifest: list[dict[str, Any]] = []

    for raw_doc in load_raw_markdown_documents():
        doc_chunk_start = len(child_docs)
        parent_chunks = parent_splitter.split_text(raw_doc["text"])

        for parent_index, parent_text in enumerate(parent_chunks):
            parent_section_id = f"{raw_doc['doc_id']}::parent_{parent_index:03d}"
            parent_metadata = {
                "doc_id": raw_doc["doc_id"],
                "source_path": raw_doc["source_path"],
                "parent_section_id": parent_section_id,
                "section_index": parent_index,
                "header_path": "",
                "supporting_chunk_ids": [],
                "parent_index": parent_index,
                "parent_length": len(parent_text),
            }
            parent_docs_by_id[parent_section_id] = Document(
                page_content=parent_text,
                metadata=parent_metadata,
            )

            child_texts = child_splitter.split_text(parent_text)
            for child_index, child_text in enumerate(child_texts):
                chunk_index = len(child_docs) - doc_chunk_start
                chunk_id = f"{raw_doc['doc_id']}::chunk_{chunk_index:03d}"
                metadata = {
                    "doc_id": raw_doc["doc_id"],
                    "source_path": raw_doc["source_path"],
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "section_index": parent_index,
                    "header_path": "",
                    "parent_section_id": parent_section_id,
                    "parent_index": parent_index,
                    "child_index": child_index,
                }
                parent_docs_by_id[parent_section_id].metadata["supporting_chunk_ids"].append(chunk_id)
                child_docs.append(Document(page_content=child_text, metadata=metadata))
                chunk_manifest.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": raw_doc["doc_id"],
                        "source_path": raw_doc["source_path"],
                        "chunk_index": chunk_index,
                        "section_index": parent_index,
                        "parent_section_id": parent_section_id,
                        "header_path": "",
                        "headers": build_header_fields(metadata),
                        "text": child_text,
                        "preview": normalize_preview(child_text),
                    }
                )

        doc_manifest.append(
            {
                "doc_id": raw_doc["doc_id"],
                "source_path": raw_doc["source_path"],
                "char_length": len(raw_doc["text"]),
                "chunk_count": len(child_docs) - doc_chunk_start,
                "parent_chunk_count": len(parent_chunks),
            }
        )

    return CorpusBundle(
        chunk_strategy=PARENT_CHILD_CHUNK_STRATEGY,
        child_docs=child_docs,
        parent_docs_by_id=parent_docs_by_id,
        chunk_manifest=chunk_manifest,
        doc_manifest=doc_manifest,
    )


def build_corpus_bundle(chunk_strategy: str = DEFAULT_CHUNK_STRATEGY) -> CorpusBundle:
    if chunk_strategy == DEFAULT_CHUNK_STRATEGY:
        return build_markdown_recursive_bundle()
    if chunk_strategy == PARENT_CHILD_CHUNK_STRATEGY:
        return build_parent_child_bundle()
    raise ValueError(f"unsupported chunk strategy: {chunk_strategy}")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def variant_uses_parent_child_chunking(variant: str) -> bool:
    return variant in PARENT_CHILD_VARIANTS


def chunk_strategy_for_variant(variant: str) -> str:
    if variant_uses_parent_child_chunking(variant):
        return PARENT_CHILD_CHUNK_STRATEGY
    return DEFAULT_CHUNK_STRATEGY


def manifest_paths_for_chunk_strategy(chunk_strategy: str) -> tuple[Path, Path]:
    if chunk_strategy == DEFAULT_CHUNK_STRATEGY:
        return CHUNK_MANIFEST_PATH, DOC_MANIFEST_PATH
    if chunk_strategy == PARENT_CHILD_CHUNK_STRATEGY:
        return PARENT_CHILD_CHUNK_MANIFEST_PATH, PARENT_CHILD_DOC_MANIFEST_PATH
    raise ValueError(f"unsupported chunk strategy: {chunk_strategy}")


def save_bundle_manifests(bundle: CorpusBundle) -> None:
    chunk_manifest_path, doc_manifest_path = manifest_paths_for_chunk_strategy(bundle.chunk_strategy)
    save_json(chunk_manifest_path, bundle.chunk_manifest)
    save_json(doc_manifest_path, bundle.doc_manifest)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def reciprocal_rank(ranked_ids: list[str], gold_ids: set[str], top_k: int) -> float:
    for idx, item_id in enumerate(ranked_ids[:top_k], start=1):
        if item_id in gold_ids:
            return 1.0 / idx
    return 0.0


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], top_k: int) -> float | None:
    if not gold_ids:
        return None
    return len(set(ranked_ids[:top_k]) & gold_ids) / len(gold_ids)


def derive_evidence_chunk_ids(docs: list[Document]) -> list[str]:
    chunk_ids: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        candidate_ids = doc.metadata.get("supporting_chunk_ids")
        if not candidate_ids:
            chunk_id = doc.metadata.get("chunk_id")
            candidate_ids = [chunk_id] if chunk_id else []
        for chunk_id in candidate_ids:
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                chunk_ids.append(chunk_id)
    return chunk_ids


def normalize_search_output(search_output: SearchResult | list[Document]) -> SearchResult:
    if isinstance(search_output, SearchResult):
        if search_output.evidence_chunk_ids is None:
            search_output.evidence_chunk_ids = derive_evidence_chunk_ids(search_output.docs)
        return search_output
    return SearchResult(
        docs=search_output,
        evidence_chunk_ids=derive_evidence_chunk_ids(search_output),
    )


def compute_metrics_for_question(
    qa_item: dict[str, Any],
    search_result: SearchResult,
    top_k: int,
    latency_ms: float,
    gold_chunk_field: str = "gold_chunk_ids",
) -> dict[str, Any]:
    ranked_doc_ids = dedupe_preserve_order([doc.metadata["doc_id"] for doc in search_result.docs])
    ranked_chunk_ids = (search_result.evidence_chunk_ids or [])[:]

    gold_doc_ids = set(qa_item.get("gold_doc_ids", []))
    gold_chunk_ids = set(qa_item.get(gold_chunk_field, []))

    doc_hit = int(any(doc_id in gold_doc_ids for doc_id in ranked_doc_ids[:top_k]))
    doc_mrr = reciprocal_rank(ranked_doc_ids, gold_doc_ids, top_k)
    doc_recall = recall_at_k(ranked_doc_ids, gold_doc_ids, top_k)

    chunk_metrics = {}
    if gold_chunk_ids:
        chunk_metrics = {
            "chunk_hit": int(any(chunk_id in gold_chunk_ids for chunk_id in ranked_chunk_ids[:top_k])),
            "chunk_mrr": reciprocal_rank(ranked_chunk_ids, gold_chunk_ids, top_k),
            "chunk_recall": recall_at_k(ranked_chunk_ids, gold_chunk_ids, top_k),
        }

    result = {
        "question_id": qa_item["question_id"],
        "question_type": qa_item["question_type"],
        "question": qa_item["question"],
        "gold_doc_ids": qa_item.get("gold_doc_ids", []),
        "gold_chunk_ids": qa_item.get(gold_chunk_field, []),
        "retrieved_doc_ids": ranked_doc_ids[:top_k],
        "retrieved_chunk_ids": ranked_chunk_ids[:top_k],
        "retrieved_header_paths": [
            doc.metadata.get("header_path", "")
            for doc in search_result.docs[:top_k]
        ],
        "latency_ms": round(latency_ms, 2),
        "doc_hit": doc_hit,
        "doc_mrr": round(doc_mrr, 4),
        "doc_recall": None if doc_recall is None else round(doc_recall, 4),
        **chunk_metrics,
    }
    result.update(search_result.metadata)
    return result


def mean_or_none(values: list[float | None]) -> float | None:
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return None
    return round(statistics.mean(clean_values), 4)


def summarize_results(question_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in question_results:
        by_type[result["question_type"]].append(result)

    def summarize_bucket(bucket: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(bucket),
            "doc_hit_rate": round(statistics.mean(item["doc_hit"] for item in bucket), 4),
            "doc_mrr": round(statistics.mean(item["doc_mrr"] for item in bucket), 4),
            "doc_recall": mean_or_none([item["doc_recall"] for item in bucket]),
            "avg_latency_ms": round(statistics.mean(item["latency_ms"] for item in bucket), 2),
            "chunk_hit_rate": mean_or_none([item.get("chunk_hit") for item in bucket]),
            "chunk_mrr": mean_or_none([item.get("chunk_mrr") for item in bucket]),
            "chunk_recall": mean_or_none([item.get("chunk_recall") for item in bucket]),
        }

    return {
        "overall": summarize_bucket(question_results),
        "by_question_type": {question_type: summarize_bucket(bucket) for question_type, bucket in by_type.items()},
    }


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    overall = summary["overall"]
    print("overall:", json.dumps(overall, ensure_ascii=False))
    for question_type, bucket in summary["by_question_type"].items():
        print(f"{question_type}:", json.dumps(bucket, ensure_ascii=False))


class DenseEmbeddingProvider:
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        local_files_only: bool = False,
    ):
        if HuggingFaceEmbeddings is None:
            raise ImportError("缺少 HuggingFaceEmbeddings，无法执行 dense 检索。")

        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        model_ref = model_path or model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_ref,
            model_kwargs={"device": "cpu", "local_files_only": local_files_only},
            encode_kwargs={"normalize_embeddings": True},
        )


class SparseRunner:
    def __init__(self, chunk_docs: list[Document], candidate_k: int):
        if BM25Retriever is None:
            raise ImportError("缺少 BM25Retriever，无法执行 sparse 检索。")
        self.retriever = BM25Retriever.from_documents(chunk_docs, k=candidate_k)

    def search(self, query: str, top_k: int) -> SearchResult:
        docs = self.retriever.invoke(query)[:top_k]
        return SearchResult(docs=docs)


class DenseRunner:
    def __init__(
        self,
        chunk_docs: list[Document],
        candidate_k: int,
        embedding_provider: DenseEmbeddingProvider,
    ):
        if FAISS is None:
            raise ImportError("缺少 FAISS，无法执行 dense 检索。")
        self.vectorstore = FAISS.from_documents(chunk_docs, embedding_provider.embeddings)
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int) -> SearchResult:
        docs = self.vectorstore.similarity_search(query, k=min(top_k, self.candidate_k))
        return SearchResult(docs=docs)


class ScoredSparseRunner:
    def __init__(self, chunk_docs: list[Document]):
        if BM25Retriever is None:
            raise ImportError("缺少 BM25Retriever，无法执行 sparse 检索。")
        self.retriever = BM25Retriever.from_documents(chunk_docs, k=len(chunk_docs))

    def search_scored(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        query_tokens = self.retriever.preprocess_func(query)
        scores = self.retriever.vectorizer.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [(self.retriever.docs[idx], float(scores[idx])) for idx in ranked_indices]


class ScoredDenseRunner:
    def __init__(self, chunk_docs: list[Document], embedding_provider: DenseEmbeddingProvider):
        if FAISS is None:
            raise ImportError("缺少 FAISS，无法执行 dense 检索。")
        self.vectorstore = FAISS.from_documents(chunk_docs, embedding_provider.embeddings)

    def search_scored(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        scored_pairs = self.vectorstore.similarity_search_with_score(query, k=top_k)
        results: list[tuple[Document, float]] = []
        for doc, distance in scored_pairs:
            similarity = 1.0 / (1.0 + float(distance))
            results.append((doc, similarity))
        return results


def rrf_fuse(result_lists: list[list[Document]], rrf_k: int) -> list[Document]:
    scores: dict[str, float] = {}
    docs_by_chunk_id: dict[str, Document] = {}
    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            chunk_id = doc.metadata["chunk_id"]
            docs_by_chunk_id[chunk_id] = doc
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    reranked: list[Document] = []
    for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        base_doc = docs_by_chunk_id[chunk_id]
        metadata = dict(base_doc.metadata)
        metadata["rrf_score"] = round(score, 6)
        reranked.append(Document(page_content=base_doc.page_content, metadata=metadata))
    return reranked


def scored_docs_to_docs(scored_docs: list[tuple[Document, float]]) -> list[Document]:
    return [doc for doc, _ in scored_docs]


def scored_docs_to_score_map(scored_docs: list[tuple[Document, float]]) -> dict[str, float]:
    return {doc.metadata["chunk_id"]: score for doc, score in scored_docs}


def normalize_score_map(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return {chunk_id: 1.0 for chunk_id in score_map}
    return {
        chunk_id: (score - min_value) / (max_value - min_value)
        for chunk_id, score in score_map.items()
    }


def build_ranked_docs_from_score_map(
    score_map: dict[str, float],
    docs_by_chunk_id: dict[str, Document],
    extra_score_maps: dict[str, dict[str, float]] | None = None,
) -> list[Document]:
    ranked_docs: list[Document] = []
    for chunk_id, fused_score in sorted(score_map.items(), key=lambda item: item[1], reverse=True):
        base_doc = docs_by_chunk_id[chunk_id]
        metadata = dict(base_doc.metadata)
        metadata["fusion_score"] = round(fused_score, 6)
        for field_name, field_score_map in (extra_score_maps or {}).items():
            if chunk_id in field_score_map:
                metadata[field_name] = round(field_score_map[chunk_id], 6)
        ranked_docs.append(Document(page_content=base_doc.page_content, metadata=metadata))
    return ranked_docs


class HybridRunner:
    def __init__(
        self,
        sparse_runner: SparseRunner,
        dense_runner: DenseRunner,
        sparse_candidate_k: int,
        dense_candidate_k: int,
        rrf_k: int,
    ):
        self.sparse_runner = sparse_runner
        self.dense_runner = dense_runner
        self.sparse_candidate_k = sparse_candidate_k
        self.dense_candidate_k = dense_candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> SearchResult:
        sparse_docs = normalize_search_output(
            self.sparse_runner.search(query, self.sparse_candidate_k)
        ).docs
        dense_docs = normalize_search_output(
            self.dense_runner.search(query, self.dense_candidate_k)
        ).docs
        docs = rrf_fuse([dense_docs, sparse_docs], self.rrf_k)[:top_k]
        return SearchResult(docs=docs)


class Group4HybridRunner:
    def __init__(
        self,
        strategy: str,
        sparse_runner: ScoredSparseRunner,
        dense_runner: ScoredDenseRunner,
        dense_candidate_ranker: DenseCandidateRanker,
        sparse_candidate_ranker: BM25CandidateRanker,
        dense_candidate_k: int,
        sparse_candidate_k: int,
        rrf_k: int,
        alpha: float,
    ):
        self.strategy = strategy
        self.sparse_runner = sparse_runner
        self.dense_runner = dense_runner
        self.dense_candidate_ranker = dense_candidate_ranker
        self.sparse_candidate_ranker = sparse_candidate_ranker
        self.dense_candidate_k = dense_candidate_k
        self.sparse_candidate_k = sparse_candidate_k
        self.rrf_k = rrf_k
        self.alpha = alpha

    def search(self, query: str, top_k: int) -> SearchResult:
        dense_scored = self.dense_runner.search_scored(query, self.dense_candidate_k)
        sparse_scored = self.sparse_runner.search_scored(query, self.sparse_candidate_k)
        dense_docs = scored_docs_to_docs(dense_scored)
        sparse_docs = scored_docs_to_docs(sparse_scored)

        if self.strategy == "union_rrf":
            docs = rrf_fuse([dense_docs, sparse_docs], self.rrf_k)[:top_k]
            return SearchResult(
                docs=docs,
                metadata={
                    "fusion_strategy": self.strategy,
                    "dense_candidate_k": self.dense_candidate_k,
                    "sparse_candidate_k": self.sparse_candidate_k,
                    "rrf_k": self.rrf_k,
                },
            )

        if self.strategy == "weighted_score_fusion":
            dense_score_map = normalize_score_map(scored_docs_to_score_map(dense_scored))
            sparse_score_map = normalize_score_map(scored_docs_to_score_map(sparse_scored))
            docs_by_chunk_id = {
                doc.metadata["chunk_id"]: doc
                for doc in dense_docs + sparse_docs
            }
            combined_scores: dict[str, float] = {}
            for chunk_id in docs_by_chunk_id:
                combined_scores[chunk_id] = (
                    self.alpha * dense_score_map.get(chunk_id, 0.0)
                    + (1.0 - self.alpha) * sparse_score_map.get(chunk_id, 0.0)
                )
            docs = build_ranked_docs_from_score_map(
                combined_scores,
                docs_by_chunk_id,
                extra_score_maps={
                    "dense_score_norm": dense_score_map,
                    "sparse_score_norm": sparse_score_map,
                },
            )[:top_k]
            return SearchResult(
                docs=docs,
                metadata={
                    "fusion_strategy": self.strategy,
                    "dense_candidate_k": self.dense_candidate_k,
                    "sparse_candidate_k": self.sparse_candidate_k,
                    "alpha": round(self.alpha, 4),
                },
            )

        if self.strategy == "sparse_first_dense_rerank":
            candidate_docs = sparse_docs[: self.sparse_candidate_k]
            docs = self.dense_candidate_ranker.rank(query, candidate_docs)[:top_k]
            return SearchResult(
                docs=docs,
                metadata={
                    "fusion_strategy": self.strategy,
                    "first_stage": "sparse",
                    "second_stage": "dense",
                    "sparse_candidate_k": self.sparse_candidate_k,
                },
            )

        if self.strategy == "dense_first_sparse_rerank":
            candidate_docs = dense_docs[: self.dense_candidate_k]
            docs = self.sparse_candidate_ranker.rank(query, candidate_docs)[:top_k]
            return SearchResult(
                docs=docs,
                metadata={
                    "fusion_strategy": self.strategy,
                    "first_stage": "dense",
                    "second_stage": "sparse",
                    "dense_candidate_k": self.dense_candidate_k,
                },
            )

        if self.strategy == "intersection_rrf":
            intersection_ids = {
                doc.metadata["chunk_id"] for doc in dense_docs
            } & {
                doc.metadata["chunk_id"] for doc in sparse_docs
            }
            if not intersection_ids:
                return SearchResult(
                    docs=[],
                    metadata={
                        "fusion_strategy": self.strategy,
                        "dense_candidate_k": self.dense_candidate_k,
                        "sparse_candidate_k": self.sparse_candidate_k,
                        "intersection_size": 0,
                    },
                )
            dense_intersection = [doc for doc in dense_docs if doc.metadata["chunk_id"] in intersection_ids]
            sparse_intersection = [doc for doc in sparse_docs if doc.metadata["chunk_id"] in intersection_ids]
            docs = rrf_fuse([dense_intersection, sparse_intersection], self.rrf_k)[:top_k]
            return SearchResult(
                docs=docs,
                metadata={
                    "fusion_strategy": self.strategy,
                    "dense_candidate_k": self.dense_candidate_k,
                    "sparse_candidate_k": self.sparse_candidate_k,
                    "rrf_k": self.rrf_k,
                    "intersection_size": len(intersection_ids),
                },
            )

        raise ValueError(f"unsupported group4 strategy: {self.strategy}")


class GenericQueryRewriter:
    _shared_cache: dict[str, str] = {}

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm = None
        self.chain = None
        self._setup()

    def _setup(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or ChatDeepSeek is None or PromptTemplate is None or StrOutputParser is None:
            return

        self.llm = ChatDeepSeek(
            model=self.model_name,
            temperature=0.0,
            max_tokens=128,
            api_key=api_key,
        )
        prompt = PromptTemplate(
            template="""
你是一个 RAG 检索查询优化助手，目标是帮助用户在一套 RAG 教程文档中更容易检索到相关段落。

文档主题主要包括：
- RAG 基础概念
- 文本分块
- 向量嵌入
- 向量数据库
- 入门环境配置

请根据以下规则决定是否重写用户问题：
1. 如果问题已经具体、准确、可直接检索，返回原问题。
2. 如果问题过于口语、模糊或缺少关键术语，请在不改变原意的前提下补充更适合检索的表达。
3. 保持简洁，不要扩写成多句话。
4. 不要回答问题，只输出最终查询。

示例：
- “文档怎么切比较合适？” -> “文本分块策略和 MarkdownHeader + RecursiveCharacter 分块”
- “我想先搭个能跑起来的 RAG 原型” -> “最小可行 RAG 原型搭建与本地向量存储方案”
- “RAG 是什么，它结合了哪两类知识？” -> “RAG 是什么，它结合了哪两类知识？”

原问题：{query}

最终查询：""",
            input_variables=["query"],
        )
        self.chain = prompt | self.llm | StrOutputParser()

    def rewrite(self, query: str) -> str:
        if query in self._shared_cache:
            return self._shared_cache[query]
        if self.chain is None:
            return query
        rewritten = self.chain.invoke({"query": query}).strip()
        final_query = rewritten or query
        self._shared_cache[query] = final_query
        return final_query


class QueryRewriteRunner:
    def __init__(self, base_runner: Any, query_rewriter: GenericQueryRewriter):
        self.base_runner = base_runner
        self.query_rewriter = query_rewriter

    def search(self, query: str, top_k: int) -> SearchResult:
        rewritten_query = self.query_rewriter.rewrite(query)
        result = normalize_search_output(self.base_runner.search(rewritten_query, top_k))
        metadata = dict(result.metadata)
        metadata.update(
            {
                "raw_query": query,
                "effective_query": rewritten_query,
                "query_rewritten": int(rewritten_query != query),
            }
        )
        return SearchResult(
            docs=result.docs,
            evidence_chunk_ids=result.evidence_chunk_ids,
            metadata=metadata,
        )


class ParentChildRunner:
    def __init__(self, base_runner: Any, parent_docs_by_id: dict[str, Document], child_candidate_k: int):
        self.base_runner = base_runner
        self.parent_docs_by_id = parent_docs_by_id
        self.child_candidate_k = child_candidate_k

    def search(self, query: str, top_k: int) -> SearchResult:
        child_result = normalize_search_output(self.base_runner.search(query, self.child_candidate_k))
        parent_scores: dict[str, float] = {}
        parent_supports: dict[str, list[str]] = {}

        for rank, doc in enumerate(child_result.docs, start=1):
            parent_id = doc.metadata.get("parent_section_id")
            child_chunk_id = doc.metadata.get("chunk_id")
            if not parent_id or not child_chunk_id:
                continue
            parent_scores[parent_id] = parent_scores.get(parent_id, 0.0) + 1.0 / rank
            parent_supports.setdefault(parent_id, [])
            if child_chunk_id not in parent_supports[parent_id]:
                parent_supports[parent_id].append(child_chunk_id)

        reranked_parents: list[Document] = []
        evidence_chunk_ids: list[str] = []
        for parent_id, score in sorted(parent_scores.items(), key=lambda item: item[1], reverse=True):
            parent_doc = self.parent_docs_by_id[parent_id]
            metadata = dict(parent_doc.metadata)
            metadata["parent_child_score"] = round(score, 6)
            metadata["supporting_chunk_ids"] = parent_supports.get(parent_id, [])
            reranked_parents.append(Document(page_content=parent_doc.page_content, metadata=metadata))
            for chunk_id in metadata["supporting_chunk_ids"]:
                if chunk_id not in evidence_chunk_ids:
                    evidence_chunk_ids.append(chunk_id)

        return SearchResult(
            docs=reranked_parents[:top_k],
            evidence_chunk_ids=evidence_chunk_ids,
        )


class BM25CandidateRanker:
    def rank(self, query: str, candidate_docs: list[Document]) -> list[Document]:
        if BM25Retriever is None:
            raise ImportError("缺少 BM25Retriever，无法执行 BM25 候选重排。")
        retriever = BM25Retriever.from_documents(candidate_docs, k=len(candidate_docs))
        return retriever.invoke(query)


class DenseCandidateRanker:
    def __init__(self, embedding_provider: DenseEmbeddingProvider):
        self.embedding_provider = embedding_provider

    def rank(self, query: str, candidate_docs: list[Document]) -> list[Document]:
        if FAISS is None:
            raise ImportError("缺少 FAISS，无法执行 dense 候选重排。")
        vectorstore = FAISS.from_documents(candidate_docs, self.embedding_provider.embeddings)
        return vectorstore.similarity_search(query, k=len(candidate_docs))


class RRFRerankRunner:
    def __init__(
        self,
        base_runner: Any,
        candidate_rankers: list[Any],
        candidate_k: int,
        rrf_k: int,
    ):
        self.base_runner = base_runner
        self.candidate_rankers = candidate_rankers
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> SearchResult:
        base_result = normalize_search_output(self.base_runner.search(query, self.candidate_k))
        candidate_docs = base_result.docs
        ranked_lists = [candidate_docs]
        for candidate_ranker in self.candidate_rankers:
            ranked_lists.append(candidate_ranker.rank(query, candidate_docs))
        reranked_docs = rrf_fuse(ranked_lists, self.rrf_k)[:top_k]
        return SearchResult(docs=reranked_docs)


def build_rrf_runner(
    mode: str,
    base_runner: Any,
    args: argparse.Namespace,
    dense_provider: DenseEmbeddingProvider | None,
) -> RRFRerankRunner:
    candidate_rankers: list[Any] = []
    if mode in {"dense", "hybrid"}:
        candidate_rankers.append(BM25CandidateRanker())
    if mode in {"sparse", "hybrid"}:
        if dense_provider is None:
            raise ValueError("dense_provider is required for RRF rerank on sparse/hybrid.")
        candidate_rankers.append(DenseCandidateRanker(dense_provider))
    return RRFRerankRunner(
        base_runner=base_runner,
        candidate_rankers=candidate_rankers,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )


def needs_dense_provider(mode: str, variant: str) -> bool:
    if mode in {"dense", "hybrid"}:
        return True
    if variant in {"rrf", "query_rewrite_rrf", "rrf_parent_child", "all_in"}:
        return True
    return False


def get_stage_name(variant: str) -> str:
    if variant == "base":
        return "stage_a"
    if variant in {"query_rewrite", "parent_child", "rrf"}:
        return "stage_b"
    return "stage_c"


def get_group1_output_root() -> Path:
    return OUTPUTS_DIR / "group1"


def get_stage_output_dir(stage_name: str) -> Path:
    return get_group1_output_root() / stage_name


def get_output_stem(mode: str, variant: str) -> str:
    if variant == "base":
        return f"results_{mode}"
    return f"results_{mode}_{variant}"


def get_output_path(mode: str, variant: str) -> Path:
    return get_stage_output_dir(get_stage_name(variant)) / f"{get_output_stem(mode, variant)}.json"


def get_group4_output_root() -> Path:
    return OUTPUTS_DIR / "group4"


def get_group4_output_path(variant: str) -> Path:
    return get_group4_output_root() / f"results_{variant}.json"


def get_group4_preset_runs_root() -> Path:
    return get_group4_output_root() / "preset_runs"


def get_group4_preset_dir(preset_id: str) -> Path:
    return get_group4_preset_runs_root() / preset_id


def get_group4_preset_output_path(preset_id: str, variant: str) -> Path:
    return get_group4_preset_dir(preset_id) / f"results_{variant}.json"


def evaluate_with_runner(
    mode: str,
    variant: str,
    runner: Any,
    qa_items: list[dict[str, Any]],
    top_k: int,
    output_path: Path,
    gold_chunk_field: str = "gold_chunk_ids",
) -> dict[str, Any]:
    question_results = []
    for qa_item in qa_items:
        started = time.perf_counter()
        search_result = normalize_search_output(runner.search(qa_item["question"], top_k=top_k))
        latency_ms = (time.perf_counter() - started) * 1000
        question_results.append(
            compute_metrics_for_question(
                qa_item=qa_item,
                search_result=search_result,
                top_k=top_k,
                latency_ms=latency_ms,
                gold_chunk_field=gold_chunk_field,
            )
        )

    summary = summarize_results(question_results)
    title = f"{mode}_{variant}" if variant != "base" else mode
    payload = {
        "mode": mode,
        "variant": variant,
        "stage": get_stage_name(variant),
        "top_k": top_k,
        "question_count": len(question_results),
        "output_file": output_path.relative_to(Path(__file__).resolve().parent).as_posix(),
        "summary": summary,
        "per_question": question_results,
    }
    print_summary(title, summary)
    save_json(output_path, payload)
    return payload


def build_run_specs(stage: str) -> list[RunSpec]:
    stage_a_specs = [RunSpec(mode=mode, variant="base") for mode in ["sparse", "dense", "hybrid"]]
    stage_b_specs = [
        RunSpec(mode=mode, variant=variant)
        for mode in ["sparse", "dense", "hybrid"]
        for variant in ["query_rewrite", "parent_child", "rrf"]
    ]
    stage_c_specs = [
        RunSpec(mode=mode, variant=variant)
        for mode in ["dense", "hybrid"]
        for variant in [
            "query_rewrite_rrf",
            "query_rewrite_parent_child",
            "rrf_parent_child",
            "all_in",
        ]
    ]
    if stage == "stage_a":
        return stage_a_specs
    if stage == "stage_b":
        return stage_b_specs
    if stage == "stage_c":
        return stage_c_specs
    if stage == "full":
        return stage_a_specs + stage_b_specs + stage_c_specs
    raise ValueError(f"unsupported stage: {stage}")


def evaluate_group4_runner(
    variant: str,
    runner: Any,
    qa_items: list[dict[str, Any]],
    top_k: int,
    output_path: Path,
) -> dict[str, Any]:
    question_results = []
    for qa_item in qa_items:
        started = time.perf_counter()
        search_result = normalize_search_output(runner.search(qa_item["question"], top_k=top_k))
        latency_ms = (time.perf_counter() - started) * 1000
        question_results.append(
            compute_metrics_for_question(
                qa_item=qa_item,
                search_result=search_result,
                top_k=top_k,
                latency_ms=latency_ms,
                gold_chunk_field="gold_chunk_ids",
            )
        )

    summary = summarize_results(question_results)
    payload = {
        "group": "group4",
        "variant": variant,
        "top_k": top_k,
        "question_count": len(question_results),
        "output_file": output_path.relative_to(Path(__file__).resolve().parent).as_posix(),
        "summary": summary,
        "per_question": question_results,
    }
    print_summary(f"group4_{variant}", summary)
    save_json(output_path, payload)
    return payload


def write_group4_summary(payloads: list[dict[str, Any]], args: argparse.Namespace) -> None:
    group4_dir = get_group4_output_root()
    rows = []
    for payload in payloads:
        overall = payload["summary"]["overall"]
        rows.append(
            {
                "variant": payload["variant"],
                "doc_hit_rate": overall["doc_hit_rate"],
                "doc_mrr": overall["doc_mrr"],
                "doc_recall": overall["doc_recall"],
                "chunk_hit_rate": overall["chunk_hit_rate"],
                "chunk_mrr": overall["chunk_mrr"],
                "chunk_recall": overall["chunk_recall"],
                "avg_latency_ms": overall["avg_latency_ms"],
                "question_count": payload["question_count"],
                "output_file": payload["output_file"],
            }
        )
    order = {variant: index for index, variant in enumerate(GROUP4_VARIANTS)}
    rows.sort(key=lambda row: order.get(row["variant"], 999))
    save_json(
        group4_dir / "summary.json",
        {
            "group": "group4",
            "run_count": len(payloads),
            "config": {
                "top_k": args.top_k,
                "dense_candidate_k": args.dense_candidate_k,
                "sparse_candidate_k": args.sparse_candidate_k,
                "rrf_k": args.rrf_k,
                "alpha": args.alpha,
            },
            "runs": rows,
        },
    )
    save_csv(
        group4_dir / "summary.csv",
        rows,
        [
            "variant",
            "doc_hit_rate",
            "doc_mrr",
            "doc_recall",
            "chunk_hit_rate",
            "chunk_mrr",
            "chunk_recall",
            "avg_latency_ms",
            "question_count",
            "output_file",
        ],
    )


def get_group4_preset_map() -> dict[str, Group4Preset]:
    return {preset.preset_id: preset for preset in GROUP4_PARAMETER_PRESETS}


def args_for_group4_preset(args: argparse.Namespace, preset: Group4Preset) -> argparse.Namespace:
    preset_args = argparse.Namespace(**vars(args))
    preset_args.top_k = preset.top_k
    preset_args.dense_candidate_k = preset.dense_candidate_k
    preset_args.sparse_candidate_k = preset.sparse_candidate_k
    preset_args.rrf_k = preset.rrf_k
    preset_args.alpha = preset.alpha
    return preset_args


def write_group4_preset_catalog() -> None:
    payload = {
        "group": "group4",
        "preset_count": len(GROUP4_PARAMETER_PRESETS),
        "presets": [
            {
                "preset_id": preset.preset_id,
                "description": preset.description,
                "dense_candidate_k": preset.dense_candidate_k,
                "sparse_candidate_k": preset.sparse_candidate_k,
                "rrf_k": preset.rrf_k,
                "alpha": preset.alpha,
                "top_k": preset.top_k,
            }
            for preset in GROUP4_PARAMETER_PRESETS
        ],
    }
    save_json(get_group4_preset_runs_root() / "presets.json", payload)


def write_group4_preset_summary(
    payloads_by_preset: dict[str, list[dict[str, Any]]],
    preset_map: dict[str, Group4Preset],
) -> None:
    order = {variant: index for index, variant in enumerate(GROUP4_VARIANTS)}
    summary_rows = []
    preset_entries = []

    for preset_id, payloads in payloads_by_preset.items():
        preset = preset_map[preset_id]
        rows = []
        for payload in payloads:
            overall = payload["summary"]["overall"]
            row = {
                "variant": payload["variant"],
                "doc_hit_rate": overall["doc_hit_rate"],
                "doc_mrr": overall["doc_mrr"],
                "doc_recall": overall["doc_recall"],
                "chunk_hit_rate": overall["chunk_hit_rate"],
                "chunk_mrr": overall["chunk_mrr"],
                "chunk_recall": overall["chunk_recall"],
                "avg_latency_ms": overall["avg_latency_ms"],
                "question_count": payload["question_count"],
                "output_file": payload["output_file"],
            }
            rows.append(row)
            summary_rows.append(
                {
                    "preset_id": preset_id,
                    "variant": payload["variant"],
                    "dense_candidate_k": preset.dense_candidate_k,
                    "sparse_candidate_k": preset.sparse_candidate_k,
                    "rrf_k": preset.rrf_k,
                    "alpha": preset.alpha,
                    "top_k": preset.top_k,
                    **row,
                }
            )

        rows.sort(key=lambda row: order.get(row["variant"], 999))
        preset_payload = {
            "group": "group4",
            "preset_id": preset_id,
            "description": preset.description,
            "config": {
                "dense_candidate_k": preset.dense_candidate_k,
                "sparse_candidate_k": preset.sparse_candidate_k,
                "rrf_k": preset.rrf_k,
                "alpha": preset.alpha,
                "top_k": preset.top_k,
            },
            "run_count": len(rows),
            "runs": rows,
        }
        save_json(get_group4_preset_dir(preset_id) / "summary.json", preset_payload)
        preset_entries.append(preset_payload)

    save_json(
        get_group4_preset_runs_root() / "summary.json",
        {
            "group": "group4",
            "preset_count": len(payloads_by_preset),
            "run_count": len(summary_rows),
            "presets": preset_entries,
            "flat_runs": sorted(
                summary_rows,
                key=lambda row: (row["preset_id"], order.get(row["variant"], 999)),
            ),
        },
    )


def write_stage_summary(stage_name: str, payloads: list[dict[str, Any]]) -> None:
    stage_dir = get_stage_output_dir(stage_name)
    rows = []
    for payload in payloads:
        overall = payload["summary"]["overall"]
        rows.append(
            {
                "stage": payload["stage"],
                "mode": payload["mode"],
                "variant": payload["variant"],
                "doc_hit_rate": overall["doc_hit_rate"],
                "doc_mrr": overall["doc_mrr"],
                "doc_recall": overall["doc_recall"],
                "chunk_hit_rate": overall["chunk_hit_rate"],
                "chunk_mrr": overall["chunk_mrr"],
                "chunk_recall": overall["chunk_recall"],
                "avg_latency_ms": overall["avg_latency_ms"],
                "question_count": payload["question_count"],
                "output_file": payload["output_file"],
            }
        )
    rows.sort(key=lambda row: (row["mode"], row["variant"]))

    summary_json = {
        "stage": stage_name,
        "run_count": len(payloads),
        "runs": rows,
    }
    save_json(stage_dir / "summary.json", summary_json)
    if stage_name != "stage_c":
        save_csv(
            stage_dir / "summary.csv",
            rows,
            [
                "stage",
                "mode",
                "variant",
                "doc_hit_rate",
                "doc_mrr",
                "doc_recall",
                "chunk_hit_rate",
                "chunk_mrr",
                "chunk_recall",
                "avg_latency_ms",
                "question_count",
                "output_file",
            ],
        )


def run_specs(
    specs: list[RunSpec],
    bundles: dict[str, CorpusBundle],
    qa_items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    payloads = []
    dense_provider: DenseEmbeddingProvider | None = None

    if any(needs_dense_provider(spec.mode, spec.variant) for spec in specs):
        dense_provider = DenseEmbeddingProvider(
            model_name=args.dense_model_name,
            model_path=args.dense_model_path,
            local_files_only=args.local_files_only,
        )

    for spec in specs:
        bundle = bundles[chunk_strategy_for_variant(spec.variant)]
        gold_chunk_field = (
            "gold_chunk_ids_parent_child"
            if bundle.chunk_strategy == PARENT_CHILD_CHUNK_STRATEGY
            else "gold_chunk_ids"
        )
        base_runner = build_base_runner(spec.mode, bundle, args, dense_provider)
        runner = wrap_variant_runner(spec.mode, spec.variant, base_runner, bundle, args, dense_provider)
        output_path = get_output_path(spec.mode, spec.variant)
        payload = evaluate_with_runner(
            mode=spec.mode,
            variant=spec.variant,
            runner=runner,
            qa_items=qa_items,
            top_k=args.top_k,
            output_path=output_path,
            gold_chunk_field=gold_chunk_field,
        )
        payloads.append(payload)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        grouped[payload["stage"]].append(payload)
    for stage_name, stage_payloads in grouped.items():
        write_stage_summary(stage_name, stage_payloads)
    return payloads


def build_base_runner(
    mode: str,
    bundle: CorpusBundle,
    args: argparse.Namespace,
    dense_provider: DenseEmbeddingProvider | None,
) -> Any:
    if mode == "sparse":
        return SparseRunner(bundle.child_docs, args.candidate_k)
    if mode == "dense":
        if dense_provider is None:
            raise ValueError("dense_provider is required for dense mode.")
        return DenseRunner(bundle.child_docs, args.candidate_k, dense_provider)
    if mode == "hybrid":
        if dense_provider is None:
            raise ValueError("dense_provider is required for hybrid mode.")
        sparse_runner = SparseRunner(bundle.child_docs, args.candidate_k)
        dense_runner = DenseRunner(bundle.child_docs, args.candidate_k, dense_provider)
        return HybridRunner(
            sparse_runner=sparse_runner,
            dense_runner=dense_runner,
            sparse_candidate_k=args.candidate_k,
            dense_candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )
    raise ValueError(f"unsupported mode: {mode}")


def wrap_variant_runner(
    mode: str,
    variant: str,
    base_runner: Any,
    bundle: CorpusBundle,
    args: argparse.Namespace,
    dense_provider: DenseEmbeddingProvider | None,
) -> Any:
    if variant == "base":
        return base_runner
    if variant == "query_rewrite":
        query_rewriter = GenericQueryRewriter(args.rewrite_model_name)
        return QueryRewriteRunner(base_runner, query_rewriter)
    if variant == "parent_child":
        return ParentChildRunner(
            base_runner=base_runner,
            parent_docs_by_id=bundle.parent_docs_by_id,
            child_candidate_k=args.parent_child_k,
        )
    if variant == "rrf":
        return build_rrf_runner(mode, base_runner, args, dense_provider)
    if variant == "query_rewrite_rrf":
        query_rewriter = GenericQueryRewriter(args.rewrite_model_name)
        rrf_runner = build_rrf_runner(mode, base_runner, args, dense_provider)
        return QueryRewriteRunner(rrf_runner, query_rewriter)
    if variant == "query_rewrite_parent_child":
        query_rewriter = GenericQueryRewriter(args.rewrite_model_name)
        parent_child_runner = ParentChildRunner(
            base_runner=base_runner,
            parent_docs_by_id=bundle.parent_docs_by_id,
            child_candidate_k=args.parent_child_k,
        )
        return QueryRewriteRunner(parent_child_runner, query_rewriter)
    if variant == "rrf_parent_child":
        rrf_runner = build_rrf_runner(mode, base_runner, args, dense_provider)
        return ParentChildRunner(
            base_runner=rrf_runner,
            parent_docs_by_id=bundle.parent_docs_by_id,
            child_candidate_k=args.parent_child_k,
        )
    if variant == "all_in":
        query_rewriter = GenericQueryRewriter(args.rewrite_model_name)
        rrf_runner = build_rrf_runner(mode, base_runner, args, dense_provider)
        parent_child_runner = ParentChildRunner(
            base_runner=rrf_runner,
            parent_docs_by_id=bundle.parent_docs_by_id,
            child_candidate_k=args.parent_child_k,
        )
        return QueryRewriteRunner(parent_child_runner, query_rewriter)
    raise ValueError(f"unsupported variant: {variant}")


def command_prepare_chunks(_: argparse.Namespace) -> None:
    ensure_dirs()
    bundle = build_corpus_bundle()
    save_bundle_manifests(bundle)
    print(f"documents: {len(bundle.doc_manifest)}")
    print(f"chunks: {len(bundle.chunk_manifest)}")
    print(f"chunk manifest saved to: {CHUNK_MANIFEST_PATH}")
    print(f"doc manifest saved to: {DOC_MANIFEST_PATH}")


def command_evaluate(args: argparse.Namespace) -> None:
    ensure_dirs()
    modes = [args.mode] if args.mode != "all" else ["sparse", "dense", "hybrid"]
    specs = [RunSpec(mode=mode, variant=args.variant) for mode in modes]
    bundle_strategies = {chunk_strategy_for_variant(spec.variant) for spec in specs}
    bundles = {strategy: build_corpus_bundle(strategy) for strategy in bundle_strategies}
    for bundle in bundles.values():
        save_bundle_manifests(bundle)
    if PARENT_CHILD_CHUNK_STRATEGY in bundle_strategies:
        print("note: parent-child retrieval now uses parent-child chunking; regenerate gold_chunk_ids before trusting chunk-level metrics.")
    qa_items = load_qa_items()
    run_specs(specs, bundles, qa_items, args)


def command_run_group1(args: argparse.Namespace) -> None:
    ensure_dirs()
    specs = build_run_specs(args.stage)
    bundle_strategies = {chunk_strategy_for_variant(spec.variant) for spec in specs}
    bundles = {strategy: build_corpus_bundle(strategy) for strategy in bundle_strategies}
    for bundle in bundles.values():
        save_bundle_manifests(bundle)
    if PARENT_CHILD_CHUNK_STRATEGY in bundle_strategies:
        print("note: parent-child retrieval now uses parent-child chunking; regenerate gold_chunk_ids before trusting chunk-level metrics.")
    qa_items = load_qa_items()
    run_specs(specs, bundles, qa_items, args)


def build_group4_runner(
    variant: str,
    bundle: CorpusBundle,
    args: argparse.Namespace,
    dense_provider: DenseEmbeddingProvider,
) -> Group4HybridRunner:
    sparse_runner = ScoredSparseRunner(bundle.child_docs)
    dense_runner = ScoredDenseRunner(bundle.child_docs, dense_provider)
    dense_candidate_ranker = DenseCandidateRanker(dense_provider)
    sparse_candidate_ranker = BM25CandidateRanker()
    return Group4HybridRunner(
        strategy=variant,
        sparse_runner=sparse_runner,
        dense_runner=dense_runner,
        dense_candidate_ranker=dense_candidate_ranker,
        sparse_candidate_ranker=sparse_candidate_ranker,
        dense_candidate_k=args.dense_candidate_k,
        sparse_candidate_k=args.sparse_candidate_k,
        rrf_k=args.rrf_k,
        alpha=args.alpha,
    )


def command_run_group4(args: argparse.Namespace) -> None:
    ensure_dirs()
    bundle = build_corpus_bundle(DEFAULT_CHUNK_STRATEGY)
    save_bundle_manifests(bundle)
    qa_items = load_qa_items()
    dense_provider = DenseEmbeddingProvider(
        model_name=args.dense_model_name,
        model_path=args.dense_model_path,
        local_files_only=args.local_files_only,
    )
    variants = GROUP4_VARIANTS if args.variant == "all" else [args.variant]
    payloads = []
    for variant in variants:
        runner = build_group4_runner(variant, bundle, args, dense_provider)
        output_path = get_group4_output_path(variant)
        payload = evaluate_group4_runner(
            variant=variant,
            runner=runner,
            qa_items=qa_items,
            top_k=args.top_k,
            output_path=output_path,
        )
        payloads.append(payload)
    write_group4_summary(payloads, args)


def command_run_group4_presets(args: argparse.Namespace) -> None:
    ensure_dirs()
    bundle = build_corpus_bundle(DEFAULT_CHUNK_STRATEGY)
    save_bundle_manifests(bundle)
    qa_items = load_qa_items()
    dense_provider = DenseEmbeddingProvider(
        model_name=args.dense_model_name,
        model_path=args.dense_model_path,
        local_files_only=args.local_files_only,
    )
    preset_map = get_group4_preset_map()
    selected_presets = (
        GROUP4_PARAMETER_PRESETS
        if args.preset == "all"
        else [preset_map[args.preset]]
    )
    variants = GROUP4_VARIANTS if args.variant == "all" else [args.variant]
    payloads_by_preset: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for preset in selected_presets:
        preset_args = args_for_group4_preset(args, preset)
        for variant in variants:
            runner = build_group4_runner(variant, bundle, preset_args, dense_provider)
            output_path = get_group4_preset_output_path(preset.preset_id, variant)
            payload = evaluate_group4_runner(
                variant=variant,
                runner=runner,
                qa_items=qa_items,
                top_k=preset.top_k,
                output_path=output_path,
            )
            payload["preset_id"] = preset.preset_id
            payload["preset_description"] = preset.description
            payload["preset_config"] = {
                "dense_candidate_k": preset.dense_candidate_k,
                "sparse_candidate_k": preset.sparse_candidate_k,
                "rrf_k": preset.rrf_k,
                "alpha": preset.alpha,
                "top_k": preset.top_k,
            }
            save_json(output_path, payload)
            payloads_by_preset[preset.preset_id].append(payload)

    write_group4_preset_catalog()
    write_group4_preset_summary(payloads_by_preset, preset_map)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal retrieval eval scaffold for Group 1 experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-chunks", help="Chunk the fixed markdown corpus and save manifests.")
    prepare_parser.set_defaults(func=command_prepare_chunks)

    eval_parser = subparsers.add_parser("evaluate", help="Run retrieval evaluation on the fixed QA set.")
    eval_parser.add_argument("--mode", choices=["sparse", "dense", "hybrid", "all"], default="sparse")
    eval_parser.add_argument(
        "--variant",
        choices=[
            "base",
            "query_rewrite",
            "parent_child",
            "rrf",
            "query_rewrite_rrf",
            "query_rewrite_parent_child",
            "rrf_parent_child",
            "all_in",
        ],
        default="base",
    )
    eval_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    eval_parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    eval_parser.add_argument("--parent-child-k", type=int, default=DEFAULT_PARENT_CHILD_K)
    eval_parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    eval_parser.add_argument("--dense-model-name", default=DEFAULT_DENSE_MODEL_NAME)
    eval_parser.add_argument("--dense-model-path")
    eval_parser.add_argument("--rewrite-model-name", default=DEFAULT_REWRITE_MODEL_NAME)
    eval_parser.add_argument("--local-files-only", action="store_true")
    eval_parser.set_defaults(func=command_evaluate)

    group1_parser = subparsers.add_parser("run-group1", help="Run the predefined Group 1 experiment matrix.")
    group1_parser.add_argument("--stage", choices=["stage_a", "stage_b", "stage_c", "full"], default="full")
    group1_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    group1_parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    group1_parser.add_argument("--parent-child-k", type=int, default=DEFAULT_PARENT_CHILD_K)
    group1_parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    group1_parser.add_argument("--dense-model-name", default=DEFAULT_DENSE_MODEL_NAME)
    group1_parser.add_argument("--dense-model-path")
    group1_parser.add_argument("--rewrite-model-name", default=DEFAULT_REWRITE_MODEL_NAME)
    group1_parser.add_argument("--local-files-only", action="store_true")
    group1_parser.set_defaults(func=command_run_group1)

    group4_parser = subparsers.add_parser("run-group4", help="Run Group 4 hybrid fusion experiments.")
    group4_parser.add_argument(
        "--variant",
        choices=["all", *GROUP4_VARIANTS],
        default="all",
    )
    group4_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    group4_parser.add_argument("--dense-candidate-k", type=int, default=DEFAULT_GROUP4_DENSE_CANDIDATE_K)
    group4_parser.add_argument("--sparse-candidate-k", type=int, default=DEFAULT_GROUP4_SPARSE_CANDIDATE_K)
    group4_parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    group4_parser.add_argument("--alpha", type=float, default=DEFAULT_GROUP4_ALPHA)
    group4_parser.add_argument("--dense-model-name", default=DEFAULT_DENSE_MODEL_NAME)
    group4_parser.add_argument("--dense-model-path")
    group4_parser.add_argument("--local-files-only", action="store_true")
    group4_parser.set_defaults(func=command_run_group4)

    group4_preset_parser = subparsers.add_parser(
        "run-group4-presets",
        help="Run Group 4 hybrid fusion experiments with 10 predefined parameter presets.",
    )
    group4_preset_parser.add_argument(
        "--preset",
        choices=["all", *[preset.preset_id for preset in GROUP4_PARAMETER_PRESETS]],
        default="all",
    )
    group4_preset_parser.add_argument(
        "--variant",
        choices=["all", *GROUP4_VARIANTS],
        default="all",
    )
    group4_preset_parser.add_argument("--dense-model-name", default=DEFAULT_DENSE_MODEL_NAME)
    group4_preset_parser.add_argument("--dense-model-path")
    group4_preset_parser.add_argument("--local-files-only", action="store_true")
    group4_preset_parser.set_defaults(func=command_run_group4_presets)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
