from __future__ import annotations

import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import mean, median, pvariance
from typing import Any

import numpy as np

try:
    from langchain_text_splitters import (
        CharacterTextSplitter,
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
except ImportError:
    MarkdownHeaderTextSplitter = None

    try:
        from langchain.text_splitter import (
            CharacterTextSplitter,
            RecursiveCharacterTextSplitter,
        )
    except ImportError:
        CharacterTextSplitter = None
        RecursiveCharacterTextSplitter = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:
    HuggingFaceEmbeddings = None
    SemanticChunker = None


PREVIEW_COUNT = 3
PREVIEW_WIDTH = 100
SPAN_BACKTRACK = 120
COMMON_SEPARATORS = ["\n\n", "\n", "。", "，", " ", ""]
MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]
BREAKPOINT_THRESHOLD_OPTIONS = [
    "percentile",
    "gradient",
    "standard_deviation",
    "interquartile",
]
HEADER_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$", re.MULTILINE)

DATA_MD_DIR = Path(__file__).resolve().parents[2] / "data" / "C2" / "md"
DOC_PATH = DATA_MD_DIR / "early-rl-chapter1.md"

CHARACTER_CONFIG = {
    "chunk_size": 400,
    "chunk_overlap": 100,
}

RECURSIVE_CONFIG = {
    "chunk_size": 400,
    "chunk_overlap": 100,
}

SEMANTIC_CONFIG = {
    "model_name": "BAAI/bge-small-zh-v1.5",
    "device": "cpu",
    "normalize_embeddings": True,
    "breakpoint_threshold_type": "gradient", # 也可以是 "percentile", "standard_deviation", "interquartile" 等
    "breakpoint_threshold_amount": None,
}

SEMANTIC_RECURSIVE_CONFIG = {
    "model_name": "BAAI/bge-small-zh-v1.5",
    "device": "cpu",
    "normalize_embeddings": True,
    "breakpoint_threshold_type": "gradient",
    "breakpoint_threshold_amount": None,
    "max_semantic_chunk_size": 1000,
    "recursive_chunk_size": 400,
    "recursive_chunk_overlap": 100,
}

MARKDOWN_HEADER_CONFIG = {
    "strip_headers": False,
}

MARKDOWN_RECURSIVE_CONFIG = {
    "strip_headers": False,
    "chunk_size": 400,
    "chunk_overlap": 100,
}

PARENT_CHILD_CONFIG = {
    "parent_chunk_size": 1500,
    "parent_chunk_overlap": 300,
    "child_chunk_size": 400,
    "child_chunk_overlap": 100,
}

RETRIEVAL_CONFIG = {
    "model_name": "BAAI/bge-small-zh-v1.5",
    "device": "cpu",
    "normalize_embeddings": True,
    "top_k": 3,
}

STRATEGY_DEFAULT_CONFIGS = {
    "CharacterTextSplitter": CHARACTER_CONFIG,
    "RecursiveCharacterTextSplitter": RECURSIVE_CONFIG,
    "SemanticChunker": SEMANTIC_CONFIG,
    "Semantic + Recursive": SEMANTIC_RECURSIVE_CONFIG,
    "MarkdownHeaderTextSplitter": MARKDOWN_HEADER_CONFIG,
    "MarkdownHeader + Recursive": MARKDOWN_RECURSIVE_CONFIG,
    "Parent-Child Chunking": PARENT_CHILD_CONFIG,
}

CURATED_QA_SETS = {
    "early-rl-chapter1": [
        {
            "question": "强化学习的核心目标是什么，它由哪两个基本部分组成？",
            "expected_keywords": ["智能体", "环境", "奖励"],
            "expected_phrases": [
                "强化学习由两部分组成：智能体和环境",
                "智能体的目的就是尽可能多地从环境中获取奖励",
            ],
        },
        {
            "question": "与监督学习相比，强化学习在训练信号和学习过程上为什么更困难？",
            "expected_keywords": ["正确动作", "奖励信号", "延迟"],
            "expected_phrases": [
                "学习器并没有告诉我们每一步正确的动作应该是什么",
                "奖励信号是延迟的",
            ],
        },
        {
            "question": "强化学习中的探索和利用分别指什么，为什么需要在两者之间权衡？",
            "expected_keywords": ["探索", "利用"],
            "expected_phrases": [
                "探索指尝试一些新的动作",
                "利用指采取已知的可以获得最多奖励的动作",
            ],
        },
        {
            "question": "完全可观测和部分可观测环境分别通常建模成什么问题？",
            "expected_keywords": ["马尔可夫决策过程", "部分可观测马尔可夫决策过程"],
            "expected_phrases": [
                "马尔可夫决策过程",
                "部分可观测马尔可夫决策过程",
            ],
        },
        {
            "question": "一个强化学习智能体通常由哪些核心组成成分构成？",
            "expected_keywords": ["策略", "价值函数", "模型"],
            "expected_phrases": ["策略（policy）", "价值函数（value function）", "模型（model）"],
        },
    ],
    "easy-rl-chapter1": [
        {
            "question": "强化学习的核心目标是什么，它由哪两个基本部分组成？",
            "expected_keywords": ["智能体", "环境", "奖励"],
            "expected_phrases": [
                "强化学习由两部分组成：智能体和环境",
                "智能体的目的就是尽可能多地从环境中获取奖励",
            ],
        },
        {
            "question": "与监督学习相比，强化学习在训练信号和学习过程上为什么更困难？",
            "expected_keywords": ["正确动作", "奖励信号", "延迟"],
            "expected_phrases": [
                "学习器并没有告诉我们每一步正确的动作应该是什么",
                "奖励信号是延迟的",
            ],
        },
        {
            "question": "强化学习中的探索和利用分别指什么，为什么需要在两者之间权衡？",
            "expected_keywords": ["探索", "利用"],
            "expected_phrases": [
                "探索指尝试一些新的动作",
                "利用指采取已知的可以获得最多奖励的动作",
            ],
        },
        {
            "question": "完全可观测和部分可观测环境分别通常建模成什么问题？",
            "expected_keywords": ["马尔可夫决策过程", "部分可观测马尔可夫决策过程"],
            "expected_phrases": [
                "马尔可夫决策过程",
                "部分可观测马尔可夫决策过程",
            ],
        },
        {
            "question": "一个强化学习智能体通常由哪些核心组成成分构成？",
            "expected_keywords": ["策略", "价值函数", "模型"],
            "expected_phrases": ["策略（policy）", "价值函数（value function）", "模型（model）"],
        },
    ],
}


@dataclass
class ChunkRecord:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    span: tuple[int, int] | None = None


@dataclass
class ChunkingResult:
    method_name: str
    chunks: list[ChunkRecord]
    config_used: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    note: str = ""


@dataclass
class HeaderSection:
    level: int
    title: str
    header_line: str
    span: tuple[int, int]
    path: dict[str, str]


def get_markdown_directory():
    return DATA_MD_DIR


def list_markdown_documents():
    if not DATA_MD_DIR.exists():
        return []
    return sorted(DATA_MD_DIR.glob("*.md"))


def save_markdown_file(filename: str, content: bytes):
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".md"):
        raise ValueError("只支持上传 .md 文件。")

    DATA_MD_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_MD_DIR / safe_name
    destination.write_bytes(content)
    return destination


def get_default_doc_path():
    return DOC_PATH


def resolve_doc_path(doc_path: str | Path | None = None):
    candidate = Path(doc_path) if doc_path else DOC_PATH
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"未找到文档: {candidate}")
    return candidate


def get_strategy_names():
    return list(STRATEGY_DEFAULT_CONFIGS.keys())


def get_default_strategy_config(method_name: str):
    return deepcopy(STRATEGY_DEFAULT_CONFIGS[method_name])


def get_all_default_strategy_configs():
    return {name: get_default_strategy_config(name) for name in get_strategy_names()}


def get_breakpoint_threshold_options():
    return list(BREAKPOINT_THRESHOLD_OPTIONS)


def get_default_retrieval_config():
    return deepcopy(RETRIEVAL_CONFIG)


def load_text(doc_path: str | Path | None = None):
    resolved = resolve_doc_path(doc_path)
    return resolved.read_text(encoding="utf-8"), resolved


def compact_text(text, width=PREVIEW_WIDTH):
    single_line = " ".join(text.split())
    if len(single_line) <= width:
        return single_line
    return single_line[: width - 3] + "..."


def compact_metadata(metadata):
    if not metadata:
        return "-"

    preferred_keys = [
        "Header 1",
        "Header 2",
        "Header 3",
        "Header 4",
        "split_stage",
        "semantic_index",
        "section_index",
        "parent_index",
        "child_index",
    ]
    items = []

    for key in preferred_keys:
        if key in metadata:
            items.append(f"{key}={metadata[key]}")

    if not items:
        for key in sorted(metadata)[:4]:
            items.append(f"{key}={metadata[key]}")

    return compact_text(", ".join(items), width=80)


def merge_config(default_config: dict[str, Any], override: dict[str, Any] | None = None):
    merged = deepcopy(default_config)
    if not override:
        return merged

    for key, value in override.items():
        merged[key] = value
    return merged


@lru_cache(maxsize=8)
def get_cached_embeddings(model_name: str, device: str, normalize_embeddings: bool):
    if HuggingFaceEmbeddings is None:
        raise RuntimeError(
            "缺少 HuggingFaceEmbeddings 依赖，请安装 langchain-community。"
        )

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )


def create_recursive_splitter(chunk_size, chunk_overlap):
    if RecursiveCharacterTextSplitter is None:
        raise RuntimeError(
            "缺少 RecursiveCharacterTextSplitter 依赖，请安装 langchain 或 langchain-text-splitters。"
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=COMMON_SEPARATORS,
    )


def create_markdown_header_splitter(strip_headers: bool):
    if MarkdownHeaderTextSplitter is None:
        raise RuntimeError(
            "缺少 MarkdownHeaderTextSplitter 依赖，请安装 langchain-text-splitters。"
        )

    return MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
        strip_headers=strip_headers,
    )


def normalize_chunks(chunks, base_metadata: dict[str, Any] | None = None):
    base_metadata = base_metadata or {}
    normalized = []
    for index, chunk in enumerate(chunks):
        if isinstance(chunk, ChunkRecord):
            metadata = dict(base_metadata)
            metadata.update(chunk.metadata)
            normalized.append(
                ChunkRecord(text=chunk.text, metadata=metadata, span=chunk.span)
            )
            continue

        if hasattr(chunk, "page_content"):
            metadata = dict(base_metadata)
            metadata.update(getattr(chunk, "metadata", {}) or {})
            normalized.append(ChunkRecord(text=chunk.page_content, metadata=metadata))
            continue

        normalized.append(
            ChunkRecord(
                text=str(chunk),
                metadata={**base_metadata, "chunk_index": index},
            )
        )
    return normalized


def locate_chunk_spans(source_text, chunks):
    spans = []
    search_start = 0

    for chunk in chunks:
        content = chunk.text.strip()
        if not content:
            spans.append(None)
            continue

        start = source_text.find(content, max(0, search_start - SPAN_BACKTRACK))
        if start == -1:
            start = source_text.find(content)

        if start == -1:
            spans.append(None)
            continue

        end = start + len(content)
        spans.append((start, end))
        search_start = start

    return spans


def attach_chunk_spans(chunks, source_text):
    spans = locate_chunk_spans(source_text, chunks)
    attached = []
    for chunk, span in zip(chunks, spans):
        attached.append(ChunkRecord(text=chunk.text, metadata=chunk.metadata, span=span))
    return attached


def split_with_character(text, config):
    if CharacterTextSplitter is None:
        raise RuntimeError(
            "缺少 CharacterTextSplitter 依赖，请安装 langchain 或 langchain-text-splitters。"
        )

    splitter = CharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    return normalize_chunks(splitter.split_text(text))


def split_with_recursive(text, config):
    splitter = create_recursive_splitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )
    return normalize_chunks(splitter.split_text(text))


def split_with_semantic(text, config):
    if SemanticChunker is None:
        raise RuntimeError(
            "缺少 SemanticChunker 相关依赖，请先安装 langchain-experimental。"
        )

    embeddings = get_cached_embeddings(
        config["model_name"],
        config["device"],
        bool(config["normalize_embeddings"]),
    )
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=config["breakpoint_threshold_type"],
        breakpoint_threshold_amount=config.get("breakpoint_threshold_amount"),
    )

    if hasattr(splitter, "split_text"):
        return normalize_chunks(splitter.split_text(text))

    return normalize_chunks(splitter.create_documents([text]))


def split_with_semantic_recursive(text, config):
    semantic_config = {
        "model_name": config["model_name"],
        "device": config["device"],
        "normalize_embeddings": config["normalize_embeddings"],
        "breakpoint_threshold_type": config["breakpoint_threshold_type"],
        "breakpoint_threshold_amount": config.get("breakpoint_threshold_amount"),
    }
    semantic_chunks = split_with_semantic(text, semantic_config)
    recursive_splitter = create_recursive_splitter(
        chunk_size=config["recursive_chunk_size"],
        chunk_overlap=config["recursive_chunk_overlap"],
    )

    refined_chunks = []
    for semantic_index, chunk in enumerate(semantic_chunks):
        base_metadata = dict(chunk.metadata)
        base_metadata["semantic_index"] = semantic_index

        if len(chunk.text) <= config["max_semantic_chunk_size"]:
            base_metadata["split_stage"] = "semantic"
            refined_chunks.append(ChunkRecord(text=chunk.text, metadata=base_metadata))
            continue

        child_texts = recursive_splitter.split_text(chunk.text)
        for child_index, child_text in enumerate(child_texts):
            refined_chunks.append(
                ChunkRecord(
                    text=child_text,
                    metadata={
                        **base_metadata,
                        "split_stage": "semantic_recursive",
                        "child_index": child_index,
                    },
                )
            )

    return refined_chunks


def split_with_markdown_headers(text, config):
    splitter = create_markdown_header_splitter(strip_headers=bool(config["strip_headers"]))
    return normalize_chunks(splitter.split_text(text))


def split_with_markdown_headers_recursive(text, config):
    header_config = {"strip_headers": config["strip_headers"]}
    header_chunks = split_with_markdown_headers(text, header_config)
    recursive_splitter = create_recursive_splitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )

    refined_chunks = []
    for section_index, chunk in enumerate(header_chunks):
        child_texts = recursive_splitter.split_text(chunk.text)
        for child_index, child_text in enumerate(child_texts):
            refined_chunks.append(
                ChunkRecord(
                    text=child_text,
                    metadata={
                        **chunk.metadata,
                        "section_index": section_index,
                        "child_index": child_index,
                        "split_stage": "markdown_recursive",
                    },
                )
            )

    return refined_chunks


def split_with_parent_child(text, config):
    parent_splitter = create_recursive_splitter(
        chunk_size=config["parent_chunk_size"],
        chunk_overlap=config["parent_chunk_overlap"],
    )
    child_splitter = create_recursive_splitter(
        chunk_size=config["child_chunk_size"],
        chunk_overlap=config["child_chunk_overlap"],
    )

    parent_chunks = parent_splitter.split_text(text)
    child_records = []

    for parent_index, parent_text in enumerate(parent_chunks):
        child_texts = child_splitter.split_text(parent_text)
        for child_index, child_text in enumerate(child_texts):
            child_records.append(
                ChunkRecord(
                    text=child_text,
                    metadata={
                        "parent_index": parent_index,
                        "child_index": child_index,
                        "parent_length": len(parent_text),
                        "parent_preview": compact_text(parent_text, width=60),
                    },
                )
            )

    return child_records, {"parent_chunks": len(parent_chunks)}


STRATEGY_RUNNERS = {
    "CharacterTextSplitter": split_with_character,
    "RecursiveCharacterTextSplitter": split_with_recursive,
    "SemanticChunker": split_with_semantic,
    "Semantic + Recursive": split_with_semantic_recursive,
    "MarkdownHeaderTextSplitter": split_with_markdown_headers,
    "MarkdownHeader + Recursive": split_with_markdown_headers_recursive,
    "Parent-Child Chunking": split_with_parent_child,
}


def run_strategy(
    method_name: str,
    doc_path: str | Path | None = None,
    config_override: dict[str, Any] | None = None,
):
    source_text, resolved_doc_path = load_text(doc_path)
    config_used = merge_config(get_default_strategy_config(method_name), config_override)

    try:
        raw_result = STRATEGY_RUNNERS[method_name](source_text, config_used)
        extra_note = ""
        chunks = raw_result
        if isinstance(raw_result, tuple):
            chunks, info = raw_result
            extra_note = ", ".join(f"{key}={value}" for key, value in info.items())

        chunks = attach_chunk_spans(chunks, source_text)
        return ChunkingResult(
            method_name=method_name,
            chunks=chunks,
            config_used=config_used,
            note=extra_note,
        )
    except Exception as exc:  # noqa: BLE001
        return ChunkingResult(
            method_name=method_name,
            chunks=[],
            config_used=config_used,
            error=f"{exc}",
            note=f"doc_path={resolved_doc_path.name}",
        )


def run_all_strategies(
    doc_path: str | Path | None = None,
    strategy_overrides: dict[str, dict[str, Any]] | None = None,
):
    results = []
    for method_name in get_strategy_names():
        override = (strategy_overrides or {}).get(method_name)
        results.append(run_strategy(method_name, doc_path=doc_path, config_override=override))
    return results


def extract_header_sections(source_text):
    matches = list(HEADER_PATTERN.finditer(source_text))
    if not matches:
        return []

    sections = []
    active_path: dict[int, str] = {}

    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        section_start = match.start()
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(source_text)

        active_path = {depth: text for depth, text in active_path.items() if depth < level}
        active_path[level] = title
        path = {f"Header {depth}": text for depth, text in sorted(active_path.items())}

        sections.append(
            HeaderSection(
                level=level,
                title=title,
                header_line=match.group(0).strip(),
                span=(section_start, section_end),
                path=path,
            )
        )

    return sections


def interval_intersection(a: tuple[int, int], b: tuple[int, int]):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_iou(a: tuple[int, int], b: tuple[int, int]):
    intersection = interval_intersection(a, b)
    if intersection == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    return intersection / union if union else 0.0


def chunk_preserves_header(chunk: ChunkRecord, section: HeaderSection):
    if section.header_line in chunk.text:
        return True

    leading_text = chunk.text[: min(len(chunk.text), 120)]
    if section.title in leading_text:
        return True

    if not section.path:
        return False

    for key, value in section.path.items():
        if chunk.metadata.get(key) != value:
            return False
    return True


def dominant_section_for_chunk(chunk: ChunkRecord, sections: list[HeaderSection]):
    if chunk.span is None:
        return None, 0

    best_section = None
    best_overlap = 0
    for section in sections:
        overlap = interval_intersection(chunk.span, section.span)
        if overlap > best_overlap:
            best_overlap = overlap
            best_section = section

    return best_section, best_overlap


def compute_header_preservation_score(result: ChunkingResult, source_text: str):
    sections = extract_header_sections(source_text)
    if not sections:
        return None

    weighted_hits = 0
    weighted_total = 0

    for chunk in result.chunks:
        section, overlap = dominant_section_for_chunk(chunk, sections)
        if section is None or overlap == 0:
            continue

        weighted_total += overlap
        if chunk_preserves_header(chunk, section):
            weighted_hits += overlap

    if weighted_total == 0:
        return 0.0
    return weighted_hits / weighted_total


def compute_length_metrics(result: ChunkingResult, source_text: str):
    if result.error:
        return {
            "status": "失败",
            "error": result.error,
        }

    lengths = [len(chunk.text) for chunk in result.chunks]
    if not lengths:
        return {
            "status": "成功",
            "chunk_count": 0,
            "avg_length": 0.0,
            "median_length": 0.0,
            "min_length": 0,
            "max_length": 0,
            "total_chars": 0,
            "coverage_ratio": 0.0,
            "length_variance": 0.0,
            "length_std": 0.0,
            "length_variance_score": 1.0,
        }

    avg_length = mean(lengths)
    variance = pvariance(lengths) if len(lengths) > 1 else 0.0
    std = math.sqrt(variance)
    coefficient_of_variation = std / avg_length if avg_length else 0.0
    variance_score = 1 / (1 + coefficient_of_variation)

    total_chars = sum(lengths)
    return {
        "status": "成功",
        "chunk_count": len(result.chunks),
        "avg_length": avg_length,
        "median_length": median(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_chars": total_chars,
        "coverage_ratio": total_chars / len(source_text) if source_text else 0.0,
        "length_variance": variance,
        "length_std": std,
        "length_variance_score": variance_score,
    }


def compute_pairwise_overlap_score(result_a: ChunkingResult, result_b: ChunkingResult):
    spans_a = [chunk.span for chunk in result_a.chunks if chunk.span is not None]
    spans_b = [chunk.span for chunk in result_b.chunks if chunk.span is not None]

    if result_a.error or result_b.error or not spans_a or not spans_b:
        return None

    def directed_score(source_spans, target_spans):
        weighted_score = 0.0
        total_weight = 0
        for span in source_spans:
            chunk_length = span[1] - span[0]
            if chunk_length <= 0:
                continue
            best_iou = max(interval_iou(span, other) for other in target_spans)
            weighted_score += chunk_length * best_iou
            total_weight += chunk_length
        return weighted_score / total_weight if total_weight else 0.0

    return (directed_score(spans_a, spans_b) + directed_score(spans_b, spans_a)) / 2


def compute_pairwise_overlap_matrix(results: list[ChunkingResult]):
    matrix = {result.method_name: {} for result in results}

    for result in results:
        matrix[result.method_name][result.method_name] = 1.0 if not result.error else None

    for index, result_a in enumerate(results):
        for result_b in results[index + 1 :]:
            score = compute_pairwise_overlap_score(result_a, result_b)
            matrix[result_a.method_name][result_b.method_name] = score
            matrix[result_b.method_name][result_a.method_name] = score

    return matrix


def average_pairwise_overlap(method_name: str, overlap_matrix: dict[str, dict[str, float | None]]):
    values = [
        value
        for other_method, value in overlap_matrix.get(method_name, {}).items()
        if other_method != method_name and value is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def build_qa_examples(doc_path: Path, source_text: str):
    doc_stem = doc_path.stem
    if doc_stem in CURATED_QA_SETS:
        return deepcopy(CURATED_QA_SETS[doc_stem]), "curated"

    sections = extract_header_sections(source_text)
    heuristic_examples = []
    for section in sections[:5]:
        heuristic_examples.append(
            {
                "question": f"请检索与“{section.title}”相关的内容。",
                "expected_keywords": [section.title],
                "expected_phrases": [section.title],
            }
        )

    return heuristic_examples, "heuristic"


def normalize_matrix(vectors: np.ndarray):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def qa_match(chunk_text: str, qa_example: dict[str, Any]):
    phrases = qa_example.get("expected_phrases", [])
    keywords = qa_example.get("expected_keywords", [])

    if any(phrase and phrase in chunk_text for phrase in phrases):
        return True

    if keywords and all(keyword in chunk_text for keyword in keywords):
        return True

    return False


def retrieve_top_chunks_for_questions(
    result: ChunkingResult,
    questions: list[str],
    retrieval_config: dict[str, Any] | None = None,
):
    config = merge_config(RETRIEVAL_CONFIG, retrieval_config)
    cleaned_questions = [question.strip() for question in questions if question and question.strip()]

    if result.error:
        return {
            "questions": cleaned_questions,
            "details": [],
            "top_k": config["top_k"],
            "error": result.error,
        }

    if not cleaned_questions:
        return {
            "questions": [],
            "details": [],
            "top_k": config["top_k"],
            "error": "请至少提供一个问题。",
        }

    if not result.chunks:
        return {
            "questions": cleaned_questions,
            "details": [],
            "top_k": config["top_k"],
            "error": "当前策略没有可检索的分块结果。",
        }

    try:
        embeddings = get_cached_embeddings(
            config["model_name"],
            config["device"],
            bool(config["normalize_embeddings"]),
        )
        document_embeddings = np.array(
            embeddings.embed_documents([chunk.text for chunk in result.chunks]),
            dtype=float,
        )
        query_embeddings = np.array(
            [embeddings.embed_query(question) for question in cleaned_questions],
            dtype=float,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "questions": cleaned_questions,
            "details": [],
            "top_k": config["top_k"],
            "error": f"{exc}",
        }

    document_embeddings = normalize_matrix(document_embeddings)
    query_embeddings = normalize_matrix(query_embeddings)
    top_k = min(config["top_k"], len(result.chunks))
    details = []

    for question_index, question in enumerate(cleaned_questions):
        similarities = document_embeddings @ query_embeddings[question_index]
        ranked_indices = np.argsort(-similarities)[:top_k]

        details.append(
            {
                "question": question,
                "top_chunks": [
                    {
                        "chunk_index": int(chunk_index),
                        "score": float(similarities[int(chunk_index)]),
                        "span": result.chunks[int(chunk_index)].span,
                        "metadata_preview": compact_metadata(result.chunks[int(chunk_index)].metadata),
                        "preview": compact_text(result.chunks[int(chunk_index)].text, width=100),
                        "text": result.chunks[int(chunk_index)].text,
                    }
                    for chunk_index in ranked_indices
                ],
            }
        )

    return {
        "questions": cleaned_questions,
        "details": details,
        "top_k": top_k,
        "error": None,
    }


def compute_retrieval_evaluation(
    result: ChunkingResult,
    qa_examples: list[dict[str, Any]],
    retrieval_config: dict[str, Any] | None = None,
):
    config = merge_config(RETRIEVAL_CONFIG, retrieval_config)

    if result.error:
        return {
            "score": None,
            "hit_rate": None,
            "mrr": None,
            "details": [],
            "qa_count": len(qa_examples),
            "top_k": config["top_k"],
            "error": result.error,
        }

    if not qa_examples:
        return {
            "score": None,
            "hit_rate": None,
            "mrr": None,
            "details": [],
            "qa_count": 0,
            "top_k": config["top_k"],
            "error": "没有可用的 QA 集。",
        }

    if not result.chunks:
        return {
            "score": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "details": [],
            "qa_count": len(qa_examples),
            "top_k": config["top_k"],
            "error": None,
        }

    retrieval = retrieve_top_chunks_for_questions(
        result,
        [example["question"] for example in qa_examples],
        retrieval_config=config,
    )
    if retrieval["error"]:
        return {
            "score": None,
            "hit_rate": None,
            "mrr": None,
            "details": [],
            "qa_count": len(qa_examples),
            "top_k": config["top_k"],
            "error": retrieval["error"],
        }

    top_k = retrieval["top_k"]
    hits = 0
    reciprocal_ranks = []
    details = []

    for qa_example, retrieval_detail in zip(qa_examples, retrieval["details"]):
        hit_rank = None
        matched_chunk_index = None
        for rank, top_chunk in enumerate(retrieval_detail["top_chunks"], start=1):
            chunk_index = top_chunk["chunk_index"]
            if qa_match(result.chunks[int(chunk_index)].text, qa_example):
                hit_rank = rank
                matched_chunk_index = int(chunk_index)
                break

        reciprocal_rank = 1 / hit_rank if hit_rank else 0.0
        hits += 1 if hit_rank else 0
        reciprocal_ranks.append(reciprocal_rank)

        details.append(
            {
                "question": qa_example["question"],
                "hit_rank": hit_rank,
                "matched_chunk_index": matched_chunk_index,
                "top_chunks": [
                    {
                        "chunk_index": top_chunk["chunk_index"],
                        "score": top_chunk["score"],
                        "preview": top_chunk["preview"],
                    }
                    for top_chunk in retrieval_detail["top_chunks"]
                ],
            }
        )

    hit_rate = hits / len(qa_examples)
    mrr = sum(reciprocal_ranks) / len(qa_examples)
    return {
        "score": (hit_rate + mrr) / 2,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "details": details,
        "qa_count": len(qa_examples),
        "top_k": top_k,
        "error": None,
    }


def build_chunk_rows(result: ChunkingResult):
    rows = []
    for index, chunk in enumerate(result.chunks):
        start = chunk.span[0] if chunk.span else None
        end = chunk.span[1] if chunk.span else None
        rows.append(
            {
                "chunk_index": index,
                "length": len(chunk.text),
                "start": start,
                "end": end,
                "metadata_preview": compact_metadata(chunk.metadata),
                "preview": compact_text(chunk.text, width=120),
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
        )
    return rows


def build_comparison_bundle(
    results: list[ChunkingResult],
    doc_path: str | Path | None = None,
    retrieval_methods: set[str] | None = None,
):
    source_text, resolved_doc_path = load_text(doc_path)
    qa_examples, qa_source = build_qa_examples(resolved_doc_path, source_text)
    overlap_matrix = compute_pairwise_overlap_matrix(results)

    metrics_by_method = {}
    retrieval_by_method = {}
    result_map = {result.method_name: result for result in results}

    for result in results:
        metrics = compute_length_metrics(result, source_text)
        metrics["header_preservation_score"] = (
            None if result.error else compute_header_preservation_score(result, source_text)
        )
        metrics["avg_pairwise_overlap"] = average_pairwise_overlap(
            result.method_name,
            overlap_matrix,
        )

        should_run_retrieval = retrieval_methods is None or result.method_name in retrieval_methods
        if should_run_retrieval:
            retrieval = compute_retrieval_evaluation(result, qa_examples)
        else:
            retrieval = {
                "score": None,
                "hit_rate": None,
                "mrr": None,
                "details": [],
                "qa_count": len(qa_examples),
                "top_k": RETRIEVAL_CONFIG["top_k"],
                "error": None,
            }

        metrics["retrieval_score"] = retrieval["score"]
        metrics["retrieval_hit_rate"] = retrieval["hit_rate"]
        metrics["retrieval_mrr"] = retrieval["mrr"]

        metrics_by_method[result.method_name] = metrics
        retrieval_by_method[result.method_name] = retrieval

    return {
        "doc_path": resolved_doc_path,
        "source_text": source_text,
        "results": result_map,
        "metrics": metrics_by_method,
        "retrieval": retrieval_by_method,
        "pairwise_overlap_matrix": overlap_matrix,
        "qa_examples": qa_examples,
        "qa_source": qa_source,
    }


def run_full_comparison(
    doc_path: str | Path | None = None,
    strategy_overrides: dict[str, dict[str, Any]] | None = None,
    retrieval_methods: set[str] | None = None,
):
    results = run_all_strategies(doc_path=doc_path, strategy_overrides=strategy_overrides)
    return build_comparison_bundle(
        results,
        doc_path=doc_path,
        retrieval_methods=retrieval_methods,
    )


def run_selected_strategy_bundle(
    method_name: str,
    doc_path: str | Path | None = None,
    config_override: dict[str, Any] | None = None,
):
    return run_full_comparison(
        doc_path=doc_path,
        strategy_overrides={method_name: config_override or {}},
        retrieval_methods={method_name},
    )


def summarize_method(method_name: str, result: ChunkingResult, metrics: dict[str, Any]):
    if result.error:
        return {
            "method": method_name,
            "status": "失败",
            "chunk_count": "-",
            "avg_length": "-",
            "variance_score": "-",
            "header_score": "-",
            "retrieval_score": "-",
            "avg_overlap": "-",
            "note": compact_text(result.error, width=60),
        }

    return {
        "method": method_name,
        "status": metrics["status"],
        "chunk_count": str(metrics["chunk_count"]),
        "avg_length": f"{metrics['avg_length']:.1f}",
        "variance_score": f"{metrics['length_variance_score']:.3f}",
        "header_score": format_optional_score(metrics["header_preservation_score"]),
        "retrieval_score": format_optional_score(metrics["retrieval_score"]),
        "avg_overlap": format_optional_score(metrics["avg_pairwise_overlap"]),
        "note": compact_text(result.note or str(result.config_used), width=60),
    }


def format_optional_score(value):
    if value is None:
        return "-"
    return f"{value:.3f}"


def print_summary_table(bundle):
    rows = [
        summarize_method(
            method_name,
            bundle["results"][method_name],
            bundle["metrics"][method_name],
        )
        for method_name in get_strategy_names()
    ]

    headers = ["方法", "状态", "块数", "平均长度", "方差得分", "Header得分", "检索得分", "平均重叠", "备注"]
    widths = [
        max(len(headers[0]), max(len(row["method"]) for row in rows)),
        max(len(headers[1]), max(len(row["status"]) for row in rows)),
        max(len(headers[2]), max(len(row["chunk_count"]) for row in rows)),
        max(len(headers[3]), max(len(row["avg_length"]) for row in rows)),
        max(len(headers[4]), max(len(row["variance_score"]) for row in rows)),
        max(len(headers[5]), max(len(row["header_score"]) for row in rows)),
        max(len(headers[6]), max(len(row["retrieval_score"]) for row in rows)),
        max(len(headers[7]), max(len(row["avg_overlap"]) for row in rows)),
        max(len(headers[8]), max(len(row["note"]) for row in rows)),
    ]

    def format_row(values):
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    print("\n=== 分块结果汇总 ===")
    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))

    for row in rows:
        print(
            format_row(
                [
                    row["method"],
                    row["status"],
                    row["chunk_count"],
                    row["avg_length"],
                    row["variance_score"],
                    row["header_score"],
                    row["retrieval_score"],
                    row["avg_overlap"],
                    row["note"],
                ]
            )
        )


def print_pairwise_overlap_table(bundle):
    matrix = bundle["pairwise_overlap_matrix"]
    method_names = get_strategy_names()
    widths = [max(18, max(len(name) for name in method_names))]
    widths.extend([12] * len(method_names))

    def format_row(values):
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    print("\n=== Pairwise Overlap 矩阵 ===")
    header = ["方法"] + method_names
    print(format_row(header))
    print("-+-".join("-" * width for width in widths))

    for method_name in method_names:
        row = [method_name]
        for other_method in method_names:
            row.append(format_optional_score(matrix[method_name].get(other_method)))
        print(format_row(row))


def print_chunk_previews(bundle):
    print(f"\n=== 前 {PREVIEW_COUNT} 个块横向对比 ===")

    for index in range(PREVIEW_COUNT):
        print(f"\n--- 块索引 {index + 1} ---")
        for method_name in get_strategy_names():
            result = bundle["results"][method_name]
            if result.error:
                print(f"[{method_name}] 跳过: {result.error}")
                continue

            if index >= len(result.chunks):
                print(f"[{method_name}] 无对应块")
                continue

            chunk = result.chunks[index]
            span_text = "位置未知" if chunk.span is None else f"[{chunk.span[0]}, {chunk.span[1]})"
            print(
                f"[{method_name}] 长度={len(chunk.text):>4} 位置={span_text}\n"
                f"元数据: {compact_metadata(chunk.metadata)}\n"
                f"{compact_text(chunk.text)}"
            )


def print_boundary_observations(bundle):
    print("\n=== 边界观察 ===")

    for method_name in get_strategy_names():
        result = bundle["results"][method_name]
        if result.error:
            print(f"{method_name}: 未生成结果。")
            continue

        if not result.chunks:
            print(f"{method_name}: 生成了 0 个块。")
            continue

        first_chunk = compact_text(result.chunks[0].text, width=60)
        last_chunk = compact_text(result.chunks[-1].text, width=60)
        print(f"{method_name}: 首块预览 -> {first_chunk} | 末块预览 -> {last_chunk}")


def print_retrieval_summary(bundle):
    print("\n=== 检索评估摘要 ===")
    print(f"QA 集来源: {bundle['qa_source']}, 题目数: {len(bundle['qa_examples'])}")

    for method_name in get_strategy_names():
        retrieval = bundle["retrieval"][method_name]
        if retrieval["error"]:
            print(f"{method_name}: 检索评估失败 -> {retrieval['error']}")
            continue
        print(
            f"{method_name}: score={format_optional_score(retrieval['score'])}, "
            f"hit@{retrieval['top_k']}={format_optional_score(retrieval['hit_rate'])}, "
            f"MRR={format_optional_score(retrieval['mrr'])}"
        )


def main():
    try:
        bundle = run_full_comparison()
    except FileNotFoundError as exc:
        print(f"文档读取失败: {exc}")
        return

    print("=== 文档信息 ===")
    print(f"文档路径: {bundle['doc_path']}")
    print(f"原始文本长度: {len(bundle['source_text'])} 字符")

    print_summary_table(bundle)
    print_pairwise_overlap_table(bundle)
    print_chunk_previews(bundle)
    print_boundary_observations(bundle)
    print_retrieval_summary(bundle)


if __name__ == "__main__":
    main()
