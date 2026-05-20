from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

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

DOC_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "C2" / "md" / "early-rl-chapter1.md"
)

CHARACTER_CONFIG = {
    "chunk_size": 260,
    "chunk_overlap": 40,
}

RECURSIVE_CONFIG = {
    "chunk_size": 260,
    "chunk_overlap": 40,
    "separators": ["\n\n", "\n", "。", "，", " ", ""],
}

SEMANTIC_CONFIG = {
    "model_name": "BAAI/bge-small-zh-v1.5",
    "model_kwargs": {"device": "cpu"},
    "encode_kwargs": {"normalize_embeddings": True},
    "breakpoint_threshold_type": "gradient",
}

SEMANTIC_RECURSIVE_CONFIG = {
    "max_semantic_chunk_size": 420,
    "recursive_chunk_size": 240,
    "recursive_chunk_overlap": 40,
    "recursive_separators": ["\n\n", "\n", "。", "，", " ", ""],
}

MARKDOWN_HEADER_CONFIG = {
    "headers_to_split_on": [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ],
    "strip_headers": False,
}

MARKDOWN_RECURSIVE_CONFIG = {
    "chunk_size": 300,
    "chunk_overlap": 50,
    "separators": ["\n\n", "\n", "。", "，", " ", ""],
}

PARENT_CHILD_CONFIG = {
    "parent_chunk_size": 1200,
    "parent_chunk_overlap": 120,
    "child_chunk_size": 260,
    "child_chunk_overlap": 40,
    "separators": ["\n\n", "\n", "。", "，", " ", ""],
}


@dataclass
class ChunkRecord:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingResult:
    method_name: str
    chunks: list[ChunkRecord]
    error: str | None = None
    note: str = ""


def load_text():
    return DOC_PATH.read_text(encoding="utf-8")


def create_recursive_splitter(chunk_size, chunk_overlap, separators):
    if RecursiveCharacterTextSplitter is None:
        raise RuntimeError(
            "缺少 RecursiveCharacterTextSplitter 依赖，请安装 langchain 或 langchain-text-splitters。"
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )


def normalize_chunks(chunks, base_metadata: dict[str, Any] | None = None):
    base_metadata = base_metadata or {}
    normalized = []
    for index, chunk in enumerate(chunks):
        if isinstance(chunk, ChunkRecord):
            metadata = dict(base_metadata)
            metadata.update(chunk.metadata)
            normalized.append(ChunkRecord(text=chunk.text, metadata=metadata))
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


def split_with_character():
    if CharacterTextSplitter is None:
        raise RuntimeError(
            "缺少 CharacterTextSplitter 依赖，请安装 langchain 或 langchain-text-splitters。"
        )

    splitter = CharacterTextSplitter(
        chunk_size=CHARACTER_CONFIG["chunk_size"],
        chunk_overlap=CHARACTER_CONFIG["chunk_overlap"],
    )
    chunks = splitter.split_text(load_text())
    return normalize_chunks(chunks)


def split_with_recursive():
    splitter = create_recursive_splitter(
        chunk_size=RECURSIVE_CONFIG["chunk_size"],
        chunk_overlap=RECURSIVE_CONFIG["chunk_overlap"],
        separators=RECURSIVE_CONFIG["separators"],
    )
    chunks = splitter.split_text(load_text())
    return normalize_chunks(chunks)


def split_with_semantic():
    if HuggingFaceEmbeddings is None or SemanticChunker is None:
        raise RuntimeError(
            "缺少 SemanticChunker 相关依赖，请先安装 langchain-experimental 和相关 embedding 组件。"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=SEMANTIC_CONFIG["model_name"],
        model_kwargs=SEMANTIC_CONFIG["model_kwargs"],
        encode_kwargs=SEMANTIC_CONFIG["encode_kwargs"],
    )
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=SEMANTIC_CONFIG["breakpoint_threshold_type"],
    )

    text = load_text()
    if hasattr(splitter, "split_text"):
        return normalize_chunks(splitter.split_text(text))

    return normalize_chunks(splitter.create_documents([text]))


def split_with_semantic_recursive():
    semantic_chunks = split_with_semantic()
    recursive_splitter = create_recursive_splitter(
        chunk_size=SEMANTIC_RECURSIVE_CONFIG["recursive_chunk_size"],
        chunk_overlap=SEMANTIC_RECURSIVE_CONFIG["recursive_chunk_overlap"],
        separators=SEMANTIC_RECURSIVE_CONFIG["recursive_separators"],
    )

    refined_chunks = []
    for semantic_index, chunk in enumerate(semantic_chunks):
        base_metadata = dict(chunk.metadata)
        base_metadata["semantic_index"] = semantic_index

        if len(chunk.text) <= SEMANTIC_RECURSIVE_CONFIG["max_semantic_chunk_size"]:
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


def create_markdown_header_splitter():
    if MarkdownHeaderTextSplitter is None:
        raise RuntimeError(
            "缺少 MarkdownHeaderTextSplitter 依赖，请安装 langchain-text-splitters。"
        )

    return MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADER_CONFIG["headers_to_split_on"],
        strip_headers=MARKDOWN_HEADER_CONFIG["strip_headers"],
    )


def split_with_markdown_headers():
    splitter = create_markdown_header_splitter()
    return normalize_chunks(splitter.split_text(load_text()))


def split_with_markdown_headers_recursive():
    header_chunks = split_with_markdown_headers()
    recursive_splitter = create_recursive_splitter(
        chunk_size=MARKDOWN_RECURSIVE_CONFIG["chunk_size"],
        chunk_overlap=MARKDOWN_RECURSIVE_CONFIG["chunk_overlap"],
        separators=MARKDOWN_RECURSIVE_CONFIG["separators"],
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


def split_with_parent_child():
    parent_splitter = create_recursive_splitter(
        chunk_size=PARENT_CHILD_CONFIG["parent_chunk_size"],
        chunk_overlap=PARENT_CHILD_CONFIG["parent_chunk_overlap"],
        separators=PARENT_CHILD_CONFIG["separators"],
    )
    child_splitter = create_recursive_splitter(
        chunk_size=PARENT_CHILD_CONFIG["child_chunk_size"],
        chunk_overlap=PARENT_CHILD_CONFIG["child_chunk_overlap"],
        separators=PARENT_CHILD_CONFIG["separators"],
    )

    parent_chunks = parent_splitter.split_text(load_text())
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

    return child_records, len(parent_chunks)


def run_chunker(method_name, splitter_func, note=""):
    try:
        result = splitter_func()
        if isinstance(result, tuple):
            chunks, extra_info = result
            return ChunkingResult(
                method_name=method_name,
                chunks=chunks,
                note=f"{note} parent_chunks={extra_info}".strip(),
            )

        return ChunkingResult(method_name=method_name, chunks=result, note=note)
    except Exception as exc:  # noqa: BLE001
        return ChunkingResult(method_name=method_name, chunks=[], error=str(exc), note=note)


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


def summarize_result(result, source_text):
    if result.error:
        return {
            "method": result.method_name,
            "status": "失败",
            "chunk_count": "-",
            "avg_length": "-",
            "median_length": "-",
            "min_max": "-",
            "coverage": "-",
            "note": compact_text(result.error, width=60),
        }

    lengths = [len(chunk.text) for chunk in result.chunks]
    if not lengths:
        return {
            "method": result.method_name,
            "status": "成功",
            "chunk_count": "0",
            "avg_length": "0.0",
            "median_length": "0.0",
            "min_max": "0/0",
            "coverage": "0 (0.00x)",
            "note": compact_text(result.note or "-", width=60),
        }

    total_chars = sum(lengths)
    ratio = total_chars / len(source_text) if source_text else 0
    note = result.note or "-"
    return {
        "method": result.method_name,
        "status": "成功",
        "chunk_count": str(len(result.chunks)),
        "avg_length": f"{mean(lengths):.1f}",
        "median_length": f"{median(lengths):.1f}",
        "min_max": f"{min(lengths)}/{max(lengths)}",
        "coverage": f"{total_chars} ({ratio:.2f}x)",
        "note": compact_text(note, width=60),
    }


def print_summary_table(results, source_text):
    rows = [summarize_result(result, source_text) for result in results]
    headers = ["方法", "状态", "块数", "平均长度", "中位长度", "最短/最长", "总字符数", "备注"]
    widths = [
        max(len(headers[0]), max(len(row["method"]) for row in rows)),
        max(len(headers[1]), max(len(row["status"]) for row in rows)),
        max(len(headers[2]), max(len(row["chunk_count"]) for row in rows)),
        max(len(headers[3]), max(len(row["avg_length"]) for row in rows)),
        max(len(headers[4]), max(len(row["median_length"]) for row in rows)),
        max(len(headers[5]), max(len(row["min_max"]) for row in rows)),
        max(len(headers[6]), max(len(row["coverage"]) for row in rows)),
        max(len(headers[7]), max(len(row["note"]) for row in rows)),
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
                    row["median_length"],
                    row["min_max"],
                    row["coverage"],
                    row["note"],
                ]
            )
        )


def print_chunk_previews(results, source_text):
    span_map = {
        result.method_name: locate_chunk_spans(source_text, result.chunks)
        for result in results
        if not result.error
    }

    print(f"\n=== 前 {PREVIEW_COUNT} 个块横向对比 ===")
    for index in range(PREVIEW_COUNT):
        print(f"\n--- 块索引 {index + 1} ---")
        for result in results:
            if result.error:
                print(f"[{result.method_name}] 跳过: {result.error}")
                continue

            if index >= len(result.chunks):
                print(f"[{result.method_name}] 无对应块")
                continue

            chunk = result.chunks[index]
            span = span_map[result.method_name][index]
            span_text = "位置未知" if span is None else f"[{span[0]}, {span[1]})"
            preview = compact_text(chunk.text)
            metadata_preview = compact_metadata(chunk.metadata)
            print(
                f"[{result.method_name}] 长度={len(chunk.text):>4} 位置={span_text}\n"
                f"元数据: {metadata_preview}\n"
                f"{preview}"
            )


def print_boundary_observations(results):
    print("\n=== 边界观察 ===")
    for result in results:
        if result.error:
            print(f"{result.method_name}: 未生成结果。")
            continue

        if not result.chunks:
            print(f"{result.method_name}: 生成了 0 个块。")
            continue

        first_chunk = compact_text(result.chunks[0].text, width=60)
        last_chunk = compact_text(result.chunks[-1].text, width=60)
        print(
            f"{result.method_name}: 首块预览 -> {first_chunk} | 末块预览 -> {last_chunk}"
        )


def main():
    source_text = load_text()

    print("=== 文档信息 ===")
    print(f"文档路径: {DOC_PATH}")
    print(f"原始文本长度: {len(source_text)} 字符")

    results = [
        run_chunker("CharacterTextSplitter", split_with_character, note=str(CHARACTER_CONFIG)),
        run_chunker(
            "RecursiveCharacterTextSplitter",
            split_with_recursive,
            note=str({k: v for k, v in RECURSIVE_CONFIG.items() if k != "separators"}),
        ),
        run_chunker(
            "SemanticChunker",
            split_with_semantic,
            note=SEMANTIC_CONFIG["breakpoint_threshold_type"],
        ),
        run_chunker(
            "Semantic + Recursive",
            split_with_semantic_recursive,
            note=str(
                {
                    "max_semantic_chunk_size": SEMANTIC_RECURSIVE_CONFIG["max_semantic_chunk_size"],
                    "recursive_chunk_size": SEMANTIC_RECURSIVE_CONFIG["recursive_chunk_size"],
                    "recursive_chunk_overlap": SEMANTIC_RECURSIVE_CONFIG["recursive_chunk_overlap"],
                }
            ),
        ),
        run_chunker("MarkdownHeaderTextSplitter", split_with_markdown_headers, note="header-aware"),
        run_chunker(
            "MarkdownHeader + Recursive",
            split_with_markdown_headers_recursive,
            note=str(
                {
                    "chunk_size": MARKDOWN_RECURSIVE_CONFIG["chunk_size"],
                    "chunk_overlap": MARKDOWN_RECURSIVE_CONFIG["chunk_overlap"],
                }
            ),
        ),
        run_chunker(
            "Parent-Child Chunking",
            split_with_parent_child,
            note=str(
                {
                    "parent_chunk_size": PARENT_CHILD_CONFIG["parent_chunk_size"],
                    "child_chunk_size": PARENT_CHILD_CONFIG["child_chunk_size"],
                }
            ),
        ),
    ]

    print_summary_table(results, source_text)
    print_chunk_previews(results, source_text)
    print_boundary_observations(results)


if __name__ == "__main__":
    main()
