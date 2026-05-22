from __future__ import annotations

import argparse
import json

from config import (
    CHUNK_MANIFEST_PATH,
    DEFAULT_CHUNK_STRATEGY,
    GOLD_WORKSPACE_PATH,
    PARENT_CHILD_CHUNK_MANIFEST_PATH,
    PARENT_CHILD_CHUNK_STRATEGY,
    PARENT_CHILD_GOLD_WORKSPACE_PATH,
    QA_PATH,
)


def resolve_paths(chunk_strategy: str) -> tuple[str, str, str]:
    if chunk_strategy == DEFAULT_CHUNK_STRATEGY:
        return str(CHUNK_MANIFEST_PATH), str(GOLD_WORKSPACE_PATH), "gold_chunk_ids"
    if chunk_strategy == PARENT_CHILD_CHUNK_STRATEGY:
        return (
            str(PARENT_CHILD_CHUNK_MANIFEST_PATH),
            str(PARENT_CHILD_GOLD_WORKSPACE_PATH),
            "gold_chunk_ids_parent_child",
        )
    raise ValueError(f"unsupported chunk strategy: {chunk_strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gold chunk labeling workspace.")
    parser.add_argument(
        "--chunk-strategy",
        choices=[DEFAULT_CHUNK_STRATEGY, PARENT_CHILD_CHUNK_STRATEGY],
        default=DEFAULT_CHUNK_STRATEGY,
    )
    args = parser.parse_args()

    chunk_manifest_path_str, workspace_path_str, gold_chunk_field = resolve_paths(args.chunk_strategy)
    chunk_manifest_path = CHUNK_MANIFEST_PATH if args.chunk_strategy == DEFAULT_CHUNK_STRATEGY else PARENT_CHILD_CHUNK_MANIFEST_PATH
    workspace_path = GOLD_WORKSPACE_PATH if args.chunk_strategy == DEFAULT_CHUNK_STRATEGY else PARENT_CHILD_GOLD_WORKSPACE_PATH

    if not chunk_manifest_path.exists():
        raise FileNotFoundError(
            f"缺少 {chunk_manifest_path.name}。请先生成对应 chunk strategy 的 manifest。"
        )

    chunk_manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    qa_items = json.loads(QA_PATH.read_text(encoding="utf-8"))

    chunks_by_doc_id: dict[str, list[dict]] = {}
    for chunk in chunk_manifest:
        chunks_by_doc_id.setdefault(chunk["doc_id"], []).append(chunk)

    workspace = []
    for qa_item in qa_items:
        candidate_chunks = []
        for doc_id in qa_item.get("gold_doc_ids", []):
            for chunk in chunks_by_doc_id.get(doc_id, []):
                candidate_chunks.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "header_path": chunk["header_path"],
                        "preview": chunk["preview"],
                    }
                )

        workspace.append(
            {
                "question_id": qa_item["question_id"],
                "question_type": qa_item["question_type"],
                "question": qa_item["question"],
                "gold_doc_ids": qa_item.get("gold_doc_ids", []),
                gold_chunk_field: qa_item.get(gold_chunk_field, []),
                "gold_section_hints": qa_item.get("gold_section_hints", []),
                "label_status": "todo",
                "annotation_note": "请从 candidate_chunks_from_gold_docs 中选出最小充分证据块。",
                "candidate_chunks_from_gold_docs": candidate_chunks,
            }
        )

    workspace_path.write_text(
        json.dumps(workspace, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"chunk_strategy: {args.chunk_strategy}")
    print(f"chunk manifest: {chunk_manifest_path_str}")
    print(f"gold chunk labeling workspace saved to: {workspace_path_str}")


if __name__ == "__main__":
    main()
