import argparse
from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="加载 LlamaIndex 本地存储索引并执行相似性搜索。"
    )
    parser.add_argument(
        "--query",
        default="LlamaIndex是做什么的？",
        help="要检索的查询文本。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="返回最相似结果的数量。",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(script_dir / "llamaindex_index_store"),
        help="LlamaIndex 持久化目录。",
    )
    parser.add_argument(
        "--embed-model",
        default="BAAI/bge-small-zh-v1.5",
        help="与建索引时保持一致的嵌入模型名称或本地路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    persist_dir = Path(args.persist_dir).resolve()

    if not persist_dir.exists():
        raise FileNotFoundError(f"未找到索引目录: {persist_dir}")

    # 加载索引时，查询嵌入模型必须与建索引时保持一致。
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embed_model)

    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=args.top_k)

    results = retriever.retrieve(args.query)

    print(f"索引目录: {persist_dir}")
    print(f"查询内容: {args.query}")
    print(f"返回条数: {len(results)}\n")

    for idx, item in enumerate(results, start=1):
        score = f"{item.score:.4f}" if item.score is not None else "N/A"
        content = item.node.get_content(metadata_mode="none").strip()

        print(f"[结果 {idx}] 相似度分数: {score}")
        print(f"文本内容: {content}")
        if item.node.metadata:
            print(f"元数据: {item.node.metadata}")
        print("-" * 60)


if __name__ == "__main__":
    main()
