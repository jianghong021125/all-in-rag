import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
# 原始单次分块示例（保留作学习参考）
# text_splitter = RecursiveCharacterTextSplitter()
# chunks = text_splitter.split_documents(docs)

splitter_test_configs = [
    {"chunk_size": 500, "chunk_overlap": 50},
    {"chunk_size": 1000, "chunk_overlap": 100},
    {"chunk_size": 2000, "chunk_overlap": 150},
    {"chunk_size": 4000, "chunk_overlap": 200},
    {"chunk_size": 8000, "chunk_overlap": 400},
]

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    # model_name="BAAI/bge-base-zh-v1.5",
    # model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
# 原始单次索引构建示例（保留作学习参考）
# vectorstore = InMemoryVectorStore(embeddings)
# vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

# 使用 AIHubmix
# llm = ChatOpenAI(
#    model="glm-4.7-flash-free",
#   temperature=0.7,
#   max_tokens=4096,
#   api_key=os.getenv("DEEPSEEK_API_KEY"),
#   base_url="https://aihubmix.com/v1"
# )

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=16384,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 用户查询
question = "文中举了哪些强化学习应用的例子？人类的哪些探索与利用行为和强化学习逻辑类似？请分别回答这两个问题。"

def format_cache_hit_rate(response):
    response_metadata = getattr(response, "response_metadata", {}) or {}
    usage_metadata = getattr(response, "usage_metadata", {}) or {}
    token_usage = response_metadata.get("token_usage", {}) or {}

    prompt_tokens = token_usage.get("prompt_tokens") or usage_metadata.get("input_tokens")
    hit_tokens = token_usage.get("prompt_cache_hit_tokens")

    if hit_tokens is None:
        prompt_token_details = token_usage.get("prompt_tokens_details", {}) or {}
        hit_tokens = prompt_token_details.get("cached_tokens")

    if hit_tokens is None:
        input_token_details = usage_metadata.get("input_token_details", {}) or {}
        hit_tokens = input_token_details.get("cache_read")

    if not prompt_tokens or hit_tokens is None:
        return "N/A (当前模型响应未返回缓存命中信息)"

    return f"{hit_tokens / prompt_tokens:.2%} ({hit_tokens}/{prompt_tokens})"


def run_splitter_test(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(docs)

    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(chunks)

    retrieved_docs = vectorstore.similarity_search(question, k=5)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    response = llm.invoke(prompt.format(question=question, context=docs_content))

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "cache_hit_rate": format_cache_hit_rate(response),
        "answer_only": response.content,
    }


# 原始单次查询示例（保留作学习参考）
# retrieved_docs = vectorstore.similarity_search(question, k=5)
# docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
# answer_only = llm.invoke(prompt.format(question=question, context=docs_content)).content
# print(answer_only)

for index, config in enumerate(splitter_test_configs, start=1):
    result = run_splitter_test(**config)
    print("=" * 80)
    print(f"测试 {index}")
    print(
        "分块参数使用: "
        f"chunk_size={result['chunk_size']}, chunk_overlap={result['chunk_overlap']}"
    )
    print(f"缓存命中率: {result['cache_hit_rate']}")
    print("回答")
    print(result["answer_only"])
