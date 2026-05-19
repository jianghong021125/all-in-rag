import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# 使用 AIHubmix
# Settings.llm = OpenAILike(
#     model="glm-4.7-flash-free",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://aihubmix.com/v1",
#     is_chat_model=True
# )
# 使用 DeepSeek
Settings.llm = OpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com"
)
# 中文 Embedding 模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 加载本地markdown文件
docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

# 构建向量索引
index = VectorStoreIndex.from_documents(docs)

# 创建查询引擎
query_engine = index.as_query_engine()

# 提示词模板使用
print(query_engine.get_prompts())

# 查询问题
print(query_engine.query("文中举了哪些例子?"))