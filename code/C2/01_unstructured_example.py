from unstructured.partition.pdf import partition_pdf
# from unstructured.partition.auto import partition

# PDF文件路径
pdf_path = "../../data/C2/pdf/rag.pdf"

# 使用Unstructured加载并解析PDF文档

# partition 版本
# elements = partition(
#     filename=pdf_path,
#     content_type="application/pdf"
# )

# partition_pdf 版本 (hi_res策略)
# elements = partition_pdf(
#     filename=pdf_path,
#     strategy="hi_res",
#     languages=["chi_sim"]
# )

# partition_pdf 版本 (ocr_only策略)
elements = partition_pdf(
    filename=pdf_path,
    strategy="ocr_only",
    languages=["chi_sim"]
)

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
from collections import Counter
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)