from __future__ import annotations

import html
import importlib.util
import re
import sys
import textwrap
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BACKEND_PATH = APP_DIR / "05_chunker_cross_comparison.py"


@st.cache_resource
def load_backend():
    spec = importlib.util.spec_from_file_location("chunker_eval_backend", BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载后端文件: {BACKEND_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

st.set_page_config(
    page_title="简易分块评测",
    page_icon=":bar_chart:",
    layout="wide",
)

backend = load_backend()


def get_config_label(key: str):
    label_map = {
        "model_name": "模型名称",
        "device": "运行设备",
        "normalize_embeddings": "归一化向量",
        "breakpoint_threshold_type": "断点阈值类型",
        "breakpoint_threshold_amount": "断点阈值",
        "max_semantic_chunk_size": "最大语义块长度",
        "recursive_chunk_size": "递归分块长度",
        "recursive_chunk_overlap": "递归重叠长度",
        "chunk_size": "分块长度",
        "chunk_overlap": "重叠长度",
        "strip_headers": "移除标题",
        "parent_chunk_size": "父块长度",
        "parent_chunk_overlap": "父块重叠长度",
        "child_chunk_size": "子块长度",
        "child_chunk_overlap": "子块重叠长度",
        "top_k": "Top-K",
    }
    return label_map.get(key, key.replace("_", " ").title())


def get_default_breakpoint_amount(threshold_type: str):
    if threshold_type == "percentile":
        return 95.0
    if threshold_type == "interquartile":
        return 1.5
    return 1.0


def render_config_editor(config: dict):
    edited = {}

    for key, value in config.items():
        label = get_config_label(key)

        if key == "breakpoint_threshold_type":
            options = backend.get_breakpoint_threshold_options()
            current_index = options.index(value) if value in options else 0
            edited[key] = st.selectbox(label, options, index=current_index)
            continue

        if key == "breakpoint_threshold_amount":
            threshold_type = edited.get(
                "breakpoint_threshold_type",
                config.get("breakpoint_threshold_type", "gradient"),
            )
            enabled = st.checkbox(
                "启用自定义断点阈值",
                value=value is not None,
            )
            if enabled:
                default_amount = (
                    float(value)
                    if value is not None
                    else get_default_breakpoint_amount(threshold_type)
                )
                edited[key] = float(
                    st.number_input(
                        label,
                        value=default_amount,
                        step=0.1,
                        format="%.3f",
                    )
                )
            else:
                edited[key] = None
            continue

        if key == "device":
            options = ["cpu", "cuda"]
            current_index = options.index(value) if value in options else 0
            edited[key] = st.selectbox(label, options, index=current_index)
            continue

        if isinstance(value, bool):
            edited[key] = st.checkbox(label, value=value)
            continue

        if isinstance(value, int):
            step = max(1, value // 10) if value else 1
            edited[key] = int(st.number_input(label, value=value, step=step))
            continue

        if isinstance(value, float):
            edited[key] = float(st.number_input(label, value=value, step=0.1, format="%.3f"))
            continue

        edited[key] = st.text_input(label, value=str(value))

    return edited


def render_pairwise_scores(method_name: str, overlap_matrix: dict[str, dict[str, float | None]]):
    rows = []
    for other_method, score in overlap_matrix.get(method_name, {}).items():
        if other_method == method_name:
            continue
        rows.append(
            {
                "对比策略": other_method,
                "重叠得分": None if score is None else round(score, 4),
            }
        )

    rows.sort(
        key=lambda row: (
            row["重叠得分"] is None,
            0 if row["重叠得分"] is None else -row["重叠得分"],
            row["对比策略"],
        )
    )
    st.dataframe(rows, use_container_width=True)


def render_retrieval_details(retrieval: dict):
    if retrieval["error"]:
        st.warning(f"检索评估未完成: {retrieval['error']}")
        return

    detail_rows = []
    for item in retrieval["details"]:
        best_preview = item["top_chunks"][0]["preview"] if item["top_chunks"] else "-"
        detail_rows.append(
            {
                "问题": item["question"],
                "命中排序": item["hit_rank"],
                "命中分块索引": item["matched_chunk_index"],
                "最高分块预览": best_preview,
            }
        )
    st.dataframe(detail_rows, use_container_width=True)

    for item in retrieval["details"]:
        with st.expander(f"问答: {item['question']}"):
            st.write(f"命中排序: {item['hit_rank']}")
            for top_chunk in item["top_chunks"]:
                st.markdown(
                    f"- 分块 `{top_chunk['chunk_index']}` | 分数={top_chunk['score']:.4f} | "
                    f"{top_chunk['preview']}"
                )


def render_custom_qa_results(custom_result: dict | None):
    if not custom_result:
        st.info("输入问题并运行后，这里会展示自定义检索结果。")
        return

    if custom_result["error"]:
        st.warning(f"自定义检索未完成: {custom_result['error']}")
        return

    summary_rows = []
    for item in custom_result["details"]:
        top_preview = item["top_chunks"][0]["preview"] if item["top_chunks"] else "-"
        summary_rows.append(
            {
                "问题": item["question"],
                "返回块数": len(item["top_chunks"]),
                "Top-1 预览": top_preview,
            }
        )

    st.dataframe(summary_rows, use_container_width=True)

    for item in custom_result["details"]:
        with st.expander(f"自定义问题: {item['question']}"):
            for top_chunk in item["top_chunks"]:
                span = top_chunk["span"]
                span_text = "未知" if span is None else f"{span[0]} - {span[1]}"
                st.markdown(
                    f"- 分块 `{top_chunk['chunk_index']}` | 分数={top_chunk['score']:.4f} | "
                    f"区间={span_text} | 元数据={top_chunk['metadata_preview']}"
                )
                st.code(top_chunk["text"], language="markdown")


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
UNORDERED_LIST_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
ORDERED_LIST_RE = re.compile(r"^\s*\d+\.\s+(.*)$")


def render_markdown_like_html(text: str) -> str:
    html_parts: list[str] = []
    paragraph_lines: list[str] = []
    list_items: list[str] = []
    list_type: str | None = None

    def flush_paragraph():
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        paragraph_html = "<br>".join(html.escape(line) for line in paragraph_lines)
        html_parts.append(f"<p>{paragraph_html}</p>")
        paragraph_lines = []

    def flush_list():
        nonlocal list_items, list_type
        if not list_items or list_type is None:
            return
        items_html = "".join(f"<li>{html.escape(item)}</li>" for item in list_items)
        html_parts.append(f"<{list_type}>{items_html}</{list_type}>")
        list_items = []
        list_type = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            flush_paragraph()
            flush_list()
            level = len(heading_match.group(1))
            content = html.escape(heading_match.group(2).strip())
            html_parts.append(f"<h{level}>{content}</h{level}>")
            continue

        unordered_match = UNORDERED_LIST_RE.match(line)
        ordered_match = ORDERED_LIST_RE.match(line)
        if unordered_match or ordered_match:
            flush_paragraph()
            next_list_type = "ul" if unordered_match else "ol"
            item_text = (unordered_match or ordered_match).group(1).strip()
            if list_type != next_list_type:
                flush_list()
                list_type = next_list_type
            list_items.append(item_text)
            continue

        flush_list()
        paragraph_lines.append(stripped)

    flush_paragraph()
    flush_list()
    return "".join(html_parts)


def render_chunk_cards(chunk_rows: list[dict]):
    card_html_parts = [
        textwrap.dedent(
            """
            <style>
            .chunk-card {
                width: 100%;
                margin-bottom: 14px;
                background: transparent;
            }
            .chunk-text {
                padding: 10px 12px;
                font-size: 10.5px;
                line-height: 1.45;
                overflow-wrap: anywhere;
                color: #1f2937;
                background: #ffffff;
                border: 1px solid #94a3b8;
                border-radius: 10px;
                box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
            }
            .chunk-text h1,
            .chunk-text h2,
            .chunk-text h3,
            .chunk-text h4,
            .chunk-text h5,
            .chunk-text h6 {
                margin: 0 0 0.45em 0;
                line-height: 1.3;
                font-weight: 700;
                color: #0f172a;
            }
            .chunk-text h1 { font-size: 16px; }
            .chunk-text h2 { font-size: 14.5px; }
            .chunk-text h3 { font-size: 13.5px; }
            .chunk-text h4 { font-size: 12.5px; }
            .chunk-text h5 { font-size: 11.5px; }
            .chunk-text h6 { font-size: 10.8px; }
            .chunk-text p {
                margin: 0 0 0.65em 0;
                font-size: 11px;
                line-height: 1.5;
            }
            .chunk-text ul,
            .chunk-text ol {
                margin: 0 0 0.65em 1.1em;
                padding-left: 0.9em;
                font-size: 11px;
                line-height: 1.5;
            }
            .chunk-text li {
                margin: 0.1em 0;
            }
            .chunk-text > :last-child {
                margin-bottom: 0;
            }
            .chunk-info {
                min-height: 32px;
                padding: 7px 2px 0 2px;
                background: transparent;
                font-size: 11px;
                color: #334155;
                display: flex;
                flex-wrap: wrap;
                gap: 6px 12px;
                align-items: center;
            }
            .chunk-info-line {
                margin: 0;
                white-space: nowrap;
            }
            .chunk-info-label {
                font-weight: 700;
                color: #0f172a;
            }
            </style>
            <div>
            """
        ).strip()
    ]

    for row in chunk_rows:
        span_text = "未知" if row["start"] is None else f"{row['start']} - {row['end']}"
        text_html = render_markdown_like_html(row["text"])
        metadata_html = html.escape(str(row["metadata_preview"]))
        card_html_parts.append(
            (
                f'<div class="chunk-card">'
                f'<div class="chunk-text">{text_html}</div>'
                f'<div class="chunk-info">'
                f'<div class="chunk-info-line"><span class="chunk-info-label">分块编号:</span> {row["chunk_index"]}</div>'
                f'<div class="chunk-info-line"><span class="chunk-info-label">长度:</span> {row["length"]}</div>'
                f'<div class="chunk-info-line"><span class="chunk-info-label">区间:</span> {html.escape(span_text)}</div>'
                f'<div class="chunk-info-line"><span class="chunk-info-label">元数据预览:</span> {metadata_html}</div>'
                f"</div>"
                f"</div>"
            )
        )

    card_html_parts.append("</div>")
    st.markdown("".join(card_html_parts), unsafe_allow_html=True)


st.title("简易分块评测")
st.caption("对单个分块策略进行可视化、打分，并和其他策略做重叠度参考比较。")

available_docs = backend.list_markdown_documents()
doc_names = [path.name for path in available_docs]

with st.sidebar:
    st.header("控制面板")
    st.write(f"Markdown 目录: `{backend.get_markdown_directory()}`")

    uploaded_file = st.file_uploader("上传 .md 文件", type=["md"])
    if uploaded_file is not None:
        if st.button("保存上传文件"):
            saved_path = backend.save_markdown_file(uploaded_file.name, uploaded_file.getvalue())
            st.success(f"已保存: {saved_path.name}")
            st.rerun()

    if not doc_names:
        st.error("当前没有可用的 Markdown 文档。请先上传 .md 文件。")
        st.stop()

    selected_doc_name = st.selectbox("文档", doc_names)
    selected_doc_path = next(path for path in available_docs if path.name == selected_doc_name)

    strategy_name = st.selectbox("分块策略", backend.get_strategy_names())
    st.caption("可调参数仅包含策略配置，分隔符保持固定。")

    default_config = backend.get_default_strategy_config(strategy_name)
    config_override = render_config_editor(default_config)

    run_clicked = st.button("运行评测", type="primary")


if "chunker_eval_state" not in st.session_state:
    st.session_state.chunker_eval_state = None
if "custom_qa_state" not in st.session_state:
    st.session_state.custom_qa_state = None

if run_clicked:
    with st.spinner("正在运行所选策略并计算评测结果..."):
        bundle = backend.run_selected_strategy_bundle(
            strategy_name,
            doc_path=selected_doc_path,
            config_override=config_override,
        )
        st.session_state.chunker_eval_state = {
            "strategy_name": strategy_name,
            "doc_path": str(selected_doc_path),
            "config_override": config_override,
            "bundle": bundle,
        }
        st.session_state.custom_qa_state = None


state = st.session_state.chunker_eval_state
if state is None:
    st.info("在左侧选择文档与策略后，点击 `运行评测` 开始。")
    st.stop()


bundle = state["bundle"]
selected_strategy = state["strategy_name"]
selected_result = bundle["results"][selected_strategy]
selected_metrics = bundle["metrics"][selected_strategy]
selected_retrieval = bundle["retrieval"][selected_strategy]
selected_chunk_rows = backend.build_chunk_rows(selected_result)

st.subheader("运行上下文")
st.write(f"文档: `{bundle['doc_path'].name}`")
st.write(f"所选策略: `{selected_strategy}`")
st.write(f"原文长度: `{len(bundle['source_text'])}` 字符")
st.json(state["config_override"], expanded=False)

if selected_result.error:
    st.error(selected_result.error)
    st.stop()


metric_columns = st.columns(6)
metric_columns[0].metric("分块数", selected_metrics["chunk_count"])
metric_columns[1].metric("平均长度", f"{selected_metrics['avg_length']:.1f}")
metric_columns[2].metric("方差得分", f"{selected_metrics['length_variance_score']:.3f}")
metric_columns[3].metric(
    "标题保留得分",
    backend.format_optional_score(selected_metrics["header_preservation_score"]),
)
metric_columns[4].metric(
    "检索得分",
    backend.format_optional_score(selected_metrics["retrieval_score"]),
)
metric_columns[5].metric(
    "平均重叠度",
    backend.format_optional_score(selected_metrics["avg_pairwise_overlap"]),
)

secondary_metrics = st.columns(4)
secondary_metrics[0].metric("中位长度", f"{selected_metrics['median_length']:.1f}")
secondary_metrics[1].metric(
    "最短 / 最长",
    f"{selected_metrics['min_length']} / {selected_metrics['max_length']}",
)
secondary_metrics[2].metric(
    f"命中率@{selected_retrieval['top_k']}",
    backend.format_optional_score(selected_metrics["retrieval_hit_rate"]),
)
secondary_metrics[3].metric(
    "MRR",
    backend.format_optional_score(selected_metrics["retrieval_mrr"]),
)

st.subheader("两两重叠对比")
st.caption("这里显示当前所选策略与其他默认策略之间的 chunk 边界重叠度。")
render_pairwise_scores(selected_strategy, bundle["pairwise_overlap_matrix"])

retrieval_left, retrieval_right = st.columns(2)

with retrieval_left:
    st.subheader("检索评测")
    st.caption(f"QA 集来源: `{bundle['qa_source']}` | QA 数量: `{len(bundle['qa_examples'])}`")
    render_retrieval_details(selected_retrieval)

with retrieval_right:
    st.subheader("自定义问答检索")
    st.caption("适用于新上传文档。最多输入 5 个问题，仅返回检索结果，不自动计算标准答案得分。")
    with st.form("custom_qa_form"):
        custom_top_k = st.number_input(
            "返回块数 Top-K",
            min_value=1,
            max_value=10,
            value=backend.get_default_retrieval_config()["top_k"],
            step=1,
        )
        custom_questions = [
            st.text_input(f"问题 {index + 1}", key=f"custom_question_{index}")
            for index in range(5)
        ]
        custom_run_clicked = st.form_submit_button("运行自定义检索")

    if custom_run_clicked:
        cleaned_questions = [question.strip() for question in custom_questions if question.strip()]
        st.session_state.custom_qa_state = backend.retrieve_top_chunks_for_questions(
            selected_result,
            cleaned_questions[:5],
            retrieval_config={"top_k": int(custom_top_k)},
        )

    render_custom_qa_results(st.session_state.custom_qa_state)

st.subheader("分块序列")
st.caption("每个分块单独展示为带边框的正文块，正文统一使用较小字号，信息显示在正文下方。")
render_chunk_cards(selected_chunk_rows)
