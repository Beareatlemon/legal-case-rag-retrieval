from __future__ import annotations

from html import escape
from numbers import Real

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="LeCaRD Legal RAG Workbench", layout="wide")


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --paper: #f7f1e4;
            --paper-soft: #fffaf0;
            --ink: #182522;
            --muted: #61716d;
            --line: rgba(24, 37, 34, 0.12);
            --jade: #0f6b5f;
            --gold: #bf8a2c;
            --field: rgba(255, 248, 232, 0.88);
            --field-strong: rgba(246, 236, 214, 0.96);
            --field-focus: rgba(231, 244, 236, 0.98);
            --field-border: rgba(15, 107, 95, 0.16);
            --shadow: 0 22px 70px rgba(22, 35, 31, 0.12);
        }

        .stApp {
            color: var(--ink);
            background:
                radial-gradient(circle at top left, rgba(191, 138, 44, 0.18), transparent 32rem),
                radial-gradient(circle at 82% 0%, rgba(15, 107, 95, 0.16), transparent 30rem),
                linear-gradient(135deg, #f8f0df 0%, #eef4ef 54%, #f7efe3 100%);
        }

        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 4rem;
            max-width: 1320px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #16312c 0%, #233d38 54%, #294740 100%);
            border-right: 1px solid rgba(24, 37, 34, 0.12);
        }

        [data-testid="stSidebar"] * {
            font-family: "Noto Serif SC", "Source Han Serif SC", "Microsoft YaHei", serif;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] span {
            color: rgba(255, 248, 232, 0.92) !important;
        }

        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: rgba(255, 248, 232, 0.92);
        }

        h1, h2, h3, .hero-title, .section-title {
            font-family: "Noto Serif SC", "Source Han Serif SC", "Microsoft YaHei", serif;
            letter-spacing: -0.02em;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.2rem;
            border: 1px solid rgba(24, 37, 34, 0.1);
            border-radius: 30px;
            background:
                linear-gradient(135deg, rgba(255, 250, 240, 0.94), rgba(237, 245, 239, 0.84)),
                repeating-linear-gradient(90deg, rgba(24, 37, 34, 0.035) 0, rgba(24, 37, 34, 0.035) 1px, transparent 1px, transparent 46px);
            box-shadow: var(--shadow);
        }

        .hero::after {
            content: "";
            position: absolute;
            right: -7rem;
            top: -8rem;
            width: 21rem;
            height: 21rem;
            border-radius: 50%;
            border: 1px solid rgba(15, 107, 95, 0.18);
            background: radial-gradient(circle, rgba(15, 107, 95, 0.18), transparent 64%);
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--jade);
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-size: 0.78rem;
        }

        .eyebrow::before {
            content: "";
            width: 2rem;
            height: 1px;
            background: var(--gold);
        }

        .hero-title {
            margin: 0.45rem 0 0.65rem;
            max-width: 820px;
            color: var(--ink);
            font-size: clamp(2.15rem, 5vw, 4.25rem);
            line-height: 1.04;
        }

        .hero-body {
            max-width: 780px;
            color: var(--muted);
            font-size: 1.04rem;
            line-height: 1.85;
        }

        .hero-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1.35rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.48rem 0.72rem;
            border-radius: 999px;
            color: #17302b;
            background: rgba(255, 255, 255, 0.68);
            border: 1px solid rgba(24, 37, 34, 0.1);
            font-size: 0.82rem;
            font-weight: 650;
        }

        .section-head {
            margin: 0.25rem 0 1rem;
        }

        .section-title {
            margin: 0;
            color: var(--ink);
            font-size: 1.52rem;
        }

        .section-copy {
            color: var(--muted);
            line-height: 1.75;
            margin-top: 0.35rem;
        }

        .asset-card {
            min-height: 7.2rem;
            padding: 1rem;
            border-radius: 20px;
            border: 1px solid var(--line);
            background: rgba(255, 250, 240, 0.84);
            box-shadow: 0 10px 30px rgba(22, 35, 31, 0.06);
        }

        .asset-card.ready {
            border-color: rgba(15, 107, 95, 0.22);
            background: linear-gradient(180deg, rgba(255, 250, 240, 0.88), rgba(231, 244, 236, 0.86));
        }

        .asset-card.missing {
            border-color: rgba(168, 71, 50, 0.18);
            background: linear-gradient(180deg, rgba(255, 250, 240, 0.88), rgba(251, 231, 224, 0.74));
        }

        .asset-label {
            color: var(--muted);
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .asset-value {
            margin-top: 0.45rem;
            color: var(--ink);
            font-size: 1.15rem;
            font-weight: 760;
        }

        .asset-note {
            margin-top: 0.45rem;
            color: var(--muted);
            font-size: 0.82rem;
        }

        .answer-card {
            padding: 1.45rem;
            border-radius: 26px;
            color: var(--ink);
            background:
                linear-gradient(135deg, rgba(255, 250, 240, 0.95), rgba(233, 243, 238, 0.88));
            border: 1px solid rgba(15, 107, 95, 0.18);
            box-shadow: var(--shadow);
            line-height: 1.85;
        }

        .source-card {
            margin-bottom: 0.85rem;
            padding: 1rem;
            border-radius: 20px;
            background: rgba(255, 250, 240, 0.72);
            border: 1px solid var(--line);
        }

        .source-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            color: var(--muted);
            font-size: 0.82rem;
            margin-bottom: 0.55rem;
        }

        .source-title {
            color: var(--ink);
            font-weight: 760;
            margin-bottom: 0.42rem;
        }

        .source-text {
            color: #30423e;
            line-height: 1.75;
            font-size: 0.94rem;
        }

        .config-summary {
            margin: 0.55rem 0 0.85rem;
            padding: 0.78rem 0.9rem;
            border-radius: 16px;
            color: #30423e;
            background: linear-gradient(180deg, rgba(255, 248, 232, 0.78), rgba(237, 245, 239, 0.68));
            border: 1px solid rgba(15, 107, 95, 0.13);
            line-height: 1.65;
            font-size: 0.88rem;
            overflow-wrap: anywhere;
        }

        .config-summary strong {
            color: #17302b;
        }

        [data-testid="stSidebar"] .config-summary {
            color: rgba(255, 248, 232, 0.94);
            background: rgba(255, 250, 240, 0.12);
            border-color: rgba(255, 248, 232, 0.24);
        }

        [data-testid="stSidebar"] .config-summary strong {
            color: #fff8e8;
        }

        .sidebar-brand {
            padding: 1rem 0.2rem 1.25rem;
            color: #fff8e8;
        }

        .sidebar-brand-title {
            font-size: 1.15rem;
            font-weight: 820;
            line-height: 1.35;
        }

        .sidebar-brand-copy {
            margin-top: 0.45rem;
            color: rgba(255, 248, 232, 0.76);
            font-size: 0.86rem;
            line-height: 1.65;
        }

        [data-testid="stTabs"] button [data-testid="stMarkdownContainer"] p,
        [data-testid="stTabs"] button p,
        label,
        .stMarkdown,
        .stCaption,
        .stAlert,
        .stCheckbox,
        .stSelectbox,
        .stTextArea,
        .stTextInput,
        .stNumberInput {
            color: var(--ink) !important;
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"],
        [data-testid="stMetric"] [data-testid="stMetricValue"] *,
        [data-testid="stMetricValue"] {
            color: #000000 !important;
        }

        .stTextArea textarea,
        .stTextInput input,
        .stNumberInput input,
        [data-baseweb="select"] > div,
        [data-baseweb="select"] input {
            color: var(--ink) !important;
            background-color: rgba(255, 250, 240, 0.92) !important;
        }

        [data-baseweb="popover"] *,
        [role="listbox"] *,
        [role="option"] {
            color: var(--ink) !important;
            background-color: rgba(255, 250, 240, 0.98) !important;
        }

        .stButton > button {
            color: #fffaf0 !important;
            background: linear-gradient(135deg, #0f6b5f, #1e8a76) !important;
            border: 1px solid rgba(11, 64, 56, 0.18) !important;
        }

        .stButton > button *,
        .stButton > button p,
        .stButton > button span,
        .stButton > button div {
            color: #fffaf0 !important;
            fill: #fffaf0 !important;
        }

        .stButton > button:hover,
        .stButton > button:focus,
        .stButton > button:active {
            color: #fffaf0 !important;
            background: linear-gradient(135deg, #0d5f55, #197562) !important;
        }

        .stButton > button:hover *,
        .stButton > button:focus *,
        .stButton > button:active * {
            color: #fffaf0 !important;
            fill: #fffaf0 !important;
        }

        .stCheckbox label,
        .stCheckbox p,
        .stCheckbox span,
        .stCheckbox div,
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] p,
        [data-testid="stCheckbox"] span,
        [data-testid="stCheckbox"] div,
        [data-baseweb="checkbox"] + div,
        [data-baseweb="checkbox"] ~ div,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }

        textarea::placeholder,
        input::placeholder,
        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
            color: #556762 !important;
            opacity: 1 !important;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero {
                padding: 1.35rem;
                border-radius: 22px;
            }

            .hero-title {
                font-size: 2.2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def post_json(api_base: str, path: str, payload: dict, timeout: int = 120) -> dict:
    resp = requests.post(f"{api_base}{path}", json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} {resp.text}")
    return resp.json()


def get_json(api_base: str, path: str, timeout: int = 30) -> dict:
    resp = requests.get(f"{api_base}{path}", timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} {resp.text}")
    return resp.json()


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">LeCaRDv2 Legal Retrieval</div>
            <div class="hero-title">Legal Case Retrieval RAG Workbench</div>
            <div class="hero-body">
                Built on LeCaRDv2 with bge-m3 retrieval, fusion ranking, and answer generation,
                this workspace provides one place for similar-case search, benchmarking, and demos.
            </div>
            <div class="hero-strip">
                <span class="pill">Case-level retrieval</span>
                <span class="pill">Vector fusion</span>
                <span class="pill">Structured chunks</span>
                <span class="pill">Benchmark ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_head(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <div class="eyebrow">{escape(eyebrow)}</div>
            <div class="section-title">{escape(title)}</div>
            <div class="section-copy">{escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def config_summary(label: str, value: str) -> None:
    st.markdown(
        f'<div class="config-summary"><strong>{escape(label)}:</strong> {escape(value)}</div>',
        unsafe_allow_html=True,
    )


def asset_card(label: str, ready: bool, note: str) -> None:
    state = "ready" if ready else "missing"
    value = "Ready" if ready else "Missing"
    st.markdown(
        f"""
        <div class="asset-card {state}">
            <div class="asset-label">{escape(label)}</div>
            <div class="asset-value">{value}</div>
            <div class="asset-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_asset_overview(status: dict) -> None:
    checks = [
        ("LeCaRDv2", status.get("lecardv2_ready", False), "Knowledge source"),
        ("Train pairs", status.get("train_pairs_ready", False), "Fine-tuning samples"),
        ("RAG index", status.get("rag_index_ready", False), "Retrieval index"),
    ]
    cols = st.columns(len(checks))
    for col, (label, ready, note) in zip(cols, checks):
        with col:
            asset_card(label, bool(ready), note)


def render_answer(answer: str) -> None:
    paragraphs = "<br><br>".join(escape(part.strip()) for part in answer.split("\n") if part.strip())
    st.markdown(f'<div class="answer-card">{paragraphs}</div>', unsafe_allow_html=True)


def render_source_cards(source_chunks: list[dict]) -> None:
    if not source_chunks:
        st.info("No retrieved cases were returned for this answer.")
        return

    preview_chars = 900
    for item in source_chunks:
        title = item.get("title") or item.get("doc_id", "")
        section = item.get("section") or "Unlabeled section"
        citation = item.get("citation") or item.get("chunk_id", "")
        support_count = item.get("support_count")
        result_type = item.get("result_type") or "case"
        score = item.get("score")
        score_text = f"{float(score):.4f}" if isinstance(score, Real) else str(score)
        text = item.get("text", "")
        supporting_chunks = item.get("supporting_chunks") or []
        best_chunk = supporting_chunks[0] if supporting_chunks else {}
        best_chunk_text = str(best_chunk.get("text") or "")
        best_chunk_citation = best_chunk.get("citation") or best_chunk.get("chunk_id") or citation
        best_chunk_score = best_chunk.get("score")
        preview_source = best_chunk_text or text
        preview = preview_source if len(preview_source) <= preview_chars else f"{preview_source[:preview_chars]}..."
        support_text = f"{int(support_count)} supporting chunks" if isinstance(support_count, int) else "case result"
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-meta">
                    <span>Rank {item.get("rank", "-")}</span>
                    <span>{escape(str(result_type)).title()}</span>
                    <span>Score {escape(score_text)}</span>
                    <span>{escape(support_text)}</span>
                    <span>{escape(section)}</span>
                </div>
                <div class="source-title">{escape(title)}</div>
                <div class="source-meta"><span>Case ID: {escape(str(item.get("doc_id", "")))}</span><span>Reference: {escape(citation)}</span></div>
                <div class="source-meta"><span>Preview: highest-scoring chunk</span><span>{escape(str(best_chunk_citation))}</span><span>Chunk score: {escape(str(best_chunk_score))}</span></div>
                <div class="source-text">{escape(preview)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if text:
            with st.expander(f"Expand full case {item.get('doc_id', '')}", expanded=False):
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-meta">
                            <span>Full case</span>
                            <span>Case ID: {escape(str(item.get("doc_id", "")))}</span>
                        </div>
                        <div class="source-text">{escape(text)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if supporting_chunks:
                    st.markdown("**Supporting chunks used for retrieval**")
                    for idx, chunk in enumerate(supporting_chunks, start=1):
                        chunk_text = str(chunk.get("text") or "")
                        chunk_preview = chunk_text if len(chunk_text) <= 700 else f"{chunk_text[:700]}..."
                        st.caption(
                            f"{idx}. {chunk.get('citation') or chunk.get('chunk_id')} | "
                            f"section={chunk.get('section') or 'general'} | score={chunk.get('score')}"
                        )
                        st.write(chunk_preview)


inject_theme()
render_hero()

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">Legal RAG Console</div>
        <div class="sidebar-brand-copy">Connect to the backend, inspect project assets, and run retrieval, QA, and benchmark workflows.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
api_base = "http://127.0.0.1:8000"
st.sidebar.markdown(
    f'<div class="config-summary"><strong>Backend URL:</strong> {escape(api_base)}</div>',
    unsafe_allow_html=True,
)

status: dict = {}
available_paths: set[str] = set()

try:
    status = get_json(api_base, "/api/status/assets")
except Exception as exc:
    st.warning(f"Unable to read backend asset status yet: {exc}")

if status:
    st.write("")
    render_asset_overview(status)
    if status.get("latest_checkpoint"):
        st.sidebar.caption(f"Latest checkpoint: {status['latest_checkpoint']}")
    st.sidebar.caption(f"Checkpoint root: {status['checkpoints_dir']}")
    st.sidebar.caption(f"Runtime logs: {status['runtime_logs_dir']}")

try:
    openapi = get_json(api_base, "/openapi.json")
    available_paths = set(openapi.get("paths", {}).keys())
except Exception:
    available_paths = set()


tab_rag_query, tab_model_eval, tab_train = st.tabs(["RAG QA", "Benchmark", "Training"])

with tab_rag_query:
    section_head(
        "Interactive Demo",
        "Similar-Case Retrieval QA",
        "Enter case facts or a dispute focus. The system retrieves similar LeCaRDv2 cases first, then uses the most relevant evidence to generate the answer.",
    )
    left, right = st.columns([1.35, 0.65], gap="large")
    with left:
        rag_query = st.text_area(
            "Case facts or retrieval question",
            height=220,
            placeholder=(
                "示例：被告人在数月内多次贩卖毒品，每次交易金额约为100至130元，庭审中对主要事实没有异议，"
                "案卷材料包含证人证言、交易记录和现场检测报告。请检索最相似的案例。"
            ),
        )
        ask_clicked = st.button("Retrieve and generate answer", use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("#### Retrieval Settings")
            retrieval_presets = {
                "Standard demo: top5 / retrieve20": {"top_k": 5, "retrieve_k": 20, "vector_weight": 1.0, "bm25_weight": 1.0},
                "High recall: top8 / retrieve50": {"top_k": 8, "retrieve_k": 50, "vector_weight": 1.0, "bm25_weight": 1.0},
                "Fast response: top3 / retrieve10": {"top_k": 3, "retrieve_k": 10, "vector_weight": 1.0, "bm25_weight": 1.0},
            }
            retrieval_mode = st.selectbox("Retrieval mode", options=["hybrid", "vector", "vector_fusion", "bm25"], index=2)
            retrieval_preset = st.selectbox("Retrieval preset", options=list(retrieval_presets), index=0)
            selected_retrieval = retrieval_presets[retrieval_preset]
            top_k = selected_retrieval["top_k"]
            retrieve_k = selected_retrieval["retrieve_k"]
            vector_weight = selected_retrieval["vector_weight"]
            bm25_weight = selected_retrieval["bm25_weight"]
            config_summary("Top-K contexts", str(top_k))
            config_summary("Retrieve-K", str(retrieve_k))
            include_contexts = st.checkbox("Include evidence text", value=True)
            use_reranker = st.checkbox("Enable experimental reranker", value=False)
            reranker_type = "bi_encoder_fusion"
            reranker_model = "models/experiments/bgem3_labeled_hn/bgem3_qrels_boost_margin_q120_cuda"
            reranker_fusion_weight = 0.15
            reranker_window = 10
            config_summary(
                "Reranker",
                (
                    f"Experimental bge-m3 fusion reranker / window {reranker_window} / weight {reranker_fusion_weight:.2f}"
                    if use_reranker
                    else "Disabled"
                ),
            )
            config_summary("Fusion weights", f"Vector {vector_weight:.1f} / BM25 {bm25_weight:.1f}")

    if ask_clicked:
        if not rag_query.strip():
            st.warning("Please enter case facts or a retrieval question first.")
        else:
            try:
                with st.spinner("Retrieving similar cases and generating the answer..."):
                    data = post_json(
                        api_base,
                        "/api/rag/query",
                        {
                            "query": rag_query,
                            "top_k": int(top_k),
                            "retrieve_k": int(retrieve_k),
                            "include_contexts": include_contexts,
                            "retrieval_mode": retrieval_mode,
                            "vector_weight": float(vector_weight),
                            "bm25_weight": float(bm25_weight),
                            "use_reranker": use_reranker,
                            "reranker_type": reranker_type,
                            "reranker_model": reranker_model,
                            "reranker_fusion_weight": reranker_fusion_weight,
                            "reranker_window": reranker_window,
                        },
                        timeout=300,
                    )
            except Exception as exc:
                st.error(f"RAG query failed: {exc}")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Cases", data["contexts_used"])
                c2.metric("Candidates", data["retrieved_candidates"])
                c3.metric("Mode", data["retrieval_mode"])
                c4.metric("Backend", data["backend"])
                st.caption(
                    f"Retriever: {data['retrieval_model']} | "
                    f"Reranker: {data.get('reranker_model') or 'disabled'}"
                )
                render_answer(data["answer"])
                st.write("")
                section_head("Evidence", "Retrieved cases", "Each card is one case. The preview is the strongest supporting chunk from that case.")
                render_source_cards(data["source_chunks"])
                with st.expander("View raw source table"):
                    st.dataframe(pd.DataFrame(data["source_chunks"]), use_container_width=True)

with tab_model_eval:
    section_head(
        "Evaluation",
        "Retrieval and generation benchmark",
        "Compare retrieval settings, reranking, and optional generation evaluation in one benchmark panel.",
    )
    rag_eval_available = "/api/eval/rag" in available_paths
    default_data_dir = status.get("default_lecardv2_dir", "data/lecardv2")

    with st.container(border=True):
        st.markdown("#### RAG Pipeline Benchmark")
        rag_eval_data_dir = default_data_dir
        rag_eval_presets = {
            "Standard comparison": {
                "top_k": 5,
                "retrieve_k": 20,
                "vector_weight": 1.0,
                "bm25_weight": 1.0,
                "max_queries": 20,
                "generation_sample_size": 10,
            },
            "Full retrieval benchmark": {
                "top_k": 5,
                "retrieve_k": 20,
                "vector_weight": 1.0,
                "bm25_weight": 1.0,
                "max_queries": 1000000,
                "generation_sample_size": 10,
            },
            "Quick smoke benchmark": {
                "top_k": 3,
                "retrieve_k": 10,
                "vector_weight": 1.0,
                "bm25_weight": 1.0,
                "max_queries": 5,
                "generation_sample_size": 3,
            },
        }
        config_summary("RAG eval data", rag_eval_data_dir)
        rag_eval_mode = st.selectbox("RAG retrieval mode", options=["hybrid", "vector", "vector_fusion", "bm25"], index=2)
        rag_eval_preset = st.selectbox("RAG benchmark preset", options=list(rag_eval_presets), index=0)
        selected_rag_eval = rag_eval_presets[rag_eval_preset]
        rag_eval_top_k = selected_rag_eval["top_k"]
        rag_eval_retrieve_k = selected_rag_eval["retrieve_k"]
        rag_eval_vector_weight = selected_rag_eval["vector_weight"]
        rag_eval_bm25_weight = selected_rag_eval["bm25_weight"]
        rag_eval_max_queries = selected_rag_eval["max_queries"]
        rag_eval_generation_sample = selected_rag_eval["generation_sample_size"]
        config_summary(
            "Retrieval scope",
            f"Top-K {rag_eval_top_k} / Retrieve-K {rag_eval_retrieve_k} / Queries {'all' if rag_eval_max_queries >= 1000000 else rag_eval_max_queries}",
        )
        config_summary("Fusion weights", f"Vector {rag_eval_vector_weight:.1f} / BM25 {rag_eval_bm25_weight:.1f}")
        rag_eval_use_reranker = st.checkbox("Benchmark experimental reranker", value=False)
        rag_eval_reranker_type = "bi_encoder_fusion"
        rag_eval_reranker_model = "models/experiments/bgem3_labeled_hn/bgem3_qrels_boost_margin_q120_cuda"
        rag_eval_reranker_fusion_weight = 0.15
        rag_eval_reranker_window = 10
        config_summary(
            "Benchmark reranker",
            (
                f"Experimental bge-m3 fusion reranker / window {rag_eval_reranker_window} / "
                f"weight {rag_eval_reranker_fusion_weight:.2f}"
            )
            if rag_eval_use_reranker
            else "Disabled",
        )
        rag_eval_generation = st.checkbox("Benchmark generation layer", value=True)
        config_summary("Generation sample", str(rag_eval_generation_sample))

        if not rag_eval_available:
            st.error("Current backend does not expose `/api/eval/rag`. Restart the backend with the latest code.")

        run_rag_eval = st.button("Run RAG benchmark", use_container_width=True, disabled=not rag_eval_available)

    if run_rag_eval:
        try:
            with st.spinner("Running the RAG benchmark..."):
                data = post_json(
                    api_base,
                    "/api/eval/rag",
                    {
                        "data_dir": rag_eval_data_dir,
                        "retrieval_mode": rag_eval_mode,
                        "top_k": int(rag_eval_top_k),
                        "retrieve_k": int(rag_eval_retrieve_k),
                        "vector_weight": float(rag_eval_vector_weight),
                        "bm25_weight": float(rag_eval_bm25_weight),
                        "max_queries": int(rag_eval_max_queries),
                        "use_reranker": rag_eval_use_reranker,
                        "reranker_type": rag_eval_reranker_type,
                        "reranker_model": rag_eval_reranker_model,
                        "reranker_fusion_weight": rag_eval_reranker_fusion_weight,
                        "reranker_window": rag_eval_reranker_window,
                        "evaluate_generation": rag_eval_generation,
                        "generation_sample_size": int(rag_eval_generation_sample),
                    },
                    timeout=1800,
                )
        except Exception as exc:
            st.error(f"RAG benchmark failed: {exc}")
        else:
            st.success("RAG benchmark finished.")
            st.caption(
                f"Retrieval: {data['retrieval_mode']} | Retriever: {data['retrieval_model']} | "
                f"Reranker: {data.get('reranker_model') or 'disabled'} | Queries: {data['query_count']}"
            )
            st.write("Retrieval layer")
            st.dataframe(pd.DataFrame([data["retrieval_layer"]["metrics"]]), use_container_width=True)
            if data.get("rerank_layer"):
                st.write("Rerank layer")
                st.dataframe(pd.DataFrame([data["rerank_layer"]["metrics"]]), use_container_width=True)
            if data.get("generation_layer"):
                st.write("Generation layer")
                st.dataframe(pd.DataFrame([data["generation_layer"]["metrics"]]), use_container_width=True)
            if data.get("notes"):
                for note in data["notes"]:
                    st.caption(note)

with tab_train:
    section_head(
        "Training",
        "Bi-Encoder Training",
        "This panel is kept for experiment reproduction. Day-to-day demos should usually use the RAG QA and benchmark tabs.",
    )
    with st.container(border=True):
        preset_options = [
            "none",
            "rtx4060",
            "rtx4090",
            "bgem3_rtx4060",
            "bgem3_triplet_rtx4060",
            "bgem3_conservative_rtx4060",
            "bgem3_margin_rtx4060",
            "bgem3_labeled_rtx4060",
        ]
        preset = st.selectbox(
            "Hardware preset",
            options=preset_options,
            index=preset_options.index("bgem3_labeled_rtx4060"),
        )
        base_model = "BAAI/bge-large-zh-v1.5"
        default_train_pairs_path = status.get("default_train_pairs", "data/train/lecardv2_train_pairs.jsonl")
        default_dev_pairs_path = status.get("default_dev_pairs", "data/train/lecardv2_dev_pairs.jsonl")
        default_bgem3_train_pairs_path = status.get("default_lecardv2_train_pairs", default_train_pairs_path)
        default_bgem3_dev_pairs_path = status.get("default_lecardv2_dev_pairs", "data/train/lecardv2_dev_pairs.jsonl")
        default_bgem3_triplet_pairs_path = status.get("default_lecardv2_chunk_train_triplets", default_bgem3_train_pairs_path)
        default_bgem3_triplet_dev_pairs_path = status.get(
            "default_lecardv2_chunk_dev_triplets",
            "data/train/lecardv2_chunk_dev_triplets.jsonl",
        )
        default_bgem3_labeled_pairs_path = status.get(
            "default_lecardv2_chunk_train_labeled_pairs",
            "data/train/experiments/bgem3_labeled_hn/lecardv2_chunk_train_labeled_pairs.jsonl",
        )
        default_bgem3_labeled_dev_pairs_path = status.get(
            "default_lecardv2_chunk_dev_labeled_pairs",
            "data/train/experiments/bgem3_labeled_hn/lecardv2_chunk_dev_labeled_pairs.jsonl",
        )
        default_output_dir = ""
        default_checkpoint_dir = ""
        if preset in {"bgem3_rtx4060", "bgem3_conservative_rtx4060"}:
            base_model = status.get("default_bgem3_model_dir", "BAAI/bge-m3")
            default_train_pairs_path = default_bgem3_train_pairs_path
            default_dev_pairs_path = default_bgem3_dev_pairs_path
            default_output_dir = status.get("default_bgem3_experiment_model_dir", "")
            default_checkpoint_dir = status.get("default_bgem3_experiment_checkpoint_dir", "")
        elif preset == "bgem3_triplet_rtx4060":
            base_model = status.get("default_bgem3_model_dir", "BAAI/bge-m3")
            default_train_pairs_path = default_bgem3_triplet_pairs_path
            default_dev_pairs_path = default_bgem3_triplet_dev_pairs_path
            default_output_dir = status.get("default_bgem3_triplet_experiment_model_dir", "")
            default_checkpoint_dir = status.get("default_bgem3_triplet_experiment_checkpoint_dir", "")
        elif preset == "bgem3_margin_rtx4060":
            base_model = status.get("default_bgem3_model_dir", "BAAI/bge-m3")
            default_train_pairs_path = status.get(
                "default_lecardv2_chunk_train_margin_triplets",
                "data/train/lecardv2_chunk_train_margin_triplets.jsonl",
            )
            default_dev_pairs_path = status.get(
                "default_lecardv2_chunk_dev_margin_triplets",
                "data/train/lecardv2_chunk_dev_margin_triplets.jsonl",
            )
            default_output_dir = "models/bgem3_lecardv2_margin_experiment"
            default_checkpoint_dir = "models/checkpoints/bgem3_lecardv2_margin_experiment"
        elif preset == "bgem3_labeled_rtx4060":
            base_model = status.get("default_bgem3_model_dir", "BAAI/bge-m3")
            default_train_pairs_path = default_bgem3_labeled_pairs_path
            default_dev_pairs_path = default_bgem3_labeled_dev_pairs_path
            default_output_dir = "models/experiments/bgem3_labeled_hn/bgem3_lecardv2_labeled_hn"
            default_checkpoint_dir = "models/experiments/bgem3_labeled_hn/checkpoints/bgem3_lecardv2_labeled_hn"

        train_pairs_path = default_train_pairs_path
        dev_pairs_path = default_dev_pairs_path
        output_dir = default_output_dir
        checkpoint_dir = default_checkpoint_dir
        config_summary("Base model", base_model)
        config_summary("Train pairs", train_pairs_path)
        config_summary("Dev evaluator", dev_pairs_path or "Not set")
        config_summary("Output model", output_dir or "Use backend default output directory")
        config_summary("Checkpoint", checkpoint_dir or "Use backend default checkpoint directory")

        train_run_presets = {
            "Smoke run": {
                "epochs": 1,
                "batch_size": 2,
                "max_seq_length": 384,
                "limit": 200,
                "checkpoint_steps": 100,
                "checkpoint_limit": 2,
                "monitor_seconds": 0,
                "resume": False,
                "use_amp": True,
            },
            "Small experiment": {
                "epochs": 1,
                "batch_size": 4,
                "max_seq_length": 384,
                "limit": 2000,
                "checkpoint_steps": 500,
                "checkpoint_limit": 2,
                "monitor_seconds": 30,
                "resume": False,
                "use_amp": True,
            },
            "Candidate full run": {
                "epochs": 1,
                "batch_size": 4,
                "max_seq_length": 384,
                "limit": 0,
                "checkpoint_steps": 1000,
                "checkpoint_limit": 2,
                "monitor_seconds": 60,
                "resume": False,
                "use_amp": True,
            },
        }
        train_run_preset = st.selectbox("Training run preset", options=list(train_run_presets), index=1)
        selected_train_run = train_run_presets[train_run_preset]
        epochs = selected_train_run["epochs"]
        batch_size = selected_train_run["batch_size"]
        max_seq_length = selected_train_run["max_seq_length"]
        limit = selected_train_run["limit"]
        checkpoint_steps = selected_train_run["checkpoint_steps"]
        checkpoint_limit = selected_train_run["checkpoint_limit"]
        monitor_seconds = selected_train_run["monitor_seconds"]
        resume = selected_train_run["resume"]
        use_amp = selected_train_run["use_amp"]
        config_summary("Training scale", f"Epochs {epochs} / Batch {batch_size} / Seq {max_seq_length} / Limit {'full' if limit == 0 else limit}")
        config_summary("Runtime", f"Checkpoint every {checkpoint_steps} steps / keep {checkpoint_limit} / monitor {monitor_seconds}s / AMP {'on' if use_amp else 'off'}")

        if st.button("Start training", use_container_width=True):
            payload = {
                "preset": None if preset == "none" else preset,
                "train_pairs": train_pairs_path or None,
                "dev_pairs": dev_pairs_path or None,
                "base_model": base_model,
                "output_dir": output_dir or None,
                "checkpoint_dir": checkpoint_dir or None,
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "limit": None if limit == 0 else int(limit),
                "max_seq_length": int(max_seq_length),
                "checkpoint_save_steps": int(checkpoint_steps),
                "checkpoint_save_total_limit": int(checkpoint_limit),
                "monitor_seconds": int(monitor_seconds),
                "resume": resume,
                "use_amp": use_amp,
            }
            try:
                with st.spinner("Training is running. This step may take a while..."):
                    data = post_json(api_base, "/api/train/biencoder", payload, timeout=1800)
            except Exception as exc:
                st.error(f"Training failed: {exc}")
            else:
                st.success("Training finished.")
                st.json(data)

