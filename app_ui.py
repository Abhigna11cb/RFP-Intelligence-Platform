"""
Streamlit Web UI for the RFP Intelligence Platform.
Bonus deliverable — Part B 6.3 "simple web UI for search and chat".

Features:
  - Search tab: hybrid search with reranking across indexed bids
  - Chat tab: Q&A with citations over bid documents
  - Extract tab: run full extraction pipeline and view JSON output
  - Eval tab: run retrieval benchmark and see results table

Run:
    streamlit run app_ui.py
"""
from __future__ import annotations

import json
import sys
import os

import streamlit as st

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RFP Intelligence Platform",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0; font-size: 1rem; }
    .result-card {
        background: #1e2a3a;
        border: 1px solid #2d4a6e;
        border-left: 4px solid #e94560;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .result-card .score { color: #64ffda; font-size: 0.85rem; font-weight: 600; }
    .result-card .source { color: #a8b2d8; font-size: 0.8rem; }
    .result-card .text { color: #ccd6f6; margin-top: 0.5rem; font-size: 0.9rem; }
    .field-row { border-bottom: 1px solid #1e2a3a; padding: 0.4rem 0; }
    .conf-high { color: #64ffda; }
    .conf-med  { color: #ffd700; }
    .conf-low  { color: #ff6b6b; }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c23152);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(233,69,96,0.4); }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📄 RFP Intelligence Platform</h1>
    <p>AI-powered search, extraction &amp; Q&amp;A over bid documents</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    bid_filter = st.selectbox(
        "Bid Scope",
        options=["All Bids", "Bid1", "Bid2"],
        index=0,
    )
    bid_id = None if bid_filter == "All Bids" else bid_filter

    top_k = st.slider("Max results", min_value=3, max_value=20, value=5)
    use_reranker = st.toggle("Enable Re-ranking", value=True,
                             help="Cross-encoder re-ranks top candidates for higher precision")

    st.markdown("---")
    st.markdown("**Indexed Bids**")
    try:
        from rfp_platform.search.tools import get_bid_documents
        for b in ["Bid1", "Bid2"]:
            docs = get_bid_documents(b)
            chunks = sum(d.get("chunk_count", 0) for d in docs)
            st.markdown(f"- **{b}**: {len(docs)} docs · {chunks} chunks")
    except Exception:
        st.warning("DB not connected")

    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("🔍 text-embedding-3-large · pgvector")
    st.markdown("🤖 GPT-4o-mini · LangGraph")
    st.markdown("🗄️ PostgreSQL 15")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_search, tab_chat, tab_extract, tab_eval = st.tabs(
    ["🔍 Search", "💬 Q&A Chat", "📊 Extract Fields", "📈 Evaluation"]
)


# ════════════════════════════════════════════════════════════════
# TAB 1: SEARCH
# ════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown("### 🔍 Hybrid Search")
    st.caption("Semantic + BM25 keyword search with optional cross-encoder re-ranking")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search query",
            placeholder="e.g. submission deadline, model number, bid bond requirement",
            label_visibility="collapsed",
        )
    with col2:
        search_btn = st.button("Search", use_container_width=True)

    doc_type_filter = st.radio(
        "Filter by doc type",
        options=["All", "rfp", "addendum", "specs", "affidavit", "bid_page"],
        index=0,
        horizontal=True,
    )
    dt = None if doc_type_filter == "All" else doc_type_filter

    if search_btn and query:
        with st.spinner("Searching..."):
            try:
                if use_reranker:
                    from rfp_platform.search.reranker import hybrid_search_reranked
                    results = hybrid_search_reranked(
                        query, bid_id=bid_id, doc_type=dt,
                        top_k=top_k, candidate_k=top_k * 4
                    )
                    st.caption(f"✅ {len(results)} results — hybrid + cross-encoder re-ranked")
                else:
                    from rfp_platform.search.tools import hybrid_search
                    results = hybrid_search(query, bid_id=bid_id, doc_type=dt, top_k=top_k)
                    st.caption(f"✅ {len(results)} results — hybrid (RRF)")

                if not results:
                    st.info("No results found. Try a different query or expand the bid scope.")
                else:
                    for r in results:
                        score_key = "rerank_score" if "rerank_score" in r else "score"
                        score = r.get(score_key, 0.0)
                        st.markdown(f"""
<div class="result-card">
    <div class="score">
        Rank #{r.get('rank', '?')} &nbsp;·&nbsp; Score: {score:.3f}
        &nbsp;·&nbsp; <span style="color:#a8b2d8">{r.get('bid_id','?')} / {r.get('doc_type','?')}</span>
    </div>
    <div class="source">📄 {r.get('file_name','?')} &nbsp; p.{r.get('page_number','?')}</div>
    <div class="text">{r.get('chunk_text','')[:400]}...</div>
</div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Search failed: {e}")


# ════════════════════════════════════════════════════════════════
# TAB 2: Q&A CHAT
# ════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### 💬 Ask Questions About Bids")
    st.caption("Answers grounded in bid documents with source citations")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Suggested questions
    suggestions = [
        "What is the submission deadline for Bid1 after all addendums?",
        "Which affidavits are required for the Dell laptop bid?",
        "What is the bid number for Bid2?",
        "Is a bid bond required for Bid1?",
        "What Dell laptop model and specs are specified in Bid2?",
    ]
    st.markdown("**Quick questions:**")
    cols = st.columns(len(suggestions))
    for col, s in zip(cols, suggestions):
        if col.button(s[:35] + "...", use_container_width=True):
            st.session_state.pending_question = s

    st.markdown("---")

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    question = st.chat_input("Ask anything about the bids...")
    if not question and "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            answer = ""
            with st.spinner("Thinking..."):
                try:
                    from rfp_platform.agents.orchestrator import run_qa
                    result = run_qa(question, bid_id=bid_id)
                    answer = result.get("answer") or ""
                except Exception as e:
                    answer = f"⚠️ Error: {e}"

            # Render OUTSIDE spinner — spinner clears its container on exit
            if answer:
                st.markdown(answer)
            else:
                st.warning("No answer generated. Try rephrasing your question.")
                answer = "No answer generated."

            st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()


# ════════════════════════════════════════════════════════════════
# TAB 3: EXTRACT FIELDS
# ════════════════════════════════════════════════════════════════
with tab_extract:
    st.markdown("### 📊 Extract All 20 Fields")
    st.caption("Run the multi-agent extraction pipeline and view structured JSON output")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        extract_bid = st.selectbox("Select bid to extract", ["Bid1", "Bid2"])
    with col_b:
        extract_btn = st.button("🚀 Run Extraction", use_container_width=True)

    if extract_btn:
        with st.spinner(f"Running extraction pipeline for {extract_bid}... (30-60 sec)"):
            try:
                from rfp_platform.agents.orchestrator import run_extraction
                result = run_extraction(extract_bid)

                val = result.get("validation", {})
                c1, c2, c3 = st.columns(3)
                c1.metric("✅ Extracted", val.get("passed", 0))
                c2.metric("❓ Not Found", val.get("not_found", 0))
                c3.metric("❌ Failed", val.get("failed", 0))

                st.markdown("#### Field Results")
                fields = result.get("fields", {})
                for field_name, data in fields.items():
                    value = data.get("value", "NOT FOUND")
                    conf = data.get("confidence", 0.0)
                    sources = data.get("sources", [])
                    src_str = f"{sources[0]['file']} p.{sources[0]['page']}" if sources else "—"

                    conf_class = "conf-high" if conf >= 0.8 else "conf-med" if conf >= 0.5 else "conf-low"
                    st.markdown(f"""
<div class="field-row">
    <b>{field_name}</b> &nbsp;
    <span class="{conf_class}">[{conf:.0%}]</span><br>
    <span style="color:#ccd6f6">{str(value)[:120] if value else '<i>Not found</i>'}</span><br>
    <span style="color:#7a8ba8; font-size:0.8rem">📄 {src_str}</span>
</div>""", unsafe_allow_html=True)

                # Addendum changes
                changes = result.get("addendum_changes", [])
                if changes:
                    st.markdown("#### Addendum Changes")
                    for c in changes:
                        st.info(
                            f"**Addendum {c.get('addendum_number','?')}** — "
                            f"**{c.get('field_name','?')}**: "
                            f"`{c.get('original_value','?')}` → `{c.get('new_value','?')}`\n\n"
                            f"_{c.get('notes','')}_"
                        )

                # Raw JSON download
                st.markdown("#### Raw JSON Output")
                json_str = json.dumps(result, indent=2, default=str)
                st.code(json_str[:3000] + ("\n..." if len(json_str) > 3000 else ""), language="json")
                st.download_button(
                    f"⬇️ Download {extract_bid}_output.json",
                    data=json_str,
                    file_name=f"{extract_bid}_output.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.exception(e)


# ════════════════════════════════════════════════════════════════
# TAB 4: EVALUATION
# ════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 📈 Retrieval Evaluation")
    st.caption("Recall@5, Recall@10, MRR across 4 retrieval configurations")

    col_run, col_info = st.columns([1, 3])
    with col_run:
        eval_btn = st.button("▶️ Run Evaluation", use_container_width=True)
    with col_info:
        st.caption("Runs 17 questions × 4 configs (vector-only, keyword-only, hybrid, hybrid+rerank). ~60 sec.")

    if eval_btn:
        with st.spinner("Running evaluation benchmark..."):
            try:
                from rfp_platform.eval.evaluate import run_evaluation, EVAL_QUESTIONS
                results = run_evaluation()

                st.markdown("#### Results Summary")
                import pandas as pd
                rows = []
                for name in ["vector_only", "keyword_only", "hybrid", "hybrid_rerank"]:
                    if name in results:
                        r = results[name]
                        rows.append({
                            "Configuration": name.replace("_", " ").title(),
                            "Recall@5": r["recall@5"],
                            "Recall@10": r["recall@10"],
                            "MRR": r["mrr"],
                        })

                df = pd.DataFrame(rows)
                st.dataframe(df.style.highlight_max(
                    subset=["Recall@5", "Recall@10", "MRR"],
                    color="#1a4a1a"
                ), use_container_width=True)

                st.markdown(f"**Questions evaluated:** {results['n_questions']}")

                # Per-question breakdown
                with st.expander("Per-question breakdown"):
                    for q in results.get("per_question", []):
                        st.markdown(f"**{q['id']}** [{q['bid_id']}]: {q['question']}")

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.exception(e)
