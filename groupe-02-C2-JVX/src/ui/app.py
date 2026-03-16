"""
FinRAG — Interface Streamlit Professionnelle
Design : fond blanc, typographie Barlow, bordures carrées, palette noir/blanc/vert.
Inspiré d'un design enterprise épuré.
"""

from __future__ import annotations

import sys
import time
import datetime
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
import plotly.graph_objects as go
from loguru import logger

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import settings

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinRAG — Intelligence Financière",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": "FinRAG — Système RAG Financier"},
)

# ─── CSS ─────────────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Barlow+Condensed:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Variables ── */
:root {
    --white:     #FFFFFF;
    --black:     #0C0C0C;
    --g50:       #FAFAFA;
    --g100:      #F5F5F5;
    --g200:      #EBEBEB;
    --g300:      #D4D4D4;
    --g400:      #A3A3A3;
    --g600:      #5C5C5C;
    --g800:      #2C2C2C;
    --green:     #16A34A;
    --green-bg:  #F0FDF4;
    --green-txt: #15803D;
    --amber:     #D97706;
    --amber-bg:  #FFFBEB;
    --red:       #DC2626;
    --red-bg:    #FEF2F2;
    --border:    #EBEBEB;
    --shadow:    0 1px 3px rgba(0,0,0,0.06);
}

/* ── Reset Streamlit ── */
html, body, [class*="css"] { font-family: 'Barlow', sans-serif !important; }
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
.reportview-container .main .block-container { padding-top: 0 !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { background: var(--white) !important; border-bottom: 1px solid var(--border); height: 0 !important; }

/* ── App background ── */
.stApp { background: var(--white) !important; color: var(--black) !important; }
.stApp > div { background: var(--white) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: var(--white) !important; border-right: 1px solid var(--border) !important; min-width: 240px !important; max-width: 240px !important; }
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; background: var(--white) !important; }
[data-testid="stSidebar"] * { color: var(--black) !important; }
[data-testid="stSidebar"] .stMarkdown { margin: 0 !important; padding: 0 !important; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 0 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 9px 18px !important;
    border: 1px solid var(--black) !important;
    background: var(--white) !important;
    color: var(--black) !important;
    transition: background 0.12s ease, color 0.12s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: var(--black) !important;
    color: var(--white) !important;
    border-color: var(--black) !important;
}
.stButton > button[kind="primary"] {
    background: var(--black) !important;
    color: var(--white) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--g800) !important;
}
.stButton > button:focus { box-shadow: none !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 0 !important;
    border: 1px solid var(--g200) !important;
    background: var(--white) !important;
    color: var(--black) !important;
    font-family: 'Barlow', sans-serif !important;
    font-size: 14px !important;
    box-shadow: none !important;
    transition: border-color 0.12s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--black) !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextInput > div > div, .stTextArea > div > div { border-radius: 0 !important; }

/* ── Select ── */
.stSelectbox > div > div {
    border-radius: 0 !important;
    border: 1px solid var(--g200) !important;
    background: var(--white) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--white) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 0 32px !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--g400) !important;
    padding: 14px 28px !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    margin-bottom: -1px !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--g800) !important; background: transparent !important; }
.stTabs [aria-selected="true"] {
    color: var(--black) !important;
    border-bottom: 2px solid var(--black) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--g50) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 20px !important;
}
[data-testid="stMetricLabel"] { font-family: 'Barlow Condensed', sans-serif !important; font-size: 10px !important; font-weight: 600 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: var(--g400) !important; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; font-size: 24px !important; font-weight: 500 !important; color: var(--black) !important; }

/* ── Expander ── */
.streamlit-expander { border: 1px solid var(--border) !important; border-radius: 0 !important; }
.streamlit-expander > div:first-child { border-radius: 0 !important; background: var(--g50) !important; border-bottom: 1px solid var(--border) !important; }

/* ── File uploader ── */
.stFileUploader > div {
    border-radius: 0 !important;
    border: 1px dashed var(--g300) !important;
    background: var(--g50) !important;
    padding: 16px !important;
}
.stFileUploader > div:hover { border-color: var(--black) !important; }

/* ── Toggle ── */
.stToggle { font-family: 'Barlow', sans-serif !important; font-size: 13px !important; }

/* ── Spinner ── */
.stSpinner > div { border-color: var(--black) transparent transparent transparent !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--g100); }
::-webkit-scrollbar-thumb { background: var(--g300); }
::-webkit-scrollbar-thumb:hover { background: var(--g400); }

/* ── Alert / warning ── */
.stAlert { border-radius: 0 !important; border-left-width: 3px !important; }

/* ── FinRAG Custom Components ── */

/* Logo / brand */
.brand {
    padding: 24px 20px 16px;
    border-bottom: 1px solid var(--border);
}
.brand-icon {
    font-size: 10px;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: var(--black);
    display: inline-block;
    border: 1.5px solid var(--black);
    padding: 2px 6px;
    margin-bottom: 8px;
}
.brand-name {
    display: block;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 17px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--black);
    line-height: 1.1;
}
.brand-sub {
    display: block;
    font-size: 10px;
    color: var(--g400);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'Barlow Condensed', sans-serif;
    margin-top: 2px;
}

/* Nav section label */
.nav-section-label {
    padding: 20px 20px 8px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--g400);
}

/* Nav item */
.nav-item {
    display: flex;
    align-items: flex-start;
    padding: 10px 20px;
    border-left: 2px solid transparent;
    cursor: pointer;
    transition: background 0.1s ease;
    text-decoration: none;
}
.nav-item:hover { background: var(--g50); }
.nav-item.active { border-left-color: var(--black); background: var(--g50); }
.nav-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--g400);
    margin-right: 12px;
    padding-top: 3px;
    min-width: 20px;
}
.nav-item.active .nav-number { color: var(--black); }
.nav-title { font-size: 13px; font-weight: 500; color: var(--g800); line-height: 1.2; }
.nav-item.active .nav-title { color: var(--black); font-weight: 600; }
.nav-subtitle { font-size: 11px; color: var(--g400); margin-top: 2px; }

/* Sidebar status card */
.status-card {
    margin: 16px 16px;
    border: 1px solid var(--border);
    background: var(--g50);
    padding: 14px;
}
.status-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.status-badge-active {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--green-bg);
    padding: 3px 8px;
    font-size: 10px;
    font-weight: 600;
    font-family: 'Barlow Condensed', sans-serif;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--green-txt);
}
.status-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--green); display: inline-block; }
.status-label { font-size: 10px; color: var(--g400); text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Barlow Condensed', sans-serif; }
.status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
.stat-cell { text-align: center; padding: 8px 0; border: 1px solid var(--border); background: var(--white); }
.stat-number { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 500; color: var(--black); display: block; }
.stat-lbl { font-size: 9px; color: var(--g400); text-transform: uppercase; letter-spacing: 0.06em; font-family: 'Barlow Condensed', sans-serif; }
.model-tag { font-size: 10px; color: var(--g400); font-family: 'JetBrains Mono', monospace; margin-top: 8px; }

/* Section divider in sidebar */
.sidebar-divider { border: none; border-top: 1px solid var(--border); margin: 12px 0; }

/* Top bar */
.top-bar {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 28px 32px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--white);
}
.view-meta {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--g400);
    margin-bottom: 6px;
}
.view-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 26px;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--black);
    margin: 0;
    text-transform: uppercase;
}
.top-bar-right { display: flex; align-items: center; gap: 20px; }
.doc-counter { text-align: right; }
.doc-count-num { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 500; color: var(--black); display: block; line-height: 1; }
.doc-count-label { font-size: 9px; color: var(--g400); text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Barlow Condensed', sans-serif; margin-top: 2px; display: block; }
.active-badge {
    display: flex;
    align-items: center;
    gap: 5px;
    background: var(--green-bg);
    border: 1px solid #BBF7D0;
    padding: 5px 10px;
}
.badge-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--green); }
.badge-text { font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--green-txt); font-family: 'Barlow Condensed', sans-serif; }

/* Chat area */
.chat-area { padding: 24px 32px; }

/* Message styles */
.msg-wrap { margin-bottom: 28px; }
.msg-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 8px; }
.msg-sender-user { font-family: 'Barlow Condensed', sans-serif; font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--black); }
.msg-sender-agent { font-family: 'Barlow Condensed', sans-serif; font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--g400); }
.msg-time { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--g400); }
.msg-user { background: var(--g100); padding: 14px 16px; border-left: 2px solid var(--black); font-size: 14px; color: var(--black); line-height: 1.5; }
.msg-agent { background: var(--g50); padding: 16px 18px; border-left: 2px solid var(--g200); font-size: 14px; color: var(--black); line-height: 1.6; }

/* Confidence badge */
.conf-high { display: inline-flex; align-items: center; gap: 4px; background: var(--green-bg); padding: 2px 8px; font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--green-txt); font-family: 'Barlow Condensed', sans-serif; }
.conf-medium { display: inline-flex; align-items: center; gap: 4px; background: var(--amber-bg); padding: 2px 8px; font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--amber); font-family: 'Barlow Condensed', sans-serif; }
.conf-low { display: inline-flex; align-items: center; gap: 4px; background: var(--red-bg); padding: 2px 8px; font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--red); font-family: 'Barlow Condensed', sans-serif; }

/* Message footer metrics */
.msg-metrics { display: flex; gap: 20px; padding: 8px 0; border-top: 1px solid var(--border); margin-top: 10px; }
.msg-metric { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--g400); }

/* Source panel */
.source-panel-header {
    padding: 20px 20px 14px;
    border-bottom: 1px solid var(--border);
}
.source-panel-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--g400);
    margin-bottom: 8px;
}
.source-filename {
    font-size: 12px;
    font-weight: 500;
    color: var(--black);
    display: flex;
    align-items: center;
    gap: 6px;
}
.source-section {
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--g400);
}
.source-excerpt {
    padding: 14px 20px;
    font-size: 12px;
    line-height: 1.7;
    color: var(--g600);
    border-left: 2px solid var(--g200);
    margin: 12px 16px;
    background: var(--g50);
}
.source-nav {
    padding: 8px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--g600);
}
.source-score-bar { margin: 12px 20px; }
.score-label { font-size: 10px; color: var(--g400); text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Barlow Condensed', sans-serif; margin-bottom: 4px; }
.score-val { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--black); }

/* Citation item */
.citation-item {
    padding: 10px 14px;
    border: 1px solid var(--border);
    margin-bottom: 6px;
    cursor: pointer;
    transition: border-color 0.1s ease;
}
.citation-item:hover { border-color: var(--black); }
.citation-source-name { font-size: 12px; font-weight: 600; color: var(--black); }
.citation-meta { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--g400); margin-top: 2px; }
.citation-excerpt { font-size: 11px; color: var(--g600); margin-top: 6px; line-height: 1.5; }

/* Input area */
.input-area {
    padding: 20px 32px;
    border-top: 1px solid var(--border);
    background: var(--white);
}
.input-hint { font-size: 11px; color: var(--g400); margin-top: 6px; font-family: 'Barlow', sans-serif; }

/* Subquery item */
.subq-item {
    padding: 7px 12px;
    border-left: 2px solid var(--g200);
    font-size: 12px;
    color: var(--g600);
    margin-bottom: 5px;
    background: var(--g50);
}
.subq-label { font-family: 'Barlow Condensed', sans-serif; font-size: 9px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--g400); display: block; margin-bottom: 3px; }

/* Empty state */
.empty-state { text-align: center; padding: 60px 20px; }
.empty-icon { font-size: 28px; margin-bottom: 12px; opacity: 0.3; }
.empty-title { font-family: 'Barlow Condensed', sans-serif; font-size: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--g400); margin-bottom: 6px; }
.empty-sub { font-size: 13px; color: var(--g400); }

/* Example queries */
.example-label { font-family: 'Barlow Condensed', sans-serif; font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--g400); margin-bottom: 10px; }
.example-grid { display: flex; flex-direction: column; gap: 6px; }

/* Upload area in sidebar */
.upload-section { padding: 12px 16px; }
.upload-label { font-family: 'Barlow Condensed', sans-serif; font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--g400); margin-bottom: 8px; display: block; }

/* Documents list */
.doc-row { display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.doc-row:hover { background: var(--g50); }
.doc-row-left { display: flex; align-items: center; gap: 10px; }
.doc-type-pill { font-family: 'Barlow Condensed', sans-serif; font-size: 9px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; padding: 2px 6px; border: 1px solid; }
.pill-pdf { color: var(--black); border-color: var(--black); }
.pill-csv { color: var(--green-txt); border-color: var(--green); }
.pill-json { color: #7C3AED; border-color: #7C3AED; }
.pill-xlsx { color: var(--amber); border-color: var(--amber); }
.pill-table { color: var(--g600); border-color: var(--g300); }
.doc-name { font-size: 13px; color: var(--black); font-weight: 500; }
.doc-chunks { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--g400); }
.doc-date { font-size: 11px; color: var(--g400); }

/* Analytics */
.analytics-section { padding: 0 32px 32px; }
.section-heading { font-family: 'Barlow Condensed', sans-serif; font-size: 12px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--g400); padding: 24px 0 12px; border-top: 1px solid var(--border); margin-top: 24px; }
.history-row { padding: 14px 16px; border: 1px solid var(--border); margin-bottom: 6px; }
.history-q { font-size: 13px; color: var(--black); margin-bottom: 6px; font-weight: 500; }
.history-meta { display: flex; gap: 16px; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--g400); }

/* Section separator in content */
.content-separator { border: none; border-top: 1px solid var(--border); margin: 0 32px; }

/* Ragas metric card */
.ragas-card { padding: 20px; border: 1px solid var(--border); text-align: center; background: var(--g50); }
.ragas-score { font-family: 'JetBrains Mono', monospace; font-size: 28px; font-weight: 500; color: var(--black); display: block; }
.ragas-label { font-family: 'Barlow Condensed', sans-serif; font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--g400); margin-top: 4px; display: block; }
.ragas-bar { height: 2px; background: var(--g200); margin-top: 12px; }
.ragas-fill { height: 100%; background: var(--black); }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "messages": [],
        "query_history": [],
        "total_tokens": 0,
        "total_queries": 0,
        "filters": {"doc_type": "Tous", "ticker": "", "date_from": None, "date_to": None},
        "use_decomposition": True,
        "last_citations": [],
        "last_sub_queries": [],
        "active_citation_idx": 0,
        "active_view": "assistant",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ─── System init ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _build_system():
    try:
        from src.retrieval.vector_store import FinancialVectorStore
        from src.retrieval.retriever import HybridFinancialRetriever
        from src.retrieval.reranker import CrossEncoderReRanker
        from src.generation.generator import FinancialAnswerGenerator
        from src.agents.rag_agent import FinancialRAGAgent

        vs = FinancialVectorStore()
        retriever = HybridFinancialRetriever(
            vector_store=vs,
            top_k=settings.TOP_K_RETRIEVAL,
            time_decay_factor=settings.TIME_DECAY_FACTOR,
        )
        reranker = CrossEncoderReRanker(top_k=settings.TOP_K_RERANK)
        generator = FinancialAnswerGenerator()
        agent = FinancialRAGAgent(vs, retriever, reranker, generator)
        return {"vs": vs, "retriever": retriever, "reranker": reranker,
                "generator": generator, "agent": agent, "ok": True}
    except Exception as e:
        logger.error(f"Init erreur: {e}")
        return {"ok": False, "error": str(e)}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    return f"{seconds*1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _conf_badge(score: float) -> str:
    if score >= 0.7:
        return f'<span class="conf-high"><span style="width:5px;height:5px;border-radius:50%;background:#16A34A;display:inline-block;"></span>{score:.0%}</span>'
    elif score >= 0.4:
        return f'<span class="conf-medium"><span style="width:5px;height:5px;border-radius:50%;background:#D97706;display:inline-block;"></span>{score:.0%}</span>'
    else:
        return f'<span class="conf-low"><span style="width:5px;height:5px;border-radius:50%;background:#DC2626;display:inline-block;"></span>{score:.0%}</span>'


def _pill(doc_type: str) -> str:
    mapping = {
        "annual_report":    ("PDF", "pill-pdf"),
        "quarterly_report": ("PDF", "pill-pdf"),
        "market_overview":  ("PDF", "pill-pdf"),
        "financial_table":  ("TABLE", "pill-table"),
        "news_article":     ("NEWS", "pill-json"),
        "csv_data":         ("CSV", "pill-csv"),
        "excel_data":       ("XLSX", "pill-xlsx"),
    }
    label, css = mapping.get(doc_type, ("DOC", "pill-table"))
    return f'<span class="doc-type-pill {css}">{label}</span>'


def _serialize_citation(c: Any) -> Dict[str, Any]:
    try:
        return asdict(c)
    except TypeError:
        return c if isinstance(c, dict) else vars(c)


def _index_samples(system: dict, samples_dir: Path):
    from src.ingestion.document_loader import FinancialDocumentLoader
    from src.ingestion.table_extractor import FinancialTableExtractor
    from src.ingestion.chunker import IntelligentFinancialChunker, ChunkingStrategy

    loader = FinancialDocumentLoader()
    all_docs = []
    for fp in samples_dir.rglob("*"):
        if fp.suffix.lower() in [".pdf", ".csv", ".json", ".xlsx", ".txt"]:
            all_docs.extend(loader.load(fp))
            if fp.suffix.lower() == ".pdf":
                all_docs.extend(FinancialTableExtractor().extract_tables_from_pdf(fp))

    if not all_docs:
        st.warning("Aucun document trouvé. Lancez : python data/generate_samples.py")
        return

    chunks = IntelligentFinancialChunker(strategy=ChunkingStrategy.HYBRID).chunk_documents(all_docs)
    added = system["vs"].add_documents(chunks, show_progress=False)
    system["retriever"].invalidate_bm25_cache()
    n = len(set(d.metadata.get("filename", "") for d in all_docs))
    st.success(f"✓ {added} chunks indexés — {n} fichiers")


def _upload_file(uploaded_file, system: dict) -> bool:
    from src.ingestion.document_loader import FinancialDocumentLoader
    from src.ingestion.table_extractor import FinancialTableExtractor
    from src.ingestion.chunker import IntelligentFinancialChunker, ChunkingStrategy

    upload_dir = ROOT_DIR / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    fp = upload_dir / uploaded_file.name
    fp.write_bytes(uploaded_file.getbuffer())

    loader = FinancialDocumentLoader()
    docs = loader.load(fp)
    if fp.suffix.lower() == ".pdf":
        docs.extend(FinancialTableExtractor().extract_tables_from_pdf(fp))
    if not docs:
        st.error(f"Aucun contenu extrait de {uploaded_file.name}")
        return False

    chunks = IntelligentFinancialChunker(strategy=ChunkingStrategy.HYBRID).chunk_documents(docs)
    added = system["vs"].add_documents(chunks, show_progress=False)
    system["retriever"].invalidate_bm25_cache()
    st.success(f"✓ {added} chunks indexés — {uploaded_file.name}")
    return True


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar(system: dict):
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="brand">
            <span class="brand-icon">RAG</span>
            <span class="brand-name">FinRAG</span>
            <span class="brand-sub">Intelligence Financière</span>
        </div>
        """, unsafe_allow_html=True)

        # Status card
        if system.get("ok"):
            stats = system["vs"].get_stats()
            chunks = stats.get("total_chunks", 0)
            sources = stats.get("total_sources", 0)
            model = stats.get("embedding_model", "").split("/")[-1]
            badge = '<span class="status-badge-active"><span class="status-dot"></span>BASE ACTIVE</span>' if chunks > 0 else '<span style="font-size:10px;color:#A3A3A3;">EN ATTENTE</span>'
            st.markdown(f"""
            <div class="status-card">
                <div class="status-row">
                    <span class="status-label">Statut</span>
                    {badge}
                </div>
                <div class="status-grid">
                    <div class="stat-cell">
                        <span class="stat-number">{chunks:,}</span>
                        <span class="stat-lbl">Chunks</span>
                    </div>
                    <div class="stat-cell">
                        <span class="stat-number">{sources}</span>
                        <span class="stat-lbl">Sources</span>
                    </div>
                </div>
                <div class="model-tag">↳ {model or "all-MiniLM-L6-v2"}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card"><span style="font-size:12px;color:#DC2626;">Système non initialisé</span></div>', unsafe_allow_html=True)

        # Navigation
        st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)

        nav_items = [
            ("01", "Hub d'Ingestion", "Sources & Upload", "upload"),
            ("02", "Base Documentaire", "Documents indexés", "documents"),
            ("03", "Assistant IA", "Conversations", "assistant"),
            ("04", "Analytics", "Métriques & Évaluation", "analytics"),
        ]

        for num, title, subtitle, view_key in nav_items:
            active = "active" if st.session_state.active_view == view_key else ""
            # Display nav item
            st.markdown(f"""
            <div class="nav-item {active}">
                <span class="nav-number">{num}</span>
                <div>
                    <div class="nav-title">{title}</div>
                    <div class="nav-subtitle">{subtitle}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # Upload
        if system.get("ok"):
            st.markdown('<span class="upload-label">Importer des documents</span>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "upload",
                type=["pdf", "csv", "xlsx", "json", "txt"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if uploaded:
                if st.button("↑ Indexer", use_container_width=True, type="primary"):
                    with st.spinner("Indexation..."):
                        for f in uploaded:
                            _upload_file(f, system)
                    st.rerun()

            # Quick index samples
            samples_dir = ROOT_DIR / "data" / "samples"
            if samples_dir.exists() and list(samples_dir.glob("*.pdf")):
                st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
                if st.button("→ Indexer les données d'exemple", use_container_width=True):
                    with st.spinner("Indexation..."):
                        _index_samples(system, samples_dir)
                    st.rerun()

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # OpenAI status
        if settings.use_openai_embeddings:
            st.markdown('<span style="font-size:10px;color:#16A34A;font-family:\'Barlow Condensed\';font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">✓ OpenAI GPT-4o actif</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="font-size:10px;color:#D97706;font-family:\'Barlow Condensed\';font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">⚠ Mode dégradé — clé manquante</span>', unsafe_allow_html=True)


# ─── Top bar ─────────────────────────────────────────────────────────────────

VIEW_META = {
    "upload":    ("VUE 01 — INGESTION",    "Hub d'Ingestion"),
    "documents": ("VUE 02 — DOCUMENTAIRE", "Base Documentaire"),
    "assistant": ("VUE 03 — INTELLIGENCE", "Agent FinRAG"),
    "analytics": ("VUE 04 — ANALYTIQUE",   "Analytics"),
}

def render_top_bar(system: dict):
    view = st.session_state.active_view
    meta, title = VIEW_META.get(view, ("", ""))
    chunks = 0
    sources = 0
    if system.get("ok"):
        stats = system["vs"].get_stats()
        chunks = stats.get("total_chunks", 0)
        sources = stats.get("total_sources", 0)
    badge_html = '<div class="active-badge"><span class="badge-dot"></span><span class="badge-text">Base Active</span></div>' if chunks > 0 else '<div style="font-size:10px;color:#A3A3A3;font-family:\'Barlow Condensed\';font-weight:600;letter-spacing:0.08em;text-transform:uppercase;">En Attente</div>'

    st.markdown(f"""
    <div class="top-bar">
        <div>
            <div class="view-meta">{meta}</div>
            <h1 class="view-title">{title}</h1>
        </div>
        <div class="top-bar-right">
            <div class="doc-counter">
                <span class="doc-count-num">{sources}</span>
                <span class="doc-count-label">Documents indexés</span>
            </div>
            {badge_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Tab navigation ───────────────────────────────────────────────────────────

def render_tab_nav():
    """Tab bar styled as numbered navigation."""
    tabs_def = [
        ("01 — Ingestion",    "upload"),
        ("02 — Documents",    "documents"),
        ("03 — Assistant",    "assistant"),
        ("04 — Analytics",    "analytics"),
    ]
    # We use st.tabs but the logic switches via session_state.active_view
    tab_objects = st.tabs([label for label, _ in tabs_def])
    return tab_objects, [view for _, view in tabs_def]


# ─── Source panel ─────────────────────────────────────────────────────────────

def render_source_panel():
    citations = st.session_state.last_citations
    if not citations:
        st.markdown("""
        <div style="padding: 32px 20px; text-align: center;">
            <div style="font-size:24px; margin-bottom:10px; opacity:0.2;">▪</div>
            <div style="font-family:'Barlow Condensed';font-size:11px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#A3A3A3;">Source documentaire</div>
            <div style="font-size:12px;color:#A3A3A3;margin-top:6px;">Posez une question pour voir les sources utilisées</div>
        </div>
        """, unsafe_allow_html=True)
        return

    active_idx = min(st.session_state.active_citation_idx, len(citations) - 1)
    cit = citations[active_idx]

    source = cit.get("source_file", "unknown")
    page = cit.get("page_number")
    date = cit.get("date", "")
    excerpt = cit.get("excerpt", "")
    score = cit.get("relevance_score", 0)

    page_str = f"p. {page}" if page else ""
    n = len(citations)

    st.markdown(f"""
    <div style="border-left: 1px solid #EBEBEB; height: 100%;">
        <div class="source-panel-header">
            <div class="source-panel-label">Source documentaire</div>
            <div class="source-filename">▪ {source}</div>
        </div>
        <div class="source-section">Extrait pertinent</div>
        <div style="font-family:'JetBrains Mono';font-size:10px;color:#A3A3A3;padding:6px 20px 0;display:flex;justify-content:space-between;">
            <span>{page_str}{" · " + date if date else ""}</span>
            <span>{active_idx + 1} / {n}</span>
        </div>
        <div class="source-excerpt">{excerpt}</div>
        <div class="source-score-bar">
            <div class="score-label">Score de pertinence</div>
            <div class="score-val">{score:.3f}</div>
            <div style="height:2px;background:#EBEBEB;margin-top:6px;">
                <div style="height:100%;background:#0C0C0C;width:{min(score, 1)*100:.0f}%;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if n > 1:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("← Préc.", key="src_prev", use_container_width=True,
                         disabled=active_idx == 0):
                st.session_state.active_citation_idx = max(0, active_idx - 1)
                st.rerun()
        with c2:
            if st.button("Suiv. →", key="src_next", use_container_width=True,
                         disabled=active_idx >= n - 1):
                st.session_state.active_citation_idx = min(n - 1, active_idx + 1)
                st.rerun()


# ─── Intelligence panel (sub-queries + stats) ────────────────────────────────

def render_intel_sidebar():
    sqs = st.session_state.last_sub_queries
    if sqs:
        st.markdown("""
        <div style="padding: 16px 20px 8px;">
            <div style="font-family:'Barlow Condensed';font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#A3A3A3;margin-bottom:8px;">Requêtes décomposées</div>
        </div>
        """, unsafe_allow_html=True)
        for sq in sqs:
            st.markdown(f'<div class="subq-item"><span class="subq-label">Sous-requête</span>{sq[:80]}{"…" if len(sq) > 80 else ""}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding: 12px 20px; border-top: 1px solid #EBEBEB; margin-top: 12px;">
        <div style="font-family:'Barlow Condensed';font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#A3A3A3;margin-bottom:8px;">Session</div>
        <div style="font-family:'JetBrains Mono';font-size:11px;color:#5C5C5C;">
            {st.session_state.total_queries} requête{"s" if st.session_state.total_queries != 1 else ""}<br>
            {st.session_state.total_tokens:,} tokens
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Message rendering ────────────────────────────────────────────────────────

def _render_message(msg: dict):
    role = msg.get("role", "user")
    content = msg.get("content", "")
    meta = msg.get("meta", {})
    ts = msg.get("ts", "")

    if role == "user":
        st.markdown(f"""
        <div class="msg-wrap">
            <div class="msg-header">
                <span class="msg-sender-user">Utilisateur</span>
                <span class="msg-time">{ts}</span>
            </div>
            <div class="msg-user">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        conf = meta.get("confidence_score", 0)
        proc = meta.get("processing_time", 0)
        tokens = meta.get("tokens_used", 0)
        n_src = meta.get("context_docs_count", 0)
        conf_b = _conf_badge(conf)

        st.markdown(f"""
        <div class="msg-wrap">
            <div class="msg-header">
                <span class="msg-sender-agent">Agent IA</span>
                <div style="display:flex;align-items:center;gap:10px;">
                    {conf_b}
                    <span class="msg-time">{ts}</span>
                </div>
            </div>
            <div class="msg-agent">
        """, unsafe_allow_html=True)

        st.markdown(content)

        st.markdown(f"""
            <div class="msg-metrics">
                <span class="msg-metric">⏱ {_fmt_time(proc)}</span>
                <span class="msg-metric">◆ {tokens:,} tokens</span>
                <span class="msg-metric">▪ {n_src} sources</span>
            </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Query processing ─────────────────────────────────────────────────────────

def _process_query(query: str, system: dict):
    ts = _now()
    st.session_state.messages.append({"role": "user", "content": query, "ts": ts})

    if not system.get("ok"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Système non initialisé.",
            "meta": {}, "ts": _now(),
        })
        return

    f = st.session_state.filters
    date_range = (f["date_from"], f["date_to"]) if f.get("date_from") and f.get("date_to") else None
    doc_type_map = {
        "Rapports": ["annual_report", "quarterly_report", "market_overview"],
        "News": ["news_article"],
        "Tableaux": ["financial_table"],
        "CSV": ["csv_data"],
    }
    document_type = doc_type_map.get(f.get("doc_type", "Tous"))
    ticker = f.get("ticker") or None

    answer = system["agent"].answer(
        question=query,
        use_decomposition=st.session_state.use_decomposition,
        date_range=date_range,
        document_type=document_type,
        ticker=ticker,
    )

    cits = [_serialize_citation(c) for c in answer.citations]
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer.answer,
        "meta": {
            "confidence_score": answer.confidence_score,
            "processing_time": answer.processing_time,
            "tokens_used": answer.tokens_used,
            "context_docs_count": answer.context_docs_count,
            "citations": cits,
            "sub_queries": answer.sub_queries,
        },
        "ts": _now(),
    })

    st.session_state.last_citations = cits
    st.session_state.last_sub_queries = answer.sub_queries
    st.session_state.active_citation_idx = 0
    st.session_state.total_queries += 1
    st.session_state.total_tokens += answer.tokens_used
    st.session_state.query_history.append({
        "question": query,
        "time": time.time(),
        "confidence": answer.confidence_score,
        "processing_time": answer.processing_time,
        "tokens": answer.tokens_used,
        "n_sources": answer.context_docs_count,
    })


# ─── Views ────────────────────────────────────────────────────────────────────

def view_upload(system: dict):
    st.markdown('<div style="padding: 32px 32px 0;">', unsafe_allow_html=True)

    # Filters section
    st.markdown("""
    <div style="font-family:'Barlow Condensed';font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#A3A3A3;margin-bottom:16px;">
        Filtres de recherche
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        doc_t = st.selectbox("Type", ["Tous", "Rapports", "News", "Tableaux", "CSV"], label_visibility="collapsed")
    with c2:
        ticker = st.text_input("Ticker", placeholder="AAPL, MSFT…", label_visibility="collapsed")
    with c3:
        yf = st.text_input("Année début", placeholder="2022", label_visibility="collapsed")
    with c4:
        yt = st.text_input("Année fin", placeholder="2024", label_visibility="collapsed")

    st.session_state.filters = {
        "doc_type": doc_t,
        "ticker": ticker.strip().upper() if ticker else "",
        "date_from": yf.strip() if yf else None,
        "date_to": yt.strip() if yt else None,
    }

    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

    # Options
    st.markdown('<div style="font-family:\'Barlow Condensed\';font-size:11px;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#A3A3A3;margin-bottom:16px;">Options RAG</div>', unsafe_allow_html=True)

    st.session_state.use_decomposition = st.toggle(
        "Décomposition intelligente des requêtes",
        value=st.session_state.use_decomposition,
        help="Décompose les questions complexes en sous-requêtes atomiques",
    )

    st.markdown('</div>', unsafe_allow_html=True)


def view_documents(system: dict):
    if not system.get("ok"):
        st.warning("Système non initialisé")
        return

    sources = system["vs"].list_sources()

    st.markdown('<div style="padding: 0 32px;">', unsafe_allow_html=True)

    if not sources:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">▪</div>
            <div class="empty-title">Base documentaire vide</div>
            <div class="empty-sub">Importez des documents via la sidebar ou indexez les données d'exemple.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        return

    search = st.text_input("", placeholder="Filtrer par nom de fichier…", label_visibility="collapsed")
    if search:
        sources = [s for s in sources if search.lower() in s.get("filename", "").lower()]

    st.markdown(f'<div style="font-family:\'Barlow Condensed\';font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#A3A3A3;padding: 12px 0 8px;">{len(sources)} document{"s" if len(sources) != 1 else ""}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Document rows
    for src in sources:
        fname = src.get("filename", "unknown")
        dtype = src.get("document_type", "unknown")
        count = src.get("chunk_count", 0)
        date = src.get("date", "")
        coll = src.get("collection", "unknown")
        pill = _pill(dtype)

        c1, c2, c3, c4, c5 = st.columns([4, 2, 1, 1, 1])
        with c1:
            st.markdown(f'<div style="padding:12px 32px;font-size:13px;font-weight:500;color:#0C0C0C;">{pill} {fname}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div style="padding:12px 0;font-size:12px;color:#5C5C5C;">{dtype.replace("_", " ").title()}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div style="padding:12px 0;font-family:\'JetBrains Mono\';font-size:11px;color:#A3A3A3;">{count}</div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div style="padding:12px 0;font-size:11px;color:#A3A3A3;">{date}</div>', unsafe_allow_html=True)
        with c5:
            if st.button("✕", key=f"del_{fname}_{coll}", help=f"Supprimer {fname}"):
                deleted = system["vs"].delete_by_source(fname)
                if deleted > 0:
                    system["retriever"].invalidate_bm25_cache()
                    st.rerun()
        st.markdown('<hr style="border:none;border-top:1px solid #EBEBEB;margin:0 32px;">', unsafe_allow_html=True)


def view_assistant(system: dict):
    col_chat, col_src = st.columns([3, 1], gap="small")

    # ── Chat column ──────────────────────────────────────────────────────────
    with col_chat:
        # Messages
        if st.session_state.messages:
            st.markdown('<div class="chat-area">', unsafe_allow_html=True)
            for msg in st.session_state.messages:
                _render_message(msg)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Empty state + examples
            st.markdown("""
            <div style="padding: 48px 32px 24px;">
                <div class="empty-state" style="padding: 40px 0 32px;">
                    <div style="font-family:'Barlow Condensed';font-size:10px;font-weight:600;letter-spacing:0.15em;text-transform:uppercase;color:#A3A3A3;margin-bottom:6px;">Agent FinRAG — Intelligence Artificielle</div>
                    <div style="font-size:15px;color:#5C5C5C;max-width:400px;margin:0 auto;line-height:1.5;">
                        Posez vos questions sur les documents financiers indexés. Les réponses sont générées par GPT-4o et sourcées documentairement.
                    </div>
                </div>
                <div class="example-label">Questions d'exemple</div>
            </div>
            """, unsafe_allow_html=True)

            examples = [
                "Quel est le chiffre d'affaires d'Apple en FY2023 ?",
                "Comparez la croissance d'Apple et Microsoft sur leurs derniers exercices",
                "Quelle est la marge opérationnelle de Microsoft au T4 FY2024 ?",
                "Quel est l'impact de l'IA sur les revenus de Microsoft en 2024 ?",
            ]
            st.markdown('<div style="padding: 0 32px;">', unsafe_allow_html=True)
            ec1, ec2 = st.columns(2)
            for i, ex in enumerate(examples):
                with (ec1 if i % 2 == 0 else ec2):
                    if st.button(ex, key=f"ex_{i}", use_container_width=True):
                        _process_query(ex, system)
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Input area
        st.markdown('<div class="content-separator"></div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="input-area">', unsafe_allow_html=True)
            query = st.text_area(
                "question",
                placeholder="Posez votre question financière…",
                height=72,
                label_visibility="collapsed",
                key="chat_input",
            )
            c_send, c_clear = st.columns([4, 1])
            with c_send:
                if st.button("Envoyer →", use_container_width=True, type="primary"):
                    if query and query.strip():
                        _process_query(query.strip(), system)
                        st.rerun()
                    else:
                        st.warning("Saisissez une question avant d'envoyer.")
            with c_clear:
                if st.button("Effacer", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.last_citations = []
                    st.session_state.last_sub_queries = []
                    st.rerun()
            st.markdown('<div class="input-hint">GPT-4o • Réponses sourcées documentairement</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Source panel ─────────────────────────────────────────────────────────
    with col_src:
        render_source_panel()
        st.markdown('<hr class="sidebar-divider" style="margin: 16px 0;">', unsafe_allow_html=True)
        render_intel_sidebar()


def view_analytics(system: dict):
    if not system.get("ok"):
        st.warning("Système non initialisé")
        return

    stats = system["vs"].get_stats()

    st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="margin-top:0;padding-top:24px;border-top:none;">Statistiques du vector store</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Chunks indexés", f"{stats.get('total_chunks', 0):,}")
    with c2:
        st.metric("Sources", stats.get("total_sources", 0))
    with c3:
        st.metric("Requêtes (session)", st.session_state.total_queries)
    with c4:
        st.metric("Tokens utilisés", f"{st.session_state.total_tokens:,}")

    # Collections breakdown
    colls = stats.get("collections", {})
    if any(v > 0 for v in colls.values()):
        st.markdown('<div class="section-heading">Collections ChromaDB</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=list(colls.values()),
            y=list(colls.keys()),
            orientation='h',
            marker_color=['#0C0C0C', '#5C5C5C', '#A3A3A3', '#D4D4D4'][:len(colls)],
        ))
        fig.update_layout(
            height=180,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, tickfont=dict(family='JetBrains Mono', size=11)),
            yaxis=dict(showgrid=False, tickfont=dict(family='Barlow Condensed', size=12, color='#5C5C5C')),
            font=dict(family='Barlow', color='#0C0C0C'),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Query history
    if st.session_state.query_history:
        st.markdown('<div class="section-heading">Historique des requêtes</div>', unsafe_allow_html=True)
        for q in st.session_state.query_history[-10:][::-1]:
            conf = q.get("confidence", 0)
            conf_color = "#16A34A" if conf >= 0.7 else "#D97706" if conf >= 0.4 else "#DC2626"
            st.markdown(f"""
            <div class="history-row">
                <div class="history-q">{q['question'][:100]}{"…" if len(q['question']) > 100 else ""}</div>
                <div class="history-meta">
                    <span style="color:{conf_color};">● {conf:.0%}</span>
                    <span>⏱ {_fmt_time(q.get('processing_time', 0))}</span>
                    <span>◆ {q.get('tokens', 0):,} tokens</span>
                    <span>▪ {q.get('n_sources', 0)} sources</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # RAGAS demo metrics
    st.markdown('<div class="section-heading">Métriques RAGAS (démonstration)</div>', unsafe_allow_html=True)
    st.caption("Lancez `python src/main.py evaluate` pour obtenir des métriques réelles")

    demo = {"Faithfulness": 0.87, "Answer Relevancy": 0.82, "Context Recall": 0.78, "Context Precision": 0.84}
    cols_ragas = st.columns(4)
    for i, (name, val) in enumerate(demo.items()):
        with cols_ragas[i]:
            st.markdown(f"""
            <div class="ragas-card">
                <span class="ragas-score">{val*100:.0f}%</span>
                <span class="ragas-label">{name}</span>
                <div class="ragas-bar">
                    <div class="ragas-fill" style="width:{val*100:.0f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # RAGAS report if available
    rpt = ROOT_DIR / "docs" / "evaluation_report.md"
    if rpt.exists():
        st.markdown('<div class="section-heading">Rapport d\'évaluation</div>', unsafe_allow_html=True)
        with st.expander("Afficher le rapport complet", expanded=False):
            st.markdown(rpt.read_text(encoding="utf-8"))

    st.markdown('</div>', unsafe_allow_html=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    with st.spinner("Initialisation du système…"):
        system = _build_system()

    if not system.get("ok"):
        st.error(f"Erreur d'initialisation : {system.get('error', 'Erreur inconnue')}")
        st.code("pip install -r requirements.txt\nstreamlit run src/ui/app.py")
        return

    render_sidebar(system)

    # Tab navigation
    tab_labels = [
        "01 — Ingestion",
        "02 — Documents",
        "03 — Assistant",
        "04 — Analytics",
    ]
    tab_views = ["upload", "documents", "assistant", "analytics"]

    tabs = st.tabs(tab_labels)

    for tab, view_key in zip(tabs, tab_views):
        with tab:
            st.session_state.active_view = view_key
            render_top_bar(system)

            if view_key == "upload":
                view_upload(system)
            elif view_key == "documents":
                view_documents(system)
            elif view_key == "assistant":
                view_assistant(system)
            elif view_key == "analytics":
                view_analytics(system)


if __name__ == "__main__":
    main()