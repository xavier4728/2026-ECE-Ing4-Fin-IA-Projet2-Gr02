"""
FinRAG Analytics — Interface Streamlit Premium
Système RAG pour l'analyse de documents financiers.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import settings

# ─── Page Config (MUST be first Streamlit call) ──────────────────────────────

st.set_page_config(
    page_title="FinRAG Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "FinRAG Analytics — Système RAG Financier",
    },
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* CSS Variables */
:root {
    --bg-primary: #0A0E1A;
    --bg-secondary: #111827;
    --bg-card: #1A2236;
    --accent-gold: #F0B429;
    --accent-blue: #3B82F6;
    --accent-green: #10B981;
    --accent-red: #EF4444;
    --text-primary: #F9FAFB;
    --text-secondary: #9CA3AF;
    --border: #1F2D45;
}

/* Main app background */
.stApp {
    background: linear-gradient(135deg, #0A0E1A 0%, #0D1321 50%, #0A0E1A 100%);
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-primary);
}

/* Mesh gradient background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 80%, rgba(240, 180, 41, 0.05) 0%, transparent 60%),
        radial-gradient(ellipse at 50% 50%, rgba(16, 185, 129, 0.03) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1321 0%, #111827 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Cards */
.finrag-card {
    background: rgba(26, 34, 54, 0.8);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(31, 45, 69, 0.6);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* Chat messages */
.chat-user {
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 12px 12px 2px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px 40px;
    color: var(--text-primary);
    font-size: 0.95rem;
}

.chat-assistant {
    background: rgba(26, 34, 54, 0.9);
    border: 1px solid var(--border);
    border-radius: 2px 12px 12px 12px;
    padding: 16px 20px;
    margin: 8px 40px 8px 0;
    color: var(--text-primary);
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Confidence badges */
.badge-high { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }
.badge-medium { background: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.4); border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }
.badge-low { background: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 20px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }

/* Document badges */
.doc-badge-pdf { background: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 4px; padding: 1px 6px; font-size: 0.7rem; font-weight: 600; }
.doc-badge-csv { background: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 4px; padding: 1px 6px; font-size: 0.7rem; font-weight: 600; }
.doc-badge-news { background: rgba(59, 130, 246, 0.2); color: #3B82F6; border: 1px solid rgba(59, 130, 246, 0.4); border-radius: 4px; padding: 1px 6px; font-size: 0.7rem; font-weight: 600; }
.doc-badge-json { background: rgba(168, 85, 247, 0.2); color: #A855F7; border: 1px solid rgba(168, 85, 247, 0.4); border-radius: 4px; padding: 1px 6px; font-size: 0.7rem; font-weight: 600; }
.doc-badge-xlsx { background: rgba(34, 197, 94, 0.2); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 4px; padding: 1px 6px; font-size: 0.7rem; font-weight: 600; }

/* Headers */
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

/* Metrics */
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--accent-gold);
}

.metric-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Gold accent title */
.finrag-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: var(--accent-gold);
    letter-spacing: 0.02em;
}

/* Citation box */
.citation-box {
    background: rgba(59, 130, 246, 0.08);
    border-left: 3px solid var(--accent-blue);
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    margin: 4px 0;
    font-size: 0.82rem;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
}

/* Sub-query item */
.subquery-item {
    background: rgba(16, 185, 129, 0.08);
    border-left: 2px solid var(--accent-green);
    border-radius: 0 4px 4px 0;
    padding: 4px 10px;
    margin: 3px 0;
    font-size: 0.82rem;
    color: #86efac;
}

/* Status indicator */
.status-dot-green { width: 8px; height: 8px; background: #10B981; border-radius: 50%; display: inline-block; margin-right: 6px; box-shadow: 0 0 6px #10B981; }
.status-dot-yellow { width: 8px; height: 8px; background: #F59E0B; border-radius: 50%; display: inline-block; margin-right: 6px; }
.status-dot-red { width: 8px; height: 8px; background: #EF4444; border-radius: 50%; display: inline-block; margin-right: 6px; }

/* Streamlit widget overrides */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(26, 34, 54, 0.8) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

.stSelectbox > div > div {
    background: rgba(26, 34, 54, 0.8) !important;
    border: 1px solid var(--border) !important;
}

.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(17, 24, 39, 0.8);
    border-radius: 10px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    border-radius: 6px;
}

.stTabs [aria-selected="true"] {
    background: rgba(59, 130, 246, 0.2) !important;
    color: var(--accent-blue) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2d4060; }

/* Loading spinner */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.loading-pulse { animation: pulse 1.5s ease-in-out infinite; }

/* Divider */
.finrag-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 16px 0;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── Session State Init ───────────────────────────────────────────────────────

def init_session_state():
    """Initialise l'état de session Streamlit."""
    defaults = {
        "messages": [],
        "vector_store": None,
        "retriever": None,
        "reranker": None,
        "agent": None,
        "generator": None,
        "indexed_sources": [],
        "query_history": [],
        "total_tokens_used": 0,
        "total_queries": 0,
        "system_initialized": False,
        "active_tab": "chat",
        "filters": {
            "date_from": None,
            "date_to": None,
            "doc_type": "Tous",
            "ticker": "",
        },
        "use_decomposition": True,
        "last_sub_queries": [],
        "last_sources": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ─── System Initialization ────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def initialize_rag_system():
    """Initialise le système RAG (mis en cache pour la session)."""
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
        agent = FinancialRAGAgent(
            vector_store=vs,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
        )

        return {
            "vector_store": vs,
            "retriever": retriever,
            "reranker": reranker,
            "generator": generator,
            "agent": agent,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Erreur initialisation système: {e}")
        return {"success": False, "error": str(e)}


# ─── Helper Functions ─────────────────────────────────────────────────────────

def get_confidence_badge(score: float) -> str:
    """Retourne le HTML du badge de confiance."""
    if score >= 0.7:
        return f'<span class="badge-high">✓ Confiance: {score:.0%}</span>'
    elif score >= 0.4:
        return f'<span class="badge-medium">~ Confiance: {score:.0%}</span>'
    else:
        return f'<span class="badge-low">⚠ Confiance: {score:.0%}</span>'


def get_doc_badge(doc_type: str) -> str:
    """Retourne le HTML du badge de type de document."""
    badges = {
        "annual_report": ('<span class="doc-badge-pdf">PDF</span>', "📄"),
        "quarterly_report": ('<span class="doc-badge-pdf">PDF</span>', "📊"),
        "news_article": ('<span class="doc-badge-news">NEWS</span>', "📰"),
        "financial_table": ('<span class="doc-badge-pdf">TABLE</span>', "📋"),
        "csv_data": ('<span class="doc-badge-csv">CSV</span>', "📈"),
        "excel_data": ('<span class="doc-badge-xlsx">XLSX</span>', "📊"),
        "market_overview": ('<span class="doc-badge-pdf">PDF</span>', "🌐"),
    }
    badge, icon = badges.get(doc_type, ('<span class="doc-badge-json">DOC</span>', "📁"))
    return badge, icon


def format_processing_time(seconds: float) -> str:
    """Formate le temps de traitement."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.1f}s"


def upload_and_index_file(uploaded_file, system: dict) -> bool:
    """Sauvegarde et indexe un fichier uploadé."""
    try:
        from src.ingestion.document_loader import FinancialDocumentLoader
        from src.ingestion.table_extractor import FinancialTableExtractor
        from src.ingestion.chunker import IntelligentFinancialChunker, ChunkingStrategy

        # Save uploaded file
        upload_dir = ROOT_DIR / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / uploaded_file.name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load and chunk
        loader = FinancialDocumentLoader()
        docs = loader.load(file_path)

        # Extract tables from PDFs
        if file_path.suffix.lower() == ".pdf":
            extractor = FinancialTableExtractor()
            table_docs = extractor.extract_tables_from_pdf(file_path)
            docs.extend(table_docs)

        if not docs:
            st.error(f"Aucun contenu extrait de {uploaded_file.name}")
            return False

        # Chunk
        chunker = IntelligentFinancialChunker(strategy=ChunkingStrategy.HYBRID)
        chunks = chunker.chunk_documents(docs)

        # Index
        vector_store = system["vector_store"]
        added = vector_store.add_documents(chunks, show_progress=False)

        # Invalidate BM25 cache
        system["retriever"].invalidate_bm25_cache()

        st.success(f"✅ {added} chunks indexés depuis {uploaded_file.name}")
        return True

    except Exception as e:
        st.error(f"Erreur d'indexation: {e}")
        logger.error(f"Upload error: {e}")
        return False


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(system: dict):
    """Rendu de la sidebar."""
    with st.sidebar:
        # Logo & Title
        st.markdown("""
        <div style="text-align: center; padding: 20px 0 10px;">
            <div style="font-size: 2.5rem; margin-bottom: 8px;">📊</div>
            <div class="finrag-title">FinRAG Analytics</div>
            <div style="color: #9CA3AF; font-size: 0.75rem; margin-top: 4px;">
                Système RAG Financier
            </div>
        </div>
        <hr class="finrag-divider">
        """, unsafe_allow_html=True)

        # System Status
        if system.get("success"):
            stats = system["vector_store"].get_stats()
            total_chunks = stats.get("total_chunks", 0)
            total_sources = stats.get("total_sources", 0)

            status_color = "green" if total_chunks > 0 else "yellow"
            status_text = "Opérationnel" if total_chunks > 0 else "En attente"

            st.markdown(f"""
            <div class="finrag-card">
                <div style="font-size: 0.75rem; color: #9CA3AF; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;">Statut système</div>
                <div><span class="status-dot-{status_color}"></span><span style="font-size: 0.85rem;">{status_text}</span></div>
                <div style="margin-top: 10px; display: flex; justify-content: space-between;">
                    <div>
                        <div style="font-family: 'JetBrains Mono'; font-size: 1.1rem; color: #F0B429;">{total_chunks:,}</div>
                        <div style="font-size: 0.7rem; color: #9CA3AF;">chunks</div>
                    </div>
                    <div>
                        <div style="font-family: 'JetBrains Mono'; font-size: 1.1rem; color: #3B82F6;">{total_sources}</div>
                        <div style="font-size: 0.7rem; color: #9CA3AF;">sources</div>
                    </div>
                    <div>
                        <div style="font-family: 'JetBrains Mono'; font-size: 1.1rem; color: #10B981;">{st.session_state.total_queries}</div>
                        <div style="font-size: 0.7rem; color: #9CA3AF;">requêtes</div>
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 0.7rem; color: #6B7280;">
                    Modèle: {stats.get('embedding_model', 'N/A').split('/')[-1]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="finrag-card">
                <span class="status-dot-red"></span>
                <span style="font-size: 0.85rem; color: #EF4444;">Système non initialisé</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="finrag-divider">', unsafe_allow_html=True)

        # Document Upload
        st.markdown('<div style="font-size: 0.8rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">📁 Importer des documents</div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Glissez vos fichiers ici",
            type=["pdf", "csv", "xlsx", "json", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files and system.get("success"):
            if st.button("⚡ Indexer les documents", use_container_width=True, type="primary"):
                with st.spinner("Indexation en cours..."):
                    for f in uploaded_files:
                        upload_and_index_file(f, system)
                st.rerun()

        st.markdown('<hr class="finrag-divider">', unsafe_allow_html=True)

        # Indexed Documents List
        if system.get("success"):
            sources = system["vector_store"].list_sources()
            if sources:
                st.markdown(f'<div style="font-size: 0.8rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">📚 Documents indexés ({len(sources)})</div>', unsafe_allow_html=True)

                for src in sources[:10]:
                    filename = src.get("filename", "unknown")
                    doc_type = src.get("document_type", "unknown")
                    chunk_count = src.get("chunk_count", 0)
                    badge, icon = get_doc_badge(doc_type)

                    ext = Path(filename).suffix.lower()
                    ext_badges = {
                        ".pdf": '<span class="doc-badge-pdf">PDF</span>',
                        ".csv": '<span class="doc-badge-csv">CSV</span>',
                        ".json": '<span class="doc-badge-json">JSON</span>',
                        ".xlsx": '<span class="doc-badge-xlsx">XLSX</span>',
                    }
                    type_badge = ext_badges.get(ext, badge)

                    st.markdown(f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 8px; margin-bottom: 4px; border-radius: 6px; background: rgba(31, 45, 69, 0.3);">
                        <div style="display: flex; align-items: center; gap: 6px; flex: 1; min-width: 0;">
                            <span>{icon}</span>
                            <span style="font-size: 0.78rem; color: #D1D5DB; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{filename}">{filename[:25]}{'...' if len(filename) > 25 else ''}</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 4px; flex-shrink: 0;">
                            {type_badge}
                            <span style="font-size: 0.7rem; color: #6B7280;">{chunk_count}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if len(sources) > 10:
                    st.caption(f"... et {len(sources) - 10} autres sources")

        st.markdown('<hr class="finrag-divider">', unsafe_allow_html=True)

        # Filters
        st.markdown('<div style="font-size: 0.8rem; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">🔍 Filtres de recherche</div>', unsafe_allow_html=True)

        doc_type_options = ["Tous", "Rapports", "News", "Tableaux", "CSV"]
        selected_type = st.selectbox(
            "Type de document",
            doc_type_options,
            label_visibility="collapsed",
        )

        ticker_filter = st.text_input(
            "Ticker (ex: AAPL)",
            placeholder="AAPL, MSFT...",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns(2)
        with col1:
            year_from = st.text_input("Année début", placeholder="2022", label_visibility="collapsed")
        with col2:
            year_to = st.text_input("Année fin", placeholder="2024", label_visibility="collapsed")

        st.session_state.filters = {
            "doc_type": selected_type,
            "ticker": ticker_filter.strip().upper() if ticker_filter else "",
            "date_from": year_from.strip() if year_from else None,
            "date_to": year_to.strip() if year_to else None,
        }

        st.markdown('<hr class="finrag-divider">', unsafe_allow_html=True)

        # Options
        st.session_state.use_decomposition = st.toggle(
            "🧠 Décomposition intelligente",
            value=st.session_state.use_decomposition,
            help="Décompose les questions complexes en sous-requêtes atomiques",
        )

        # Quick Index Samples
        if system.get("success"):
            samples_dir = ROOT_DIR / "data" / "samples"
            if samples_dir.exists():
                sample_files = list(samples_dir.glob("*.pdf")) + list(samples_dir.glob("*.csv"))
                if sample_files:
                    st.markdown('<hr class="finrag-divider">', unsafe_allow_html=True)
                    if st.button("🚀 Indexer les données d'exemple", use_container_width=True):
                        with st.spinner("Indexation des données d'exemple..."):
                            _index_sample_data(system, samples_dir)
                        st.rerun()


def _index_sample_data(system: dict, samples_dir: Path):
    """Indexe les données d'exemple."""
    try:
        from src.ingestion.document_loader import FinancialDocumentLoader
        from src.ingestion.table_extractor import FinancialTableExtractor
        from src.ingestion.chunker import IntelligentFinancialChunker, ChunkingStrategy

        loader = FinancialDocumentLoader()
        all_docs = []

        # Load all sample files
        for file_path in samples_dir.rglob("*"):
            if file_path.suffix.lower() in [".pdf", ".csv", ".json", ".xlsx", ".txt"]:
                docs = loader.load(file_path)
                all_docs.extend(docs)

                # Tables from PDFs
                if file_path.suffix.lower() == ".pdf":
                    extractor = FinancialTableExtractor()
                    table_docs = extractor.extract_tables_from_pdf(file_path)
                    all_docs.extend(table_docs)

        if not all_docs:
            st.warning("Aucun document trouvé dans data/samples/. Lancez d'abord: python data/generate_samples.py")
            return

        # Chunk
        chunker = IntelligentFinancialChunker(strategy=ChunkingStrategy.HYBRID)
        chunks = chunker.chunk_documents(all_docs)

        # Index
        added = system["vector_store"].add_documents(chunks, show_progress=False)
        system["retriever"].invalidate_bm25_cache()

        st.success(f"✅ {added} chunks indexés depuis {len(set(d.metadata.get('filename','') for d in all_docs))} fichiers")

    except Exception as e:
        st.error(f"Erreur: {e}")
        logger.error(f"Sample indexing error: {e}")


# ─── Chat Interface ───────────────────────────────────────────────────────────

def render_chat(system: dict):
    """Rendu de l'interface de chat principale."""
    # Main layout: 75% chat + 25% intelligence panel
    col_chat, col_intel = st.columns([3, 1])

    with col_chat:
        st.markdown("""
        <h2 style="font-family: 'DM Serif Display'; color: #F9FAFB; margin-bottom: 4px;">
            💬 Assistant Financier
        </h2>
        <p style="color: #9CA3AF; font-size: 0.85rem; margin-bottom: 20px;">
            Posez vos questions sur les documents financiers indexés
        </p>
        """, unsafe_allow_html=True)

        # Chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.messages:
                _render_message(msg)

        # Query input
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        # Example queries
        if not st.session_state.messages:
            st.markdown('<div style="color: #9CA3AF; font-size: 0.8rem; margin-bottom: 8px;">💡 Questions d\'exemple :</div>', unsafe_allow_html=True)
            example_cols = st.columns(2)
            examples = [
                "Quel est le CA d'Apple en FY2023 ?",
                "Comparez la croissance d'Apple et Microsoft",
                "Quelle est la marge opérationnelle de Microsoft en FY2024 ?",
                "Quel est l'impact de l'IA sur les revenus de Microsoft ?",
            ]
            for i, ex in enumerate(examples):
                with example_cols[i % 2]:
                    if st.button(f"→ {ex}", key=f"ex_{i}", use_container_width=True):
                        _process_query(ex, system)
                        st.rerun()

        # Input area
        with st.form(key="chat_form", clear_on_submit=True):
            query = st.text_area(
                "Votre question",
                placeholder="Posez votre question financière ici... (Shift+Enter pour nouvelle ligne, Enter pour envoyer)",
                height=80,
                label_visibility="collapsed",
            )
            col_submit, col_clear = st.columns([3, 1])
            with col_submit:
                submitted = st.form_submit_button(
                    "📤 Analyser",
                    use_container_width=True,
                    type="primary",
                )
            with col_clear:
                if st.form_submit_button("🗑️ Effacer", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()

        if submitted and query.strip():
            _process_query(query.strip(), system)
            st.rerun()

    with col_intel:
        _render_intelligence_panel()


def _render_message(msg: dict):
    """Rendu d'un message du chat."""
    role = msg.get("role", "user")
    content = msg.get("content", "")
    metadata = msg.get("metadata", {})

    if role == "user":
        st.markdown(f"""
        <div class="chat-user">
            <div style="font-size: 0.75rem; color: #9CA3AF; margin-bottom: 4px;">👤 Vous</div>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Assistant message
        confidence = metadata.get("confidence_score", 0)
        proc_time = metadata.get("processing_time", 0)
        tokens = metadata.get("tokens_used", 0)
        n_sources = metadata.get("context_docs_count", 0)

        badge = get_confidence_badge(confidence)

        st.markdown(f"""
        <div class="chat-assistant">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                <span style="font-size: 0.75rem; color: #9CA3AF;">🤖 FinRAG Assistant</span>
                {badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Render markdown content
        st.markdown(content)

        # Metrics row
        st.markdown(f"""
        <div style="display: flex; gap: 16px; margin-top: 8px; padding: 8px 0; border-top: 1px solid var(--border);">
            <span style="font-size: 0.72rem; color: #6B7280;">⏱ {format_processing_time(proc_time)}</span>
            <span style="font-size: 0.72rem; color: #6B7280;">🔤 {tokens:,} tokens</span>
            <span style="font-size: 0.72rem; color: #6B7280;">📚 {n_sources} sources</span>
        </div>
        """, unsafe_allow_html=True)

        # Citations
        citations = metadata.get("citations", [])
        if citations:
            with st.expander(f"📎 {len(citations)} citation(s)", expanded=False):
                for cit in citations[:8]:
                    source = cit.get("source_file", "unknown")
                    page = cit.get("page_number")
                    date = cit.get("date", "")
                    excerpt = cit.get("excerpt", "")
                    score = cit.get("relevance_score", 0)

                    loc = f"p.{page}" if page else ""
                    date_str = date if date else ""

                    st.markdown(f"""
                    <div class="citation-box">
                        📄 <strong>{source}</strong> {loc} {date_str} — score: {score:.2f}<br>
                        <span style="color: #D1D5DB; font-size: 0.78rem;">"{excerpt[:90]}..."</span>
                    </div>
                    """, unsafe_allow_html=True)


def _process_query(query: str, system: dict):
    """Traite une requête utilisateur."""
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })

    if not system.get("success"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Système RAG non initialisé. Veuillez vérifier la configuration.",
            "metadata": {},
        })
        return

    # Build filters
    filters = st.session_state.filters
    date_range = None
    if filters.get("date_from") and filters.get("date_to"):
        date_range = (filters["date_from"], filters["date_to"])

    doc_type_map = {
        "Rapports": ["annual_report", "quarterly_report", "market_overview"],
        "News": ["news_article"],
        "Tableaux": ["financial_table"],
        "CSV": ["csv_data"],
    }
    document_type = doc_type_map.get(filters.get("doc_type", "Tous"))
    ticker = filters.get("ticker") or None

    # Get answer
    agent = system["agent"]
    answer = agent.answer(
        question=query,
        use_decomposition=st.session_state.use_decomposition,
        date_range=date_range,
        document_type=document_type,
        ticker=ticker,
    )

    # Update session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer.answer,
        "metadata": {
            "confidence_score": answer.confidence_score,
            "processing_time": answer.processing_time,
            "tokens_used": answer.tokens_used,
            "context_docs_count": answer.context_docs_count,
            "citations": [c.__dict__ if hasattr(c, '__dict__') else c for c in answer.citations],
            "sub_queries": answer.sub_queries,
        },
    })

    st.session_state.last_sub_queries = answer.sub_queries
    st.session_state.last_sources = [
        {
            "filename": c.source_file if hasattr(c, 'source_file') else c.get("source_file", ""),
            "score": c.relevance_score if hasattr(c, 'relevance_score') else c.get("relevance_score", 0),
        }
        for c in answer.citations
    ] if answer.citations else []

    # Update counters
    st.session_state.total_queries += 1
    st.session_state.total_tokens_used += answer.tokens_used

    # Store in history
    st.session_state.query_history.append({
        "question": query,
        "time": time.time(),
        "confidence": answer.confidence_score,
        "processing_time": answer.processing_time,
        "tokens": answer.tokens_used,
        "n_sources": answer.context_docs_count,
    })


def _render_intelligence_panel():
    """Rendu du panel d'intelligence de requête (sidebar droite)."""
    st.markdown("""
    <h3 style="font-family: 'DM Serif Display'; color: #F9FAFB; font-size: 1rem; margin-bottom: 12px;">
        🧠 Query Intelligence
    </h3>
    """, unsafe_allow_html=True)

    # Sub-queries
    sub_queries = st.session_state.last_sub_queries
    if sub_queries:
        st.markdown('<div style="font-size: 0.75rem; color: #9CA3AF; margin-bottom: 6px;">Sous-requêtes décomposées :</div>', unsafe_allow_html=True)
        for sq in sub_queries:
            st.markdown(f'<div class="subquery-item">→ {sq[:60]}{"..." if len(sq) > 60 else ""}</div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: #6B7280; font-size: 0.8rem;">Posez une question pour voir les sous-requêtes générées</div>', unsafe_allow_html=True)

    st.markdown('<hr class="finrag-divider">', unsafe_allow_html=True)

    # Sources used
    sources = st.session_state.last_sources
    if sources:
        st.markdown('<div style="font-size: 0.75rem; color: #9CA3AF; margin-bottom: 8px;">Sources utilisées :</div>', unsafe_allow_html=True)

        # Mini bar chart
        if len(sources) > 0:
            filenames = [s.get("filename", "?")[:20] for s in sources[:5]]
            scores = [s.get("score", 0) for s in sources[:5]]

            fig = go.Figure(go.Bar(
                x=scores,
                y=filenames,
                orientation='h',
                marker=dict(
                    color=scores,
                    colorscale=[[0, '#1F2D45'], [1, '#3B82F6']],
                    showscale=False,
                ),
                text=[f"{s:.2f}" for s in scores],
                textposition='auto',
                textfont=dict(size=9, color='white'),
            ))
            fig.update_layout(
                height=150,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(size=9, color='#9CA3AF'),
                ),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<hr class="finrag-divider">', unsafe_allow_html=True)

    # Session stats
    st.markdown(f"""
    <div style="font-size: 0.75rem; color: #9CA3AF;">
        <div style="margin-bottom: 4px;">📊 Session actuelle :</div>
        <div>• {st.session_state.total_queries} requêtes</div>
        <div>• {st.session_state.total_tokens_used:,} tokens utilisés</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Documents Tab ────────────────────────────────────────────────────────────

def render_documents_tab(system: dict):
    """Rendu de l'onglet Documents."""
    st.markdown("""
    <h2 style="font-family: 'DM Serif Display'; color: #F9FAFB; margin-bottom: 4px;">
        📚 Documents Indexés
    </h2>
    """, unsafe_allow_html=True)

    if not system.get("success"):
        st.warning("Système non initialisé")
        return

    sources = system["vector_store"].list_sources()

    if not sources:
        st.info("📭 Aucun document indexé. Utilisez la sidebar pour importer des fichiers ou cliquez sur 'Indexer les données d'exemple'.")
        return

    # Search filter
    search_term = st.text_input("🔍 Rechercher un document", placeholder="Filtrer par nom de fichier...")

    if search_term:
        sources = [s for s in sources if search_term.lower() in s.get("filename", "").lower()]

    st.markdown(f"**{len(sources)} document(s) trouvé(s)**")

    # Documents table
    for src in sources:
        filename = src.get("filename", "unknown")
        doc_type = src.get("document_type", "unknown")
        chunk_count = src.get("chunk_count", 0)
        collection = src.get("collection", "unknown")
        date = src.get("date", "")
        badge, icon = get_doc_badge(doc_type)

        col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])

        with col1:
            st.markdown(f"{icon} **{filename}**")
        with col2:
            st.markdown(f"<span style='color: #9CA3AF; font-size: 0.85rem;'>{doc_type.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<span style='font-family: JetBrains Mono; font-size: 0.85rem;'>{chunk_count} chunks</span>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<span style='color: #9CA3AF; font-size: 0.8rem;'>{date}</span>", unsafe_allow_html=True)
        with col5:
            if st.button("🗑️", key=f"del_{filename}_{collection}", help=f"Supprimer {filename}"):
                deleted = system["vector_store"].delete_by_source(filename)
                if deleted > 0:
                    st.success(f"Supprimé {deleted} chunks")
                    system["retriever"].invalidate_bm25_cache()
                    st.rerun()

        st.markdown('<hr style="border-color: #1F2D45; margin: 4px 0;">', unsafe_allow_html=True)


# ─── Analytics Tab ────────────────────────────────────────────────────────────

def render_analytics_tab(system: dict):
    """Rendu de l'onglet Analytics."""
    st.markdown("""
    <h2 style="font-family: 'DM Serif Display'; color: #F9FAFB; margin-bottom: 4px;">
        📊 Analytics & Évaluation
    </h2>
    """, unsafe_allow_html=True)

    if not system.get("success"):
        st.warning("Système non initialisé")
        return

    # Vector Store Stats
    stats = system["vector_store"].get_stats()

    st.markdown("### 📈 Statistiques du Vector Store")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Chunks", f"{stats.get('total_chunks', 0):,}")
    with col2:
        st.metric("Sources", stats.get("total_sources", 0))
    with col3:
        st.metric("Requêtes session", st.session_state.total_queries)
    with col4:
        st.metric("Tokens utilisés", f"{st.session_state.total_tokens_used:,}")

    # Collections breakdown
    collections = stats.get("collections", {})
    if any(v > 0 for v in collections.values()):
        st.markdown("### 🗂️ Répartition par Collection")
        col_names = list(collections.keys())
        col_values = list(collections.values())

        fig_pie = go.Figure(go.Pie(
            labels=col_names,
            values=col_values,
            hole=0.5,
            marker=dict(
                colors=['#3B82F6', '#F0B429', '#10B981', '#EF4444'],
                line=dict(color='#0A0E1A', width=2),
            ),
            textfont=dict(color='white', size=12),
        ))
        fig_pie.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color='#9CA3AF')),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Query History
    if st.session_state.query_history:
        st.markdown("### 📜 Historique des Requêtes")

        history = st.session_state.query_history[-10:][::-1]

        for q in history:
            conf = q.get("confidence", 0)
            proc_t = q.get("processing_time", 0)
            conf_color = "#10B981" if conf >= 0.7 else "#F59E0B" if conf >= 0.4 else "#EF4444"

            st.markdown(f"""
            <div class="finrag-card" style="padding: 10px 14px;">
                <div style="color: #F9FAFB; font-size: 0.85rem; margin-bottom: 6px;">{q['question'][:100]}{"..." if len(q['question']) > 100 else ""}</div>
                <div style="display: flex; gap: 16px; font-size: 0.75rem; color: #9CA3AF;">
                    <span style="color: {conf_color};">● Confiance: {conf:.0%}</span>
                    <span>⏱ {format_processing_time(proc_t)}</span>
                    <span>🔤 {q.get('tokens', 0):,} tokens</span>
                    <span>📚 {q.get('n_sources', 0)} sources</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # RAGAS Evaluation section
    st.markdown("### 🧪 Évaluation RAGAS")
    st.info(
        "💡 Pour lancer une évaluation RAGAS complète, utilisez la commande :\n"
        "`python src/main.py evaluate`\n\n"
        "Ou depuis le notebook : `notebooks/04_evaluation_ragas.ipynb`"
    )

    # Check for existing evaluation report
    eval_report = ROOT_DIR / "docs" / "evaluation_report.md"
    if eval_report.exists():
        with st.expander("📄 Rapport d'évaluation disponible", expanded=True):
            content = eval_report.read_text(encoding="utf-8")
            st.markdown(content)

    # Gauge charts for RAGAS metrics (placeholder)
    st.markdown("### 📐 Métriques RAGAS (démonstration)")
    st.caption("Lancez `python src/main.py evaluate` pour obtenir des métriques réelles")

    demo_metrics = {
        "Faithfulness": 0.87,
        "Answer Relevancy": 0.82,
        "Context Recall": 0.78,
        "Context Precision": 0.84,
    }

    cols = st.columns(4)
    for i, (metric_name, value) in enumerate(demo_metrics.items()):
        with cols[i]:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value * 100,
                title={"text": metric_name, "font": {"size": 11, "color": "#9CA3AF"}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"size": 9}},
                    "bar": {"color": "#3B82F6"},
                    "bgcolor": "#1A2236",
                    "bordercolor": "#1F2D45",
                    "steps": [
                        {"range": [0, 50], "color": "#1F1010"},
                        {"range": [50, 75], "color": "#1F1A10"},
                        {"range": [75, 100], "color": "#101F14"},
                    ],
                    "threshold": {
                        "value": 80,
                        "line": {"color": "#10B981", "width": 2},
                        "thickness": 0.75,
                    },
                },
                number={"suffix": "%", "font": {"size": 20, "color": "#F0B429"}},
            ))
            fig_gauge.update_layout(
                height=200,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#F9FAFB"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    """Point d'entrée principal de l'application Streamlit."""
    # Initialize system
    with st.spinner("🔄 Initialisation du système RAG..."):
        system = initialize_rag_system()

    if not system.get("success"):
        st.error(f"⚠️ Erreur d'initialisation: {system.get('error', 'Erreur inconnue')}")
        st.info("Vérifiez votre configuration et relancez l'application.")
        st.code("pip install -r requirements.txt\nstreamlit run src/ui/app.py")
        return

    # Render sidebar
    render_sidebar(system)

    # Main content tabs
    tab_chat, tab_docs, tab_analytics = st.tabs([
        "💬 Assistant",
        "📚 Documents",
        "📊 Analytics",
    ])

    with tab_chat:
        render_chat(system)

    with tab_docs:
        render_documents_tab(system)

    with tab_analytics:
        render_analytics_tab(system)


if __name__ == "__main__":
    main()
