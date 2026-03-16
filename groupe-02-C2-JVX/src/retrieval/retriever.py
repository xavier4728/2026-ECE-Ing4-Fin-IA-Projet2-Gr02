"""
Retriever hybride (Dense + BM25) avec Reciprocal Rank Fusion.
Filtres temporels, par type de document et ticker.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from langchain_core.documents import Document
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


# ─── RRF Implementation ───────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
    top_n: int = 10,
) -> List[Tuple[Document, float]]:
    """
    Reciprocal Rank Fusion pour combiner plusieurs classements.

    Args:
        ranked_lists: Listes de (Document, score) déjà triées par pertinence.
        k: Constante RRF (typiquement 60).
        top_n: Nombre de résultats à retourner.

    Returns:
        Liste fusionnée de (Document, rrf_score) triée par score décroissant.
    """
    doc_scores: Dict[str, float] = {}
    doc_objects: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _) in enumerate(ranked_list):
            doc_key = (
                doc.metadata.get("source", "")
                + "::"
                + str(doc.metadata.get("chunk_index", 0))
                + "::"
                + doc.page_content[:50]
            )

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_key] = doc_scores.get(doc_key, 0.0) + rrf_score
            doc_objects[doc_key] = doc

    sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        (doc_objects[key], score)
        for key, score in sorted_items[:top_n]
    ]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_year(val: str) -> Optional[int]:
    """
    FIX MOYEN : parse robuste d'une année depuis des formats variés.
    Supporte : "2023", "Q4 2024", "FY2023", "2024-01-01", etc.
    Retourne None si le parsing échoue, avec un warning loggé.
    """
    if not val:
        return None
    m = re.search(r'(20\d{2})', str(val))
    if m:
        return int(m.group(1))
    return None


# ─── BM25 Retriever ───────────────────────────────────────────────────────────

class BM25Retriever:
    """BM25 sparse retriever sur le corpus complet."""

    def __init__(self, documents: List[Document]) -> None:
        from rank_bm25 import BM25Okapi

        self._documents = documents
        tokenized = [self._tokenize(d.page_content) for d in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 initialisé sur {len(documents)} documents")

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            (self._documents[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]


# ─── Main Hybrid Retriever ────────────────────────────────────────────────────

class HybridFinancialRetriever:
    """
    Retriever hybride pour documents financiers.

    Combine Dense (ChromaDB) + Sparse (BM25) via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        vector_store,
        top_k: int = 10,
        dense_k: int = 20,
        sparse_k: int = 20,
        time_decay_factor: float = 0.1,
    ) -> None:
        self._vector_store = vector_store
        self._top_k = top_k
        self._dense_k = dense_k
        self._sparse_k = sparse_k
        self._time_decay_factor = time_decay_factor
        self._bm25: Optional[BM25Retriever] = None

        logger.info(
            f"HybridRetriever initialisé (dense_k={dense_k}, sparse_k={sparse_k}, top_k={top_k})"
        )

    def _ensure_bm25(self) -> None:
        """
        FIX ÉLEVÉ : initialise BM25 avec plafond mémoire configurable.
        Sur un corpus >10 000 docs, BM25 est tronqué pour éviter les OOM.
        """
        if self._bm25 is not None:
            return

        logger.info("Initialisation BM25 (lazy)...")
        docs = self._vector_store.get_all_documents()

        if not docs:
            logger.warning("Vector store vide, BM25 non initialisé")
            return

        max_docs = settings.BM25_MAX_DOCS
        if len(docs) > max_docs:
            logger.warning(
                f"BM25: corpus tronqué à {max_docs} docs "
                f"(total={len(docs)}) pour éviter l'OOM. "
                f"Augmentez BM25_MAX_DOCS dans .env si nécessaire."
            )
            docs = docs[:max_docs]

        self._bm25 = BM25Retriever(docs)

    def retrieve(
        self,
        query: str,
        date_range: Optional[Tuple[str, str]] = None,
        document_type: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        min_relevance_score: float = 0.0,
        use_hybrid: bool = True,
    ) -> List[Tuple[Document, float]]:
        """Retrieval hybride avec filtres et RRF."""
        where_filter = self._build_where_filter(document_type, ticker)
        collection_types = self._infer_collections(document_type)

        # 1. Dense retrieval
        dense_results = self._vector_store.similarity_search(
            query=query,
            k=self._dense_k,
            collection_types=collection_types,
            where=where_filter if where_filter else None,
        )
        logger.debug(f"Dense retrieval: {len(dense_results)} résultats")

        if not use_hybrid:
            return self._apply_filters_and_boost(
                dense_results, date_range, min_relevance_score
            )[:self._top_k]

        # 2. Sparse retrieval (BM25)
        self._ensure_bm25()
        sparse_results: List[Tuple[Document, float]] = []

        if self._bm25 is not None:
            sparse_results = self._bm25.search(query, k=self._sparse_k)
            sparse_results = self._filter_sparse(sparse_results, document_type, ticker)
            logger.debug(f"Sparse retrieval: {len(sparse_results)} résultats")

        # 3. RRF fusion
        lists_to_fuse = [lst for lst in [dense_results, sparse_results] if lst]

        if len(lists_to_fuse) == 1:
            fused = list(lists_to_fuse[0])
        else:
            fused = reciprocal_rank_fusion(
                ranked_lists=lists_to_fuse,
                top_n=self._top_k * 2,
            )

        # 4. Post-filtering and time-aware boost
        final = self._apply_filters_and_boost(fused, date_range, min_relevance_score)

        logger.info(f"Hybrid retrieval: {len(final)} résultats finaux")
        return final[:self._top_k]

    def _build_where_filter(
        self,
        document_type: Optional[List[str]],
        ticker: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        conditions = []

        if document_type and len(document_type) == 1:
            conditions.append({"document_type": {"$eq": document_type[0]}})

        if ticker:
            conditions.append({"ticker_symbols": {"$contains": ticker}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _infer_collections(
        self,
        document_type: Optional[List[str]],
    ) -> Optional[List[str]]:
        if not document_type:
            return None

        mapping = {
            "news_article": ["news"],
            "financial_table": ["tables"],
            "annual_report": ["reports"],
            "quarterly_report": ["reports"],
            "csv_data": ["csv"],
            "market_overview": ["reports"],
        }

        collections = set()
        for dt in document_type:
            colls = mapping.get(dt, ["reports"])
            collections.update(colls)

        return list(collections) if collections else None

    def _filter_sparse(
        self,
        results: List[Tuple[Document, float]],
        document_type: Optional[List[str]],
        ticker: Optional[str],
    ) -> List[Tuple[Document, float]]:
        filtered = []
        for doc, score in results:
            meta = doc.metadata

            if document_type:
                if meta.get("document_type") not in document_type:
                    continue

            if ticker:
                tickers_raw = meta.get("ticker_symbols", "[]")
                if isinstance(tickers_raw, str):
                    try:
                        tickers = json.loads(tickers_raw)
                    except Exception:
                        tickers = []
                else:
                    tickers = tickers_raw or []
                if ticker not in tickers:
                    continue

            filtered.append((doc, score))
        return filtered

    def _apply_filters_and_boost(
        self,
        results: List[Tuple[Document, float]],
        date_range: Optional[Tuple[str, str]],
        min_relevance_score: float,
    ) -> List[Tuple[Document, float]]:
        """
        FIX MOYEN : parsing robuste des dates via _parse_year().
        Les formats "Q4 2024", "FY2023" sont maintenant correctement gérés.
        Avant, int("Q4 2"[:4]) levait une ValueError silencieusement ignorée.
        """
        if not results:
            return []

        current_year = datetime.now().year
        boosted: List[Tuple[Document, float]] = []

        # Pré-parser le date_range une seule fois (et logguer si invalide)
        start_year: Optional[int] = None
        end_year: Optional[int] = None
        if date_range:
            start_year = _parse_year(str(date_range[0]))
            end_year = _parse_year(str(date_range[1]))
            if start_year is None or end_year is None:
                logger.warning(
                    f"date_range non parseable: {date_range} — filtre temporel désactivé"
                )
                start_year = end_year = None

        for doc, score in results:
            if score < min_relevance_score:
                continue

            meta = doc.metadata
            doc_period = meta.get("date", "") or meta.get("time_period", "")

            # Date range filter
            if start_year is not None and end_year is not None and doc_period:
                doc_year = _parse_year(str(doc_period))
                if doc_year is not None:
                    if not (start_year <= doc_year <= end_year):
                        continue
                # Si doc_year non parseable, on garde le doc (on ne filtre pas)

            # Time-aware boost (documents récents prioritaires)
            if self._time_decay_factor > 0 and doc_period:
                doc_year = _parse_year(str(doc_period))
                if doc_year is not None:
                    years_ago = max(0, current_year - doc_year)
                    boost = 1.0 / (1.0 + self._time_decay_factor * years_ago)
                    score = score * boost

            boosted.append((doc, score))

        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def invalidate_bm25_cache(self) -> None:
        """Invalide le cache BM25 (à appeler après ajout de nouveaux documents)."""
        self._bm25 = None
        logger.info("Cache BM25 invalidé")