"""
Retriever hybride (Dense + BM25) avec Reciprocal Rank Fusion.
Filtres temporels, par type de document et ticker.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from langchain.schema import Document
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
            # Create unique key for deduplication
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

    # Sort by aggregated RRF score
    sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        (doc_objects[key], score)
        for key, score in sorted_items[:top_n]
    ]


# ─── BM25 Retriever ───────────────────────────────────────────────────────────

class BM25Retriever:
    """BM25 sparse retriever sur le corpus complet."""

    def __init__(self, documents: List[Document]) -> None:
        """
        Args:
            documents: Corpus complet de documents pour BM25.
        """
        from rank_bm25 import BM25Okapi

        self._documents = documents
        tokenized = [self._tokenize(d.page_content) for d in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 initialisé sur {len(documents)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenization simple pour BM25."""
        return text.lower().split()

    def search(self, query: str, k: int = 20) -> List[Tuple[Document, float]]:
        """
        Recherche BM25.

        Args:
            query: Requête textuelle.
            k: Nombre de résultats.

        Returns:
            Liste de (Document, score) triés par score décroissant.
        """
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = [
            (self._documents[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]
        return results


# ─── Main Hybrid Retriever ────────────────────────────────────────────────────

class HybridFinancialRetriever:
    """
    Retriever hybride pour documents financiers.

    Combine :
    - Dense retrieval (ChromaDB cosine similarity, top-k=20)
    - Sparse retrieval (BM25 sur le corpus complet, top-k=20)
    via Reciprocal Rank Fusion → top-k=10 résultats finaux.

    Filtres disponibles :
    - date_range: tuple(str, str) au format YYYY ou Q1 2024
    - document_type: list[str]
    - ticker: str
    - min_relevance_score: float

    Time-aware : boost des documents récents configurable (decay_factor).
    """

    def __init__(
        self,
        vector_store,  # FinancialVectorStore
        top_k: int = 10,
        dense_k: int = 20,
        sparse_k: int = 20,
        time_decay_factor: float = 0.1,
    ) -> None:
        """
        Args:
            vector_store: Instance FinancialVectorStore.
            top_k: Nombre de résultats finaux après RRF.
            dense_k: Top-k pour dense retrieval avant RRF.
            sparse_k: Top-k pour sparse retrieval avant RRF.
            time_decay_factor: Facteur de boost temporel (0 = désactivé).
        """
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
        """Initialise BM25 à la demande (lazy loading)."""
        if self._bm25 is None:
            logger.info("Initialisation BM25 (lazy)...")
            docs = self._vector_store.get_all_documents()
            if docs:
                self._bm25 = BM25Retriever(docs)
            else:
                logger.warning("Vector store vide, BM25 non initialisé")

    def retrieve(
        self,
        query: str,
        date_range: Optional[Tuple[str, str]] = None,
        document_type: Optional[List[str]] = None,
        ticker: Optional[str] = None,
        min_relevance_score: float = 0.0,
        use_hybrid: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieval hybride avec filtres et RRF.

        Args:
            query: Requête utilisateur.
            date_range: Tuple (start_year, end_year) pour filtre temporel.
            document_type: Liste de types de documents à inclure.
            ticker: Filtre sur le ticker boursier.
            min_relevance_score: Score minimum de pertinence.
            use_hybrid: Si False, utilise seulement le dense retrieval.

        Returns:
            Liste de (Document, score) triés par pertinence.
        """
        # Build ChromaDB where filter
        where_filter = self._build_where_filter(document_type, ticker)

        # Collection filter based on doc types
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
            # Apply filters to sparse results
            sparse_results = self._filter_sparse(
                sparse_results, document_type, ticker
            )
            logger.debug(f"Sparse retrieval: {len(sparse_results)} résultats")

        # 3. RRF fusion
        lists_to_fuse = [l for l in [dense_results, sparse_results] if l]

        if len(lists_to_fuse) == 1:
            fused = [(doc, score) for doc, score in lists_to_fuse[0]]
        else:
            fused = reciprocal_rank_fusion(
                ranked_lists=lists_to_fuse,
                top_n=self._top_k * 2,  # get more before filtering
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
        """Construit le filtre ChromaDB where."""
        conditions = []

        if document_type and len(document_type) == 1:
            conditions.append({"document_type": {"$eq": document_type[0]}})

        if ticker:
            # ChromaDB stores tickers as JSON list string
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
        """Infère les collections à chercher selon le type de document."""
        if not document_type:
            return None  # Search all

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
        """Applique les filtres sur les résultats BM25."""
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
        """Applique le filtre temporel et le boost time-aware."""
        if not results:
            return []

        current_year = datetime.now().year
        boosted: List[Tuple[Document, float]] = []

        for doc, score in results:
            if score < min_relevance_score:
                continue

            meta = doc.metadata
            doc_period = meta.get("date", "") or meta.get("time_period", "")

            # Date range filter
            if date_range and doc_period:
                try:
                    doc_year = int(str(doc_period)[:4])
                    start_year = int(str(date_range[0])[:4])
                    end_year = int(str(date_range[1])[:4])
                    if not (start_year <= doc_year <= end_year):
                        continue
                except (ValueError, TypeError):
                    pass  # Keep if can't parse

            # Time-aware boost (recent documents score higher)
            if self._time_decay_factor > 0 and doc_period:
                try:
                    doc_year = int(str(doc_period)[:4])
                    years_ago = max(0, current_year - doc_year)
                    boost = 1.0 / (1.0 + self._time_decay_factor * years_ago)
                    score = score * boost
                except (ValueError, TypeError):
                    pass

            boosted.append((doc, score))

        # Re-sort after boost
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def invalidate_bm25_cache(self) -> None:
        """Invalide le cache BM25 (à appeler après ajout de nouveaux documents)."""
        self._bm25 = None
        logger.info("Cache BM25 invalidé")
