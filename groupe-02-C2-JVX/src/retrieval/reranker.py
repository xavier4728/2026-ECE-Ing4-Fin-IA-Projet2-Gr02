"""
CrossEncoder re-ranker pour améliorer la pertinence des résultats.
Modèle : cross-encoder/ms-marco-MiniLM-L-6-v2
Cache des scores pour éviter les re-calculs.
"""

from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from langchain.schema import Document
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


# ─── Score Cache ─────────────────────────────────────────────────────────────

class ScoreCache:
    """Cache simple LRU pour les scores CrossEncoder."""

    def __init__(self, maxsize: int = 1000) -> None:
        self._cache: Dict[str, float] = {}
        self._maxsize = maxsize

    def _make_key(self, query: str, text: str) -> str:
        raw = f"{query.strip()[:200]}::{text.strip()[:200]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, text: str) -> Optional[float]:
        key = self._make_key(query, text)
        return self._cache.get(key)

    def set(self, query: str, text: str, score: float) -> None:
        if len(self._cache) >= self._maxsize:
            # Simple eviction: remove oldest half
            keys = list(self._cache.keys())
            for k in keys[: self._maxsize // 2]:
                del self._cache[k]
        key = self._make_key(query, text)
        self._cache[key] = score

    def clear(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


# ─── CrossEncoder Re-ranker ───────────────────────────────────────────────────

class CrossEncoderReRanker:
    """
    Re-ranker basé sur CrossEncoder pour améliorer la pertinence.

    Modèle par défaut : cross-encoder/ms-marco-MiniLM-L-6-v2
    Input : query + liste de chunks candidats
    Output : chunks re-scorés et re-triés, top-k gardés
    Cache des scores pour éviter les re-calculs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        max_length: int = 512,
        cache_size: int = 1000,
    ) -> None:
        """
        Args:
            model_name: Modèle CrossEncoder HuggingFace.
            top_k: Nombre de résultats à retourner après re-ranking.
            max_length: Longueur maximale d'input (query + doc).
            cache_size: Taille du cache de scores.
        """
        self._model_name = model_name
        self._top_k = top_k
        self._max_length = max_length
        self._cache = ScoreCache(maxsize=cache_size)
        self._model = None
        self._model_loaded = False

        logger.info(f"CrossEncoderReRanker initialisé (lazy loading): {model_name}")

    def _load_model(self) -> None:
        """Charge le modèle CrossEncoder (lazy loading)."""
        if self._model_loaded:
            return

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
            )
            self._model_loaded = True
            logger.info(f"CrossEncoder chargé : {self._model_name}")
        except ImportError:
            logger.error("sentence_transformers non installé")
            self._model_loaded = True  # Prevent retry
            self._model = None
        except Exception as e:
            logger.error(f"Erreur chargement CrossEncoder: {e}")
            self._model_loaded = True
            self._model = None

    def rerank(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Re-classe les documents selon leur pertinence par rapport à la requête.

        Args:
            query: Requête utilisateur.
            documents: Liste de (Document, score) à re-classer.
            top_k: Nombre de résultats à retourner (utilise self._top_k si None).

        Returns:
            Liste re-classée de (Document, cross_encoder_score).
        """
        if not documents:
            return []

        actual_top_k = top_k or self._top_k

        # Load model if needed
        self._load_model()

        if self._model is None:
            logger.warning("CrossEncoder indisponible, retour des résultats originaux")
            return documents[:actual_top_k]

        # Prepare pairs for scoring
        pairs_to_score: List[Tuple[int, str]] = []  # (index, text)
        cached_scores: Dict[int, float] = {}

        for idx, (doc, _) in enumerate(documents):
            text = doc.page_content[:800]  # Truncate for efficiency
            cached_score = self._cache.get(query, text)

            if cached_score is not None:
                cached_scores[idx] = cached_score
            else:
                pairs_to_score.append((idx, text))

        # Compute uncached scores
        new_scores: Dict[int, float] = {}
        if pairs_to_score:
            try:
                batch_pairs = [[query, text] for _, text in pairs_to_score]
                raw_scores = self._model.predict(batch_pairs)

                for (idx, text), score in zip(pairs_to_score, raw_scores):
                    score_val = float(score)
                    new_scores[idx] = score_val
                    self._cache.set(query, text, score_val)

            except Exception as e:
                logger.error(f"CrossEncoder predict erreur: {e}")
                return documents[:actual_top_k]

        # Combine all scores
        all_scores = {**cached_scores, **new_scores}

        # Re-rank
        reranked = []
        for idx, (doc, original_score) in enumerate(documents):
            ce_score = all_scores.get(idx, original_score)
            reranked.append((doc, ce_score))

        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Re-ranking: {len(documents)} → {actual_top_k} docs "
            f"(cache hits: {len(cached_scores)}/{len(documents)})"
        )

        return reranked[:actual_top_k]

    def rerank_documents_only(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Re-classe une liste de Documents (sans scores initiaux).

        Args:
            query: Requête utilisateur.
            documents: Liste de Documents à re-classer.
            top_k: Nombre de résultats à retourner.

        Returns:
            Liste re-classée de Documents.
        """
        pairs = [(doc, 1.0) for doc in documents]
        reranked_pairs = self.rerank(query, pairs, top_k=top_k)
        return [doc for doc, _ in reranked_pairs]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        return {
            "cache_size": self._cache.size,
            "model_loaded": self._model_loaded,
            "model_name": self._model_name,
        }

    def clear_cache(self) -> None:
        """Vide le cache des scores."""
        self._cache.clear()
        logger.info("Cache CrossEncoder vidé")
