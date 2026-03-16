"""
Vector store ChromaDB persistant pour FinRAG.
Supporte embeddings OpenAI (text-embedding-3-small) avec fallback local (all-MiniLM-L6-v2).
Collections séparées par type de document.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from loguru import logger
from tqdm import tqdm

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


# ─── Embedding Functions ──────────────────────────────────────────────────────

class EmbeddingProvider:
    """Fournisseur d'embeddings avec fallback local."""

    def __init__(self) -> None:
        self._embed_fn = None
        self._model_name = ""
        self._initialize()

    def _initialize(self) -> None:
        if settings.use_openai_embeddings:
            self._initialize_openai()
        else:
            self._initialize_local()

    def _initialize_openai(self) -> None:
        try:
            from langchain_openai import OpenAIEmbeddings
            self._embed_fn = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
            )
            self._model_name = settings.EMBEDDING_MODEL
            logger.info(f"Embeddings OpenAI : {self._model_name}")
        except Exception as e:
            logger.warning(f"OpenAI embeddings échec ({e}), fallback local")
            self._initialize_local()

    def _initialize_local(self) -> None:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embed_fn = HuggingFaceEmbeddings(
                model_name=settings.FALLBACK_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._model_name = settings.FALLBACK_EMBEDDING_MODEL
            logger.info(f"Embeddings locaux : {self._model_name}")
        except Exception as e:
            logger.error(f"Impossible d'initialiser les embeddings: {e}")
            raise RuntimeError("Aucun fournisseur d'embeddings disponible") from e

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Génère les embeddings pour une liste de textes."""
        return self._embed_fn.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Génère l'embedding pour une requête."""
        return self._embed_fn.embed_query(text)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def langchain_embeddings(self):
        """Retourne l'objet embeddings compatible LangChain."""
        return self._embed_fn


# ─── ChromaDB Collection Manager ─────────────────────────────────────────────

COLLECTION_TYPES = ["reports", "news", "tables", "csv"]


def _make_doc_id(content: str, metadata: Dict[str, Any]) -> str:
    """Génère un ID unique et déterministe pour un document."""
    source = metadata.get("source", "")
    page = metadata.get("page_number", 0)
    chunk_idx = metadata.get("chunk_index", 0)
    raw = f"{source}::{page}::{chunk_idx}::{content[:100]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _classify_doc_type(metadata: Dict[str, Any]) -> str:
    """Classe un document dans une collection ChromaDB."""
    doc_type = metadata.get("document_type", "")
    if doc_type in ("financial_table",) or metadata.get("contains_table"):
        return "tables"
    elif doc_type in ("news_article",):
        return "news"
    elif doc_type in ("csv_data", "excel_data"):
        return "csv"
    else:
        return "reports"


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Sérialise les métadonnées pour ChromaDB (valeurs scalaires uniquement)."""
    serialized: Dict[str, Any] = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            serialized[k] = v
        elif isinstance(v, list):
            serialized[k] = json.dumps(v)
        elif v is None:
            serialized[k] = ""
        else:
            serialized[k] = str(v)
    return serialized


# ─── Main Vector Store ────────────────────────────────────────────────────────

class FinancialVectorStore:
    """
    Vector store ChromaDB persistant pour documents financiers.

    Collections séparées : reports | news | tables | csv
    CRUD complet : add_documents, delete_by_source, list_sources, get_stats
    Embeddings : OpenAI text-embedding-3-small (+ fallback all-MiniLM-L6-v2)
    """

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        """
        Args:
            persist_dir: Répertoire de persistance ChromaDB. Utilise settings si None.
        """
        self._persist_dir = persist_dir or str(settings.chroma_persist_path)
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialisation ChromaDB : {self._persist_dir}")

        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._embedding_provider = EmbeddingProvider()
        self._collections: Dict[str, chromadb.Collection] = {}
        self._init_collections()

    def _init_collections(self) -> None:
        """Initialise (ou récupère) les collections ChromaDB."""
        for coll_type in COLLECTION_TYPES:
            name = f"{settings.COLLECTION_PREFIX}_{coll_type}"
            try:
                coll = self._client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
                self._collections[coll_type] = coll
                logger.debug(f"Collection '{name}' : {coll.count()} docs")
            except Exception as e:
                logger.error(f"Erreur initialisation collection {name}: {e}")
                raise

    # ── Public API ───────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """
        Ajoute des documents au vector store.

        Args:
            documents: Liste de Documents LangChain avec métadonnées.
            batch_size: Taille des batches pour l'indexation.
            show_progress: Affiche une barre de progression.

        Returns:
            Nombre de documents ajoutés avec succès.
        """
        if not documents:
            return 0

        added_count = 0
        # Group by collection type
        by_collection: Dict[str, List[Document]] = {t: [] for t in COLLECTION_TYPES}
        for doc in documents:
            coll_type = _classify_doc_type(doc.metadata)
            by_collection[coll_type].append(doc)

        for coll_type, docs in by_collection.items():
            if not docs:
                continue

            collection = self._collections[coll_type]
            total_batches = (len(docs) + batch_size - 1) // batch_size

            iterator = range(0, len(docs), batch_size)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=total_batches,
                    desc=f"Indexation [{coll_type}]",
                )

            for start in iterator:
                batch = docs[start: start + batch_size]
                try:
                    texts = [d.page_content for d in batch]
                    metadatas = [_serialize_metadata(d.metadata) for d in batch]
                    ids = [_make_doc_id(d.page_content, d.metadata) for d in batch]

                    # Generate embeddings
                    embeddings = self._embedding_provider.embed_texts(texts)

                    # Upsert (handles duplicates)
                    collection.upsert(
                        documents=texts,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids,
                    )
                    added_count += len(batch)

                except Exception as e:
                    logger.error(f"Erreur batch [{coll_type}] {start}-{start+len(batch)}: {e}")
                    continue

        logger.success(f"Indexé {added_count}/{len(documents)} documents")
        return added_count

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        collection_types: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Recherche par similarité cosine.

        Args:
            query: Requête textuelle.
            k: Nombre de résultats par collection.
            collection_types: Collections à interroger (None = toutes).
            where: Filtre ChromaDB (ex: {"document_type": "news_article"}).

        Returns:
            Liste de (Document, score) triés par score décroissant.
        """
        query_embedding = self._embedding_provider.embed_query(query)
        collections_to_search = collection_types or COLLECTION_TYPES

        results: List[Tuple[Document, float]] = []

        for coll_type in collections_to_search:
            if coll_type not in self._collections:
                continue

            collection = self._collections[coll_type]
            n_docs = collection.count()
            if n_docs == 0:
                continue

            actual_k = min(k, n_docs)

            try:
                query_params = {
                    "query_embeddings": [query_embedding],
                    "n_results": actual_k,
                    "include": ["documents", "metadatas", "distances"],
                }
                if where:
                    query_params["where"] = where

                response = collection.query(**query_params)

                for text, meta, distance in zip(
                    response["documents"][0],
                    response["metadatas"][0],
                    response["distances"][0],
                ):
                    score = 1.0 - distance  # cosine distance → similarity
                    doc = Document(page_content=text, metadata=meta or {})
                    results.append((doc, score))

            except Exception as e:
                logger.debug(f"Recherche [{coll_type}] erreur: {e}")
                continue

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def delete_by_source(self, source_filename: str) -> int:
        """
        Supprime tous les chunks d'une source donnée.

        Args:
            source_filename: Nom du fichier source.

        Returns:
            Nombre de chunks supprimés.
        """
        deleted = 0
        for coll_type, collection in self._collections.items():
            try:
                results = collection.get(
                    where={"filename": source_filename},
                    include=["documents"],
                )
                if results["ids"]:
                    collection.delete(ids=results["ids"])
                    deleted += len(results["ids"])
                    logger.info(f"Supprimé {len(results['ids'])} chunks [{coll_type}] pour {source_filename}")
            except Exception as e:
                logger.debug(f"delete_by_source [{coll_type}]: {e}")

        return deleted

    def list_sources(self) -> List[Dict[str, Any]]:
        """
        Liste toutes les sources indexées avec statistiques.

        Returns:
            Liste de dicts {filename, document_type, chunk_count, collection}.
        """
        sources: Dict[str, Dict[str, Any]] = {}

        for coll_type, collection in self._collections.items():
            try:
                if collection.count() == 0:
                    continue

                results = collection.get(include=["metadatas"])
                for meta in results.get("metadatas", []):
                    if not meta:
                        continue
                    filename = meta.get("filename", "unknown")
                    key = f"{coll_type}::{filename}"

                    if key not in sources:
                        sources[key] = {
                            "filename": filename,
                            "document_type": meta.get("document_type", "unknown"),
                            "chunk_count": 0,
                            "collection": coll_type,
                            "date": meta.get("date", ""),
                        }
                    sources[key]["chunk_count"] += 1

            except Exception as e:
                logger.debug(f"list_sources [{coll_type}]: {e}")

        return list(sources.values())

    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques globales du vector store.

        Returns:
            Dict avec total_documents, total_chunks, collections, embedding_model.
        """
        stats: Dict[str, Any] = {
            "embedding_model": self._embedding_provider.model_name,
            "persist_dir": self._persist_dir,
            "collections": {},
            "total_chunks": 0,
        }

        for coll_type, collection in self._collections.items():
            count = collection.count()
            stats["collections"][coll_type] = count
            stats["total_chunks"] += count

        stats["total_sources"] = len(self.list_sources())
        return stats

    def get_all_texts(self, collection_types: Optional[List[str]] = None) -> List[str]:
        """
        Récupère tous les textes du store (pour BM25).

        Args:
            collection_types: Collections à inclure (None = toutes).

        Returns:
            Liste de tous les textes indexés.
        """
        collections_to_use = collection_types or COLLECTION_TYPES
        texts: List[str] = []

        for coll_type in collections_to_use:
            if coll_type not in self._collections:
                continue
            collection = self._collections[coll_type]
            if collection.count() == 0:
                continue
            try:
                results = collection.get(include=["documents"])
                texts.extend(results.get("documents", []))
            except Exception as e:
                logger.debug(f"get_all_texts [{coll_type}]: {e}")

        return texts

    def get_all_documents(
        self,
        collection_types: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Récupère tous les Documents du store.

        Args:
            collection_types: Collections à inclure (None = toutes).

        Returns:
            Liste de tous les Documents.
        """
        collections_to_use = collection_types or COLLECTION_TYPES
        docs: List[Document] = []

        for coll_type in collections_to_use:
            if coll_type not in self._collections:
                continue
            collection = self._collections[coll_type]
            if collection.count() == 0:
                continue
            try:
                results = collection.get(include=["documents", "metadatas"])
                for text, meta in zip(
                    results.get("documents", []),
                    results.get("metadatas", []),
                ):
                    docs.append(Document(page_content=text, metadata=meta or {}))
            except Exception as e:
                logger.debug(f"get_all_documents [{coll_type}]: {e}")

        return docs
