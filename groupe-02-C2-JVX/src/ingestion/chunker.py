"""
Chunker intelligent pour documents financiers.
Plusieurs stratégies : sémantique, table-aware, sentence, hybrid.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class ChunkingStrategy(Enum):
    """Stratégies de chunking disponibles."""
    SEMANTIC = "semantic"        # RecursiveCharacterTextSplitter avec overlap
    TABLE_AWARE = "table_aware"  # Préserve les tableaux entiers
    SENTENCE = "sentence"        # Découpage par phrases
    HYBRID = "hybrid"            # Auto-détection selon le contenu


# ─── Helpers ─────────────────────────────────────────────────────────────────

FINANCIAL_ENTITIES = [
    # Companies
    "Apple", "Microsoft", "NVIDIA", "Google", "Alphabet", "Amazon", "Meta",
    "Tesla", "JPMorgan", "Goldman Sachs", "Morgan Stanley",
    # Metrics
    "chiffre d'affaires", "bénéfice net", "marge brute", "EBITDA", "EPS",
    "revenue", "net income", "gross margin", "cash flow", "dividende",
    "résultat opérationnel", "résultat net", "capitalisation",
    # Time periods
    "Q1", "Q2", "Q3", "Q4", "FY2023", "FY2024", "T1", "T2", "T3", "T4",
    # Indices
    "S&P 500", "Nasdaq", "CAC 40", "Dow Jones", "MSCI",
]

TIME_PERIOD_PATTERN = re.compile(
    r'\b(Q[1-4]\s*20\d{2}|FY20\d{2}|T[1-4]\s*20\d{2}|20\d{2})\b'
)

NUMBER_PATTERN = re.compile(r'\b\d+[\d,\.]*\s*(%|Md|M|B|K|million|billion|trillion)?\b')


def _contains_table(text: str) -> bool:
    """Détecte si le texte contient un tableau markdown."""
    return "|" in text and "---" in text


def _contains_numbers(text: str) -> bool:
    """Détecte si le texte contient des données chiffrées significatives."""
    matches = NUMBER_PATTERN.findall(text)
    return len(matches) >= 3


def _extract_financial_entities(text: str) -> List[str]:
    """Extrait les entités financières mentionnées dans le texte."""
    found = []
    text_lower = text.lower()
    for entity in FINANCIAL_ENTITIES:
        if entity.lower() in text_lower:
            found.append(entity)
    return found[:10]  # Limit to top 10


def _extract_time_period(text: str) -> str:
    """Extrait la période temporelle principale du chunk."""
    match = TIME_PERIOD_PATTERN.search(text)
    return match.group(1) if match else ""


def _build_chunk_metadata(
    chunk_text: str,
    base_metadata: Dict[str, Any],
    chunk_index: int,
    strategy: str,
) -> Dict[str, Any]:
    """Construit les métadonnées enrichies d'un chunk."""
    meta = dict(base_metadata)
    meta.update({
        "chunk_index": chunk_index,
        "chunk_strategy": strategy,
        "contains_table": _contains_table(chunk_text),
        "contains_numbers": _contains_numbers(chunk_text),
        "time_period": _extract_time_period(chunk_text),
        "financial_entities": _extract_financial_entities(chunk_text),
        "chunk_length": len(chunk_text),
    })
    return meta


# ─── Main Chunker ─────────────────────────────────────────────────────────────

class IntelligentFinancialChunker:
    """
    Chunker adaptatif pour documents financiers.

    Stratégies :
    - SEMANTIC : RecursiveCharacterTextSplitter (texte narratif, 512 tokens)
    - TABLE_AWARE : Préserve les tableaux entiers
    - SENTENCE : Découpage par phrases
    - HYBRID : Auto-détection (table → TABLE_AWARE, texte → SEMANTIC)

    Tagging automatique : {contains_table, contains_numbers,
                            time_period, financial_entities}
    Filtrage des chunks trop courts ou non informatifs.
    """

    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        chunk_size: int = 512,
        chunk_overlap: int = 51,
        min_chunk_length: int = 50,
    ) -> None:
        """
        Args:
            strategy: Stratégie de chunking.
            chunk_size: Taille cible des chunks (en caractères ~= tokens).
            chunk_overlap: Overlap entre chunks consécutifs.
            min_chunk_length: Longueur minimale pour qu'un chunk soit gardé.
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

        # Splitter sémantique standard
        self._semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

        # Splitter pour texte financier (respecte les paragraphes)
        self._financial_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Découpe une liste de documents en chunks.

        Args:
            documents: Documents LangChain à découper.

        Returns:
            Liste de chunks (Documents) avec métadonnées enrichies.
        """
        all_chunks: List[Document] = []

        for doc in documents:
            try:
                chunks = self._chunk_single(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Erreur chunking doc {doc.metadata.get('source', '?')}: {e}")
                continue

        logger.info(f"Chunking : {len(documents)} docs → {len(all_chunks)} chunks")
        return all_chunks

    def _chunk_single(self, doc: Document) -> List[Document]:
        """Découpe un seul document selon la stratégie choisie."""
        text = doc.page_content
        meta = doc.metadata

        # Determine effective strategy
        effective_strategy = self.strategy
        if self.strategy == ChunkingStrategy.HYBRID:
            if _contains_table(text):
                effective_strategy = ChunkingStrategy.TABLE_AWARE
            else:
                effective_strategy = ChunkingStrategy.SEMANTIC

        if effective_strategy == ChunkingStrategy.TABLE_AWARE:
            return self._chunk_table_aware(text, meta)
        elif effective_strategy == ChunkingStrategy.SENTENCE:
            return self._chunk_by_sentence(text, meta)
        else:  # SEMANTIC
            return self._chunk_semantic(text, meta)

    def _chunk_semantic(self, text: str, base_meta: Dict[str, Any]) -> List[Document]:
        """Chunking sémantique standard."""
        raw_chunks = self._financial_splitter.split_text(text)
        return self._build_docs(raw_chunks, base_meta, "semantic")

    def _chunk_table_aware(self, text: str, base_meta: Dict[str, Any]) -> List[Document]:
        """
        Chunking qui préserve les tableaux markdown entiers.
        Découpe le texte en sections (hors tableaux) + tableaux complets.
        """
        parts: List[str] = []
        current_pos = 0

        # Find all table blocks (markdown tables)
        table_pattern = re.compile(
            r'((?:\|[^\n]+\|\n)+(?:\|[-| :]+\|\n)(?:\|[^\n]+\|\n)*)',
            re.MULTILINE,
        )

        for match in table_pattern.finditer(text):
            # Text before table
            before = text[current_pos:match.start()].strip()
            if before:
                # Split the text before table semantically
                sub_chunks = self._financial_splitter.split_text(before)
                parts.extend(sub_chunks)

            # Keep table as single chunk
            table_text = match.group(0).strip()
            if table_text:
                parts.append(table_text)

            current_pos = match.end()

        # Remaining text
        remaining = text[current_pos:].strip()
        if remaining:
            sub_chunks = self._financial_splitter.split_text(remaining)
            parts.extend(sub_chunks)

        if not parts:
            parts = self._financial_splitter.split_text(text)

        return self._build_docs(parts, base_meta, "table_aware")

    def _chunk_by_sentence(self, text: str, base_meta: Dict[str, Any]) -> List[Document]:
        """Chunking par phrases (respecte les frontières de phrases)."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks: List[str] = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return self._build_docs(chunks, base_meta, "sentence")

    def _build_docs(
        self,
        texts: List[str],
        base_meta: Dict[str, Any],
        strategy: str,
    ) -> List[Document]:
        """Construit les objets Document avec métadonnées enrichies."""
        docs: List[Document] = []
        chunk_index = 0

        for text in texts:
            text = text.strip()
            if len(text) < self.min_chunk_length:
                continue  # Filter out too-short chunks

            meta = _build_chunk_metadata(
                chunk_text=text,
                base_metadata=base_meta,
                chunk_index=chunk_index,
                strategy=strategy,
            )
            docs.append(Document(page_content=text, metadata=meta))
            chunk_index += 1

        return docs
