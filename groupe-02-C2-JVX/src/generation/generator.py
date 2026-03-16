"""
Générateur de réponses financières avec Claude (Anthropic).
Citations vérifiables, format structuré, streaming support.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator

from langchain.schema import Document
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """Citation vérifiable pointant vers le chunk source."""
    chunk_id: str
    source_file: str
    page_number: Optional[int]
    date: Optional[str]
    excerpt: str        # 100 chars max du chunk original
    relevance_score: float

    def to_markdown(self) -> str:
        """Formate la citation en markdown inline."""
        parts = [f"**{self.source_file}**"]
        if self.page_number:
            parts.append(f"p.{self.page_number}")
        if self.date:
            parts.append(self.date)
        return f"[Source: {', '.join(parts)}]"


@dataclass
class FinancialAnswer:
    """Réponse financière structurée avec citations et métriques."""
    question: str
    answer: str                      # Réponse en markdown
    citations: List[Citation] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    tokens_used: int = 0
    context_docs_count: int = 0

    def to_dict(self) -> dict:
        """Sérialise la réponse en dict."""
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [
                {
                    "chunk_id": c.chunk_id,
                    "source_file": c.source_file,
                    "page_number": c.page_number,
                    "date": c.date,
                    "excerpt": c.excerpt,
                    "relevance_score": c.relevance_score,
                }
                for c in self.citations
            ],
            "sub_queries": self.sub_queries,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "tokens_used": self.tokens_used,
            "context_docs_count": self.context_docs_count,
        }


# ─── Prompt Engineering ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un analyste financier expert spécialisé dans l'analyse de documents financiers (rapports annuels, rapports trimestriels, articles de presse financière, données de marché).

**RÈGLES ABSOLUES :**
1. Tu réponds UNIQUEMENT en te basant sur les documents fournis dans le contexte.
2. Chaque affirmation chiffrée DOIT être suivie d'une citation [Source: {filename}, p.{page}, {date}].
3. Si l'information n'est PAS dans le contexte, dis-le EXPLICITEMENT : "Cette information n'est pas disponible dans les documents fournis."
4. Ne jamais inventer de chiffres ou de faits.
5. Distingue clairement les données historiques des projections.

**FORMAT DE RÉPONSE :**
- Réponse directe en 1-2 phrases (synthèse)
- Analyse détaillée avec données chiffrées et citations
- Tableau comparatif si pertinent (markdown)
- Limites et mises en garde si les données sont incomplètes ou anciennes

**STYLE :**
- Professionnel et précis
- Utilise le markdown pour la mise en forme
- Chiffres toujours avec leur unité (Md$, %, bp, etc.)
- Distingue M$ (millions) de Md$ (milliards)"""


def _build_context(
    documents: List[Document],
    max_context_chars: int = 8000,
) -> str:
    """
    Construit le contexte textuel à partir des documents récupérés.

    Args:
        documents: Documents LangChain récupérés.
        max_context_chars: Limite de caractères pour le contexte.

    Returns:
        Contexte formaté pour le prompt.
    """
    context_parts = []
    total_chars = 0

    for i, doc in enumerate(documents):
        meta = doc.metadata
        source = meta.get("filename", meta.get("source", "unknown"))
        page = meta.get("page_number", "")
        date = meta.get("date", "")
        doc_type = meta.get("document_type", "")

        header = f"[Document {i+1}: {source}"
        if page:
            header += f", p.{page}"
        if date:
            header += f", {date}"
        if doc_type:
            header += f", type={doc_type}"
        header += "]"

        chunk_text = f"{header}\n{doc.page_content}\n"

        if total_chars + len(chunk_text) > max_context_chars:
            # Truncate this chunk
            remaining = max_context_chars - total_chars
            if remaining > 200:
                chunk_text = chunk_text[:remaining] + "...[tronqué]"
                context_parts.append(chunk_text)
            break

        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    return "\n---\n".join(context_parts)


def _extract_citations(
    documents: List[Document],
    scores: Optional[List[float]] = None,
) -> List[Citation]:
    """Extrait les citations à partir des documents sources."""
    citations = []
    for i, doc in enumerate(documents):
        meta = doc.metadata
        score = scores[i] if scores and i < len(scores) else 0.5

        citation = Citation(
            chunk_id=f"{meta.get('source', 'unknown')}::{meta.get('chunk_index', i)}",
            source_file=meta.get("filename", meta.get("source", "unknown")),
            page_number=meta.get("page_number"),
            date=meta.get("date") or meta.get("time_period"),
            excerpt=doc.page_content[:100].replace("\n", " "),
            relevance_score=float(score),
        )
        citations.append(citation)

    return citations


def _estimate_confidence(answer: str, citations: List[Citation], n_docs: int) -> float:
    """Estime la confiance de la réponse (0.0 à 1.0)."""
    score = 0.0

    # Has citations
    if citations:
        score += 0.3
    if len(citations) >= 3:
        score += 0.2

    # Has numbers/data
    import re
    if re.search(r'\d+[\.,]\d+|\d+\s*(?:Md|M|%)', answer):
        score += 0.2

    # Good context coverage
    if n_docs >= 3:
        score += 0.2
    elif n_docs >= 1:
        score += 0.1

    # Doesn't say "not available"
    if "n'est pas disponible" not in answer.lower() and "pas dans le contexte" not in answer.lower():
        score += 0.1

    return min(1.0, score)


# ─── Main Generator ───────────────────────────────────────────────────────────

class FinancialAnswerGenerator:
    """
    Générateur de réponses financières via Claude (Anthropic).

    - Prompt système spécialisé analyse financière
    - Citations vérifiables avec métadonnées complètes
    - Streaming support
    - Fallback si API indisponible
    """

    def __init__(self) -> None:
        self._client = None
        self._model = settings.LLM_MODEL
        self._initialized = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialise le client Anthropic."""
        if not settings.use_anthropic:
            logger.warning(
                "ANTHROPIC_API_KEY non défini. "
                "Le générateur fonctionnera en mode dégradé."
            )
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            self._initialized = True
            logger.info(f"Générateur Anthropic initialisé : {self._model}")
        except ImportError:
            logger.error("Package 'anthropic' non installé. pip install anthropic")
        except Exception as e:
            logger.error(f"Erreur initialisation Anthropic: {e}")

    def generate(
        self,
        question: str,
        context_documents: List[Document],
        context_scores: Optional[List[float]] = None,
        sub_queries: Optional[List[str]] = None,
        max_tokens: int = 2000,
    ) -> FinancialAnswer:
        """
        Génère une réponse financière à partir du contexte récupéré.

        Args:
            question: Question de l'utilisateur.
            context_documents: Documents de contexte récupérés.
            context_scores: Scores de pertinence correspondants.
            sub_queries: Sous-requêtes utilisées si décomposition.
            max_tokens: Nombre max de tokens pour la réponse.

        Returns:
            FinancialAnswer avec réponse, citations et métriques.
        """
        start_time = time.time()

        if not context_documents:
            return FinancialAnswer(
                question=question,
                answer="⚠️ Aucun document pertinent trouvé dans la base de connaissances. "
                       "Veuillez d'abord indexer des documents financiers.",
                confidence_score=0.0,
                processing_time=time.time() - start_time,
            )

        # Build context
        context_text = _build_context(context_documents)
        citations = _extract_citations(context_documents, context_scores)

        # Generate answer
        if self._initialized and self._client:
            answer_text, tokens_used = self._generate_with_anthropic(
                question=question,
                context=context_text,
                max_tokens=max_tokens,
            )
        else:
            answer_text = self._generate_fallback(question, context_documents)
            tokens_used = 0

        processing_time = time.time() - start_time
        confidence = _estimate_confidence(answer_text, citations, len(context_documents))

        return FinancialAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            sub_queries=sub_queries or [],
            confidence_score=confidence,
            processing_time=processing_time,
            tokens_used=tokens_used,
            context_docs_count=len(context_documents),
        )

    def generate_stream(
        self,
        question: str,
        context_documents: List[Document],
        context_scores: Optional[List[float]] = None,
        max_tokens: int = 2000,
    ) -> Iterator[str]:
        """
        Génère une réponse en streaming (token par token).

        Args:
            question: Question de l'utilisateur.
            context_documents: Documents de contexte récupérés.
            context_scores: Scores de pertinence correspondants.
            max_tokens: Nombre max de tokens.

        Yields:
            Tokens de la réponse au fur et à mesure.
        """
        if not context_documents:
            yield "⚠️ Aucun document pertinent trouvé. Veuillez indexer des documents d'abord."
            return

        if not self._initialized or not self._client:
            answer = self._generate_fallback(question, context_documents)
            yield answer
            return

        context_text = _build_context(context_documents)

        try:
            import anthropic
            with self._client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"**Contexte documentaire :**\n\n{context_text}\n\n"
                            f"---\n\n**Question :** {question}"
                        ),
                    }
                ],
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Erreur streaming: {e}")
            yield f"\n\n⚠️ Erreur de génération: {e}"

    def _generate_with_anthropic(
        self,
        question: str,
        context: str,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Appel à l'API Anthropic."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"**Contexte documentaire :**\n\n{context}\n\n"
                            f"---\n\n**Question :** {question}"
                        ),
                    }
                ],
            )
            answer = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return answer, tokens

        except Exception as e:
            logger.error(f"Erreur API Anthropic: {e}")
            return self._generate_fallback_from_context(question, context), 0

    def _generate_fallback(self, question: str, documents: List[Document]) -> str:
        """Génération de fallback sans API (extraction directe du contexte)."""
        context = _build_context(documents, max_context_chars=3000)
        return self._generate_fallback_from_context(question, context)

    def _generate_fallback_from_context(self, question: str, context: str) -> str:
        """Retourne le contexte brut structuré quand l'API n'est pas disponible."""
        return (
            f"⚠️ **Mode dégradé** (API Anthropic non configurée)\n\n"
            f"**Question :** {question}\n\n"
            f"**Contexte disponible :**\n\n{context[:2000]}\n\n"
            "*Configurez ANTHROPIC_API_KEY dans votre fichier .env pour obtenir une réponse analysée.*"
        )
