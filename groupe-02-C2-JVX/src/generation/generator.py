"""
Générateur de réponses financières avec OpenAI GPT-4o.
Citations vérifiables, format structuré, streaming support.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Iterator

from langchain_core.documents import Document
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
    answer: str
    citations: List[Citation] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    tokens_used: int = 0
    context_docs_count: int = 0

    def to_dict(self) -> dict:
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
    import re
    score = 0.0
    if citations:
        score += 0.3
    if len(citations) >= 3:
        score += 0.2
    if re.search(r'\d+[\.,]\d+|\d+\s*(?:Md|M|%)', answer):
        score += 0.2
    if n_docs >= 3:
        score += 0.2
    elif n_docs >= 1:
        score += 0.1
    if "n'est pas disponible" not in answer.lower() and "pas dans le contexte" not in answer.lower():
        score += 0.1
    return min(1.0, score)


# ─── Main Generator ───────────────────────────────────────────────────────────

# Modèle OpenAI utilisé pour la génération
OPENAI_GENERATION_MODEL = "gpt-4o"


class FinancialAnswerGenerator:
    """
    Générateur de réponses financières via OpenAI GPT-4o.

    Utilise la même clé API (OPENAI_API_KEY) que le pipeline d'embeddings.
    Fallback automatique si la clé est absente : retourne le contexte brut.
    """

    def __init__(self) -> None:
        self._client = None
        self._model = OPENAI_GENERATION_MODEL
        self._initialized = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialise le client OpenAI."""
        if not settings.use_openai_embeddings:
            logger.warning(
                "OPENAI_API_KEY non défini. "
                "Le générateur fonctionnera en mode dégradé (contexte brut)."
            )
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self._initialized = True
            logger.info(f"Générateur OpenAI initialisé : {self._model}")
        except ImportError:
            logger.error("Package 'openai' non installé. pip install openai")
        except Exception as e:
            logger.error(f"Erreur initialisation OpenAI: {e}")

    def generate(
        self,
        question: str,
        context_documents: List[Document],
        context_scores: Optional[List[float]] = None,
        sub_queries: Optional[List[str]] = None,
        max_tokens: int = 2000,
    ) -> FinancialAnswer:
        """Génère une réponse financière à partir du contexte récupéré."""
        start_time = time.time()

        if not context_documents:
            return FinancialAnswer(
                question=question,
                answer="⚠️ Aucun document pertinent trouvé dans la base de connaissances. "
                       "Veuillez d'abord indexer des documents financiers.",
                confidence_score=0.0,
                processing_time=time.time() - start_time,
            )

        context_text = _build_context(context_documents)
        citations = _extract_citations(context_documents, context_scores)

        if self._initialized and self._client:
            answer_text, tokens_used = self._generate_with_openai(
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
        """Génère une réponse en streaming (token par token)."""
        if not context_documents:
            yield "⚠️ Aucun document pertinent trouvé. Veuillez indexer des documents d'abord."
            return

        if not self._initialized or not self._client:
            yield self._generate_fallback(question, context_documents)
            return

        context_text = _build_context(context_documents)
        user_content = (
            f"**Contexte documentaire :**\n\n{context_text}\n\n"
            f"---\n\n**Question :** {question}"
        )

        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                stream=True,
            )
            for chunk in stream:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    yield delta_content

        except Exception as e:
            error_type = type(e).__name__
            error_str = str(e).lower()

            if "timeout" in error_str or "Timeout" in error_type:
                logger.error(f"Timeout streaming OpenAI après {settings.REQUEST_TIMEOUT}s")
                yield "\n\n⚠️ Délai dépassé : la génération a pris trop de temps. Réessayez."
            elif "ratelimit" in error_str or "rate_limit" in error_str or "RateLimit" in error_type:
                logger.warning("Rate limit OpenAI atteint")
                yield "\n\n⚠️ Limite de requêtes atteinte. Attendez quelques secondes et réessayez."
            elif "auth" in error_str or "Authentication" in error_type or "Unauthorized" in error_type:
                logger.error("Erreur d'authentification OpenAI — vérifiez OPENAI_API_KEY")
                yield "\n\n⚠️ Erreur d'authentification. Vérifiez votre OPENAI_API_KEY dans .env."
            elif "insufficient_quota" in error_str:
                logger.error("Quota OpenAI insuffisant")
                yield "\n\n⚠️ Quota OpenAI insuffisant. Vérifiez votre facturation sur platform.openai.com."
            else:
                logger.error(f"Erreur streaming OpenAI: {error_type}: {e}")
                yield f"\n\n⚠️ Erreur de génération ({error_type}). Voir les logs pour le détail."

    def _generate_with_openai(
        self,
        question: str,
        context: str,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Appel synchrone à l'API OpenAI Chat Completions."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"**Contexte documentaire :**\n\n{context}\n\n"
                            f"---\n\n**Question :** {question}"
                        ),
                    },
                ],
            )
            answer = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            return answer, tokens

        except Exception as e:
            logger.error(f"Erreur API OpenAI: {e}")
            return self._generate_fallback_from_context(question, context), 0

    def _generate_fallback(self, question: str, documents: List[Document]) -> str:
        context = _build_context(documents, max_context_chars=3000)
        return self._generate_fallback_from_context(question, context)

    def _generate_fallback_from_context(self, question: str, context: str) -> str:
        return (
            f"⚠️ **Mode dégradé** (OPENAI_API_KEY non configurée)\n\n"
            f"**Question :** {question}\n\n"
            f"**Contexte disponible :**\n\n{context[:2000]}\n\n"
            "*Configurez OPENAI_API_KEY dans votre fichier .env pour obtenir une réponse analysée.*"
        )