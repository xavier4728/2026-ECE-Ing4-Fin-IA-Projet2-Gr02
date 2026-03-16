"""
Évaluateur RAGAS pour le système FinRAG.
Métriques : faithfulness, answer_relevancy, context_recall, context_precision.
Génère un rapport en markdown + CSV.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import settings


# ─── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    """Un échantillon d'évaluation RAGAS."""
    question: str
    ground_truth: str
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0


@dataclass
class EvaluationReport:
    """Rapport d'évaluation complet."""
    timestamp: str
    n_samples: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_recall: float
    avg_context_precision: float
    overall_score: float
    samples: List[EvalSample] = field(default_factory=list)
    evaluation_time: float = 0.0

    def to_markdown(self) -> str:
        """Génère le rapport en markdown."""
        lines = [
            "# Rapport d'Évaluation RAGAS — FinRAG",
            f"\n**Date** : {self.timestamp}",
            f"**Échantillons évalués** : {self.n_samples}",
            f"**Durée** : {self.evaluation_time:.1f}s",
            "\n## 📊 Métriques Globales",
            "",
            "| Métrique | Score | Interprétation |",
            "|----------|-------|----------------|",
            f"| Faithfulness | **{self.avg_faithfulness:.3f}** | {'✅ Excellent' if self.avg_faithfulness >= 0.8 else '⚠️ À améliorer' if self.avg_faithfulness >= 0.6 else '❌ Insuffisant'} |",
            f"| Answer Relevancy | **{self.avg_answer_relevancy:.3f}** | {'✅ Excellent' if self.avg_answer_relevancy >= 0.8 else '⚠️ À améliorer' if self.avg_answer_relevancy >= 0.6 else '❌ Insuffisant'} |",
            f"| Context Recall | **{self.avg_context_recall:.3f}** | {'✅ Excellent' if self.avg_context_recall >= 0.8 else '⚠️ À améliorer' if self.avg_context_recall >= 0.6 else '❌ Insuffisant'} |",
            f"| Context Precision | **{self.avg_context_precision:.3f}** | {'✅ Excellent' if self.avg_context_precision >= 0.8 else '⚠️ À améliorer' if self.avg_context_precision >= 0.6 else '❌ Insuffisant'} |",
            f"| **Score Global** | **{self.overall_score:.3f}** | {'✅ Excellent' if self.overall_score >= 0.8 else '⚠️ Acceptable' if self.overall_score >= 0.6 else '❌ Insuffisant'} |",
            "",
            "## 📝 Détail par Question",
            "",
        ]

        for i, sample in enumerate(self.samples, 1):
            lines.extend([
                f"### Q{i}: {sample.question[:80]}{'...' if len(sample.question) > 80 else ''}",
                f"- **Faithfulness** : {sample.faithfulness:.3f}",
                f"- **Answer Relevancy** : {sample.answer_relevancy:.3f}",
                f"- **Context Recall** : {sample.context_recall:.3f}",
                f"- **Context Precision** : {sample.context_precision:.3f}",
                "",
            ])

        lines.extend([
            "## 💡 Recommandations",
            "",
        ])

        if self.avg_faithfulness < 0.7:
            lines.append("- **Faithfulness faible** : Le système génère des affirmations non fondées sur le contexte. Renforcer le prompt système avec des instructions de citation strictes.")
        if self.avg_answer_relevancy < 0.7:
            lines.append("- **Relevancy faible** : Les réponses ne répondent pas précisément aux questions. Améliorer le prompt de génération.")
        if self.avg_context_recall < 0.7:
            lines.append("- **Context Recall faible** : Le retrieval ne récupère pas tous les chunks pertinents. Augmenter TOP_K_RETRIEVAL ou améliorer l'embedding.")
        if self.avg_context_precision < 0.7:
            lines.append("- **Context Precision faible** : Trop de chunks non pertinents sont récupérés. Améliorer le re-ranking ou les filtres.")

        if self.overall_score >= 0.8:
            lines.append("✅ **Excellent système RAG** — Toutes les métriques sont au-dessus du seuil de qualité.")

        return "\n".join(lines)


# ─── Main Evaluator ───────────────────────────────────────────────────────────

class RAGASEvaluator:
    """
    Évaluateur RAGAS pour le système FinRAG.

    Métriques calculées :
    - Faithfulness : les réponses sont-elles fidèles au contexte ?
    - Answer Relevancy : les réponses répondent-elles aux questions ?
    - Context Recall : le contexte contient-il les informations nécessaires ?
    - Context Precision : le contexte récupéré est-il précis ?

    Supporte l'évaluation batch (dataset JSON) et single-query.
    Exporte les résultats vers docs/evaluation_report.md et CSV.
    """

    def __init__(
        self,
        agent=None,
        eval_dataset_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            agent: Instance FinancialRAGAgent pour générer les réponses.
            eval_dataset_path: Chemin vers le dataset JSON de questions/réponses.
        """
        self._agent = agent
        self._dataset_path = eval_dataset_path or str(
            Path(__file__).parent.parent.parent / "data" / "samples" / "eval_questions.json"
        )
        self._use_ragas_lib = False
        self._ragas_available = self._check_ragas()

    def _check_ragas(self) -> bool:
        """Vérifie si le package ragas est installé et fonctionnel."""
        try:
            import ragas
            self._use_ragas_lib = True
            logger.info("Package ragas disponible")
            return True
        except ImportError:
            logger.warning("Package ragas non installé. Utilisation de métriques approchées.")
            return False

    def load_eval_dataset(self) -> List[Dict[str, Any]]:
        """Charge le dataset d'évaluation depuis le fichier JSON."""
        dataset_path = Path(self._dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset d'évaluation introuvable: {dataset_path}")
            return []

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Dataset chargé: {len(data)} questions")
        return data

    def evaluate_batch(
        self,
        max_samples: int = 10,
        save_report: bool = True,
    ) -> EvaluationReport:
        """
        Évalue le système sur le dataset d'évaluation complet.

        Args:
            max_samples: Nombre maximum d'échantillons à évaluer.
            save_report: Sauvegarde le rapport dans docs/evaluation_report.md.

        Returns:
            EvaluationReport avec toutes les métriques.
        """
        start_time = time.time()
        dataset = self.load_eval_dataset()

        if not dataset:
            logger.error("Dataset vide ou introuvable")
            return self._empty_report()

        if not self._agent:
            logger.error("Agent non configuré pour l'évaluation")
            return self._empty_report()

        samples_to_eval = dataset[:max_samples]
        logger.info(f"Évaluation de {len(samples_to_eval)} questions...")

        eval_samples: List[EvalSample] = []

        for i, item in enumerate(samples_to_eval):
            question = item.get("question", "")
            ground_truth = item.get("ground_truth", "")

            if not question:
                continue

            logger.info(f"Évaluation {i+1}/{len(samples_to_eval)}: {question[:60]}...")

            try:
                # Get agent answer
                answer_obj = self._agent.answer(question=question, use_decomposition=False)
                answer_text = answer_obj.answer
                contexts = [c.excerpt for c in answer_obj.citations] if answer_obj.citations else []

                # Compute metrics
                sample = EvalSample(
                    question=question,
                    ground_truth=ground_truth,
                    answer=answer_text,
                    contexts=contexts,
                )

                if self._use_ragas_lib and settings.use_openai_embeddings:
                    self._compute_ragas_metrics(sample)
                else:
                    self._compute_approximate_metrics(sample)

                eval_samples.append(sample)

            except Exception as e:
                logger.error(f"Erreur évaluation Q{i+1}: {e}")
                continue

        if not eval_samples:
            return self._empty_report()

        # Aggregate metrics
        report = self._build_report(eval_samples, time.time() - start_time)

        if save_report:
            self._save_report(report)

        return report

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvalSample:
        """
        Évalue une seule réponse.

        Args:
            question: Question posée.
            answer: Réponse générée.
            contexts: Contextes utilisés pour la réponse.
            ground_truth: Réponse de référence (optionnel).

        Returns:
            EvalSample avec les métriques calculées.
        """
        sample = EvalSample(
            question=question,
            ground_truth=ground_truth or "",
            answer=answer,
            contexts=contexts,
        )

        if self._use_ragas_lib and settings.use_openai_embeddings:
            self._compute_ragas_metrics(sample)
        else:
            self._compute_approximate_metrics(sample)

        return sample

    def _compute_ragas_metrics(self, sample: EvalSample) -> None:
        """Calcule les métriques RAGAS via la bibliothèque ragas."""
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
            from datasets import Dataset

            data = {
                "question": [sample.question],
                "answer": [sample.answer],
                "contexts": [sample.contexts if sample.contexts else [""]],
                "ground_truth": [sample.ground_truth],
            }
            dataset = Dataset.from_dict(data)

            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            )

            sample.faithfulness = float(result.get("faithfulness", 0) or 0)
            sample.answer_relevancy = float(result.get("answer_relevancy", 0) or 0)
            sample.context_recall = float(result.get("context_recall", 0) or 0)
            sample.context_precision = float(result.get("context_precision", 0) or 0)

        except Exception as e:
            logger.warning(f"RAGAS lib erreur: {e}, fallback métriques approchées")
            self._compute_approximate_metrics(sample)

    def _compute_approximate_metrics(self, sample: EvalSample) -> None:
        """
        Calcule des métriques approchées sans API externe.
        Basé sur la similarité de termes (BM25-like).
        """
        import re

        def tokenize(text: str) -> set:
            return set(re.findall(r'\b\w+\b', text.lower()))

        def jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        q_tokens = tokenize(sample.question)
        a_tokens = tokenize(sample.answer)
        gt_tokens = tokenize(sample.ground_truth) if sample.ground_truth else set()
        ctx_tokens = tokenize(" ".join(sample.contexts)) if sample.contexts else set()

        # Faithfulness: overlap between answer and contexts
        if ctx_tokens:
            sample.faithfulness = min(1.0, len(a_tokens & ctx_tokens) / max(len(a_tokens), 1) * 2)
        else:
            sample.faithfulness = 0.3  # No context available

        # Answer Relevancy: how much the answer addresses the question
        sample.answer_relevancy = min(1.0, jaccard(q_tokens, a_tokens) * 3)

        # Context Recall: overlap between ground truth and contexts
        if gt_tokens and ctx_tokens:
            sample.context_recall = min(1.0, len(gt_tokens & ctx_tokens) / max(len(gt_tokens), 1) * 2)
        else:
            sample.context_recall = 0.5

        # Context Precision: relevance of contexts to the question
        if ctx_tokens:
            sample.context_precision = min(1.0, jaccard(q_tokens, ctx_tokens) * 4)
        else:
            sample.context_precision = 0.0

    def _build_report(
        self, samples: List[EvalSample], elapsed: float
    ) -> EvaluationReport:
        """Construit le rapport d'évaluation."""
        n = len(samples)

        avg_faithfulness = sum(s.faithfulness for s in samples) / n
        avg_answer_relevancy = sum(s.answer_relevancy for s in samples) / n
        avg_context_recall = sum(s.context_recall for s in samples) / n
        avg_context_precision = sum(s.context_precision for s in samples) / n
        overall = (avg_faithfulness + avg_answer_relevancy + avg_context_recall + avg_context_precision) / 4

        return EvaluationReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n_samples=n,
            avg_faithfulness=avg_faithfulness,
            avg_answer_relevancy=avg_answer_relevancy,
            avg_context_recall=avg_context_recall,
            avg_context_precision=avg_context_precision,
            overall_score=overall,
            samples=samples,
            evaluation_time=elapsed,
        )

    def _empty_report(self) -> EvaluationReport:
        """Retourne un rapport vide en cas d'erreur."""
        return EvaluationReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            n_samples=0,
            avg_faithfulness=0.0,
            avg_answer_relevancy=0.0,
            avg_context_recall=0.0,
            avg_context_precision=0.0,
            overall_score=0.0,
        )

    def _save_report(self, report: EvaluationReport) -> None:
        """Sauvegarde le rapport en markdown et CSV."""
        docs_dir = Path(__file__).parent.parent.parent / "docs"
        docs_dir.mkdir(exist_ok=True)

        # Markdown report
        md_path = docs_dir / "evaluation_report.md"
        md_path.write_text(report.to_markdown(), encoding="utf-8")
        logger.info(f"Rapport markdown sauvegardé: {md_path}")

        # CSV report
        csv_path = docs_dir / "evaluation_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question", "faithfulness", "answer_relevancy",
                "context_recall", "context_precision", "overall",
            ])
            writer.writeheader()
            for s in report.samples:
                overall = (s.faithfulness + s.answer_relevancy + s.context_recall + s.context_precision) / 4
                writer.writerow({
                    "question": s.question[:100],
                    "faithfulness": round(s.faithfulness, 4),
                    "answer_relevancy": round(s.answer_relevancy, 4),
                    "context_recall": round(s.context_recall, 4),
                    "context_precision": round(s.context_precision, 4),
                    "overall": round(overall, 4),
                })
        logger.info(f"Rapport CSV sauvegardé: {csv_path}")
