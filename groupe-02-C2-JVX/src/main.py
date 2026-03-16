#!/usr/bin/env python3
"""
FinRAG — Point d'entrée CLI.
Commandes: ingest, query, evaluate, ui.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from loguru import logger
from src.config import settings


def setup_logging():
    """Configure le logging."""
    import os
    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <white>{message}</white>",
    )
    logger.add(
        str(log_dir / "finrag.log"),
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        encoding="utf-8",
    )


def _build_system():
    """Initialise et retourne le système RAG complet."""
    from src.retrieval.vector_store import FinancialVectorStore
    from src.retrieval.retriever import HybridFinancialRetriever
    from src.retrieval.reranker import CrossEncoderReRanker
    from src.generation.generator import FinancialAnswerGenerator
    from src.agents.rag_agent import FinancialRAGAgent

    logger.info("Initialisation du système FinRAG...")
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
    }


def cmd_ingest(args):
    """Commande: indexer des documents."""
    from src.ingestion.document_loader import FinancialDocumentLoader
    from src.ingestion.table_extractor import FinancialTableExtractor
    from src.ingestion.chunker import IntelligentFinancialChunker, ChunkingStrategy
    from tqdm import tqdm

    source_path = Path(args.source)
    if not source_path.exists():
        logger.error(f"Source introuvable: {source_path}")
        sys.exit(1)

    logger.info(f"Ingestion depuis: {source_path}")

    loader = FinancialDocumentLoader()
    all_docs = []

    if source_path.is_file():
        files = [source_path]
    else:
        supported_exts = {".pdf", ".csv", ".xlsx", ".json", ".txt"}
        files = [f for f in source_path.rglob("*") if f.suffix.lower() in supported_exts]

    logger.info(f"{len(files)} fichiers à traiter")

    for file_path in tqdm(files, desc="Chargement"):
        docs = loader.load(file_path)
        all_docs.extend(docs)

        # Extract tables from PDFs
        if file_path.suffix.lower() == ".pdf":
            extractor = FinancialTableExtractor()
            table_docs = extractor.extract_tables_from_pdf(file_path)
            all_docs.extend(table_docs)

    logger.info(f"{len(all_docs)} documents chargés")

    # Chunk
    chunker = IntelligentFinancialChunker(
        strategy=ChunkingStrategy.HYBRID,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = chunker.chunk_documents(all_docs)
    logger.info(f"{len(chunks)} chunks créés")

    # Index
    system = _build_system()
    added = system["vector_store"].add_documents(chunks)

    stats = system["vector_store"].get_stats()
    logger.success(
        f"Indexation terminée: {added} chunks ajoutés. "
        f"Total dans le store: {stats['total_chunks']} chunks, "
        f"{stats['total_sources']} sources."
    )


def cmd_query(args):
    """Commande: poser une question."""
    system = _build_system()
    agent = system["agent"]

    question = args.question
    logger.info(f"Question: {question}")

    answer = agent.answer(
        question=question,
        use_decomposition=not args.no_decompose,
    )

    print("\n" + "="*70)
    print(f"Q: {question}")
    print("="*70)
    print(answer.answer)
    print("-"*70)
    print(f"Confiance: {answer.confidence_score:.0%} | "
          f"Temps: {answer.processing_time:.1f}s | "
          f"Tokens: {answer.tokens_used:,} | "
          f"Sources: {answer.context_docs_count}")

    if answer.citations:
        print("\nCitations:")
        for c in answer.citations[:5]:
            print(f"  [{c.source_file}, p.{c.page_number}] {c.excerpt[:80]}...")

    if answer.sub_queries:
        print(f"\nSous-requêtes ({len(answer.sub_queries)}):")
        for sq in answer.sub_queries:
            print(f"  → {sq}")


def cmd_evaluate(args):
    """Commande: évaluation RAGAS."""
    from src.evaluation.evaluator import RAGASEvaluator

    system = _build_system()
    evaluator = RAGASEvaluator(
        agent=system["agent"],
        eval_dataset_path=args.dataset if args.dataset else None,
    )

    logger.info(f"Démarrage de l'évaluation ({args.max_samples} questions max)...")
    report = evaluator.evaluate_batch(
        max_samples=args.max_samples,
        save_report=True,
    )

    print("\n" + "="*70)
    print("RAPPORT D'ÉVALUATION RAGAS")
    print("="*70)
    print(f"Échantillons: {report.n_samples}")
    print(f"Faithfulness:     {report.avg_faithfulness:.3f}")
    print(f"Answer Relevancy: {report.avg_answer_relevancy:.3f}")
    print(f"Context Recall:   {report.avg_context_recall:.3f}")
    print(f"Context Precision:{report.avg_context_precision:.3f}")
    print(f"Score Global:     {report.overall_score:.3f}")
    print(f"Durée: {report.evaluation_time:.1f}s")
    print("="*70)
    print("Rapport sauvegardé dans docs/evaluation_report.md")


def cmd_ui(args):
    """Commande: lancer l'interface Streamlit."""
    import subprocess
    ui_path = ROOT_DIR / "src" / "ui" / "app.py"
    logger.info(f"Lancement de l'interface: {ui_path}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path)],
        check=True,
    )


def cmd_stats(args):
    """Commande: afficher les statistiques du vector store."""
    system = _build_system()
    stats = system["vector_store"].get_stats()

    print("\n" + "="*50)
    print("STATISTIQUES FINRAG")
    print("="*50)
    print(f"Total chunks:  {stats['total_chunks']:,}")
    print(f"Total sources: {stats['total_sources']}")
    print(f"Modèle:        {stats['embedding_model']}")
    print(f"Répertoire:    {stats['persist_dir']}")
    print("\nCollections:")
    for coll, count in stats.get("collections", {}).items():
        print(f"  {coll:<10}: {count:,} chunks")
    print("="*50)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="FinRAG — Système RAG pour l'Analyse Financière",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python src/main.py ingest --source data/samples/
  python src/main.py query "Quel est le CA d'Apple en 2023 ?"
  python src/main.py evaluate --max-samples 10
  python src/main.py ui
  python src/main.py stats
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Indexer des documents")
    p_ingest.add_argument("--source", required=True, help="Chemin du fichier ou répertoire")

    # query
    p_query = subparsers.add_parser("query", help="Poser une question")
    p_query.add_argument("question", help="Question financière")
    p_query.add_argument("--no-decompose", action="store_true", help="Désactive la décomposition")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Évaluation RAGAS")
    p_eval.add_argument("--max-samples", type=int, default=10, help="Nombre max de questions")
    p_eval.add_argument("--dataset", help="Chemin vers le dataset JSON (optionnel)")

    # ui
    p_ui = subparsers.add_parser("ui", help="Lancer l'interface Streamlit")

    # stats
    p_stats = subparsers.add_parser("stats", help="Statistiques du vector store")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "evaluate": cmd_evaluate,
        "ui": cmd_ui,
        "stats": cmd_stats,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur: {e}")
        raise


if __name__ == "__main__":
    main()
