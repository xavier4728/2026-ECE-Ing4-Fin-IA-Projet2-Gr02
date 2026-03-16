#!/usr/bin/env python3
"""
fix_langchain_imports.py
Corrige tous les imports LangChain cassés suite au passage à LangChain 1.x.

Breaking changes :
  - langchain.schema.Document         → langchain_core.documents.Document
  - langchain.text_splitter.*         → langchain_text_splitters.*

Usage :
    python fix_langchain_imports.py          # depuis la racine du projet
    python fix_langchain_imports.py --dry    # aperçu sans modifier les fichiers
"""

import argparse
import re
import sys
from pathlib import Path

# ─── Substitutions à effectuer ───────────────────────────────────────────────
# (pattern_regex, replacement, description)
SUBSTITUTIONS = [
    (
        r"from langchain\.schema import Document",
        "from langchain_core.documents import Document",
        "langchain.schema → langchain_core.documents",
    ),
    (
        r"from langchain\.schema import (.+)",
        r"from langchain_core.schema import \1",
        "langchain.schema (autres) → langchain_core.schema",
    ),
    (
        r"from langchain\.text_splitter import RecursiveCharacterTextSplitter",
        "from langchain_text_splitters import RecursiveCharacterTextSplitter",
        "langchain.text_splitter → langchain_text_splitters",
    ),
    (
        r"from langchain\.text_splitter import (.+)",
        r"from langchain_text_splitters import \1",
        "langchain.text_splitter (générique) → langchain_text_splitters",
    ),
]

# Fichiers Python à traiter (relatifs à la racine du projet)
TARGET_FILES = [
    "src/ingestion/document_loader.py",
    "src/ingestion/chunker.py",
    "src/ingestion/table_extractor.py",
    "src/retrieval/vector_store.py",
    "src/retrieval/retriever.py",
    "src/retrieval/reranker.py",
    "src/agents/rag_agent.py",
    "src/generation/generator.py",
    "tests/conftest.py",
    "tests/test_agents.py",
    "tests/test_generation.py",
    "tests/test_ingestion.py",
    "tests/test_retrieval.py",
]


def fix_file(path: Path, dry_run: bool) -> bool:
    """Applique toutes les substitutions sur un fichier. Retourne True si modifié."""
    if not path.exists():
        print(f"  ⚠️  Ignoré (introuvable) : {path}")
        return False

    original = path.read_text(encoding="utf-8")
    content = original

    changes = []
    for pattern, replacement, desc in SUBSTITUTIONS:
        new_content, n = re.subn(pattern, replacement, content)
        if n > 0:
            changes.append((desc, n))
            content = new_content

    if not changes:
        return False

    print(f"  ✅ {path}")
    for desc, n in changes:
        print(f"       {n}× {desc}")

    if not dry_run:
        path.write_text(content, encoding="utf-8")

    return True


def main():
    parser = argparse.ArgumentParser(description="Corrige les imports LangChain 1.x")
    parser.add_argument("--dry", action="store_true", help="Aperçu sans modifier les fichiers")
    args = parser.parse_args()

    # Chercher la racine du projet (là où se trouve src/)
    root = Path(__file__).parent
    src_dir = root / "src"
    if not src_dir.exists():
        # Script lancé depuis un autre dossier ?
        root = Path.cwd()
        src_dir = root / "src"
    if not src_dir.exists():
        print("❌ Impossible de trouver le répertoire src/.")
        print("   Lancez ce script depuis la racine du projet groupe-02-C2-JVX/")
        sys.exit(1)

    mode = "APERÇU (--dry)" if args.dry else "MODIFICATION"
    print(f"\n{'='*60}")
    print(f"  Fix LangChain imports — mode : {mode}")
    print(f"  Racine projet : {root}")
    print(f"{'='*60}\n")

    modified = 0
    for rel_path in TARGET_FILES:
        abs_path = root / rel_path
        if fix_file(abs_path, dry_run=args.dry):
            modified += 1

    print(f"\n{'='*60}")
    if args.dry:
        print(f"  {modified} fichier(s) seraient modifiés (--dry, rien n'a changé).")
        print("  Relancez sans --dry pour appliquer.")
    else:
        print(f"  ✅ {modified} fichier(s) corrigés.")
        if modified > 0:
            print("\n  Prochaine étape :")
            print("    python src/main.py ingest --source data/samples/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()