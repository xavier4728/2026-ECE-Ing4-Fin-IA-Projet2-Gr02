"""
Extracteur de tableaux financiers depuis les PDFs.
Utilise camelot (lattice + stream) avec fallback pdfplumber.
Convertit les tableaux en markdown structuré lisible par LLM.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
from langchain.schema import Document
from loguru import logger


# ─── Helpers ─────────────────────────────────────────────────────────────────

FINANCIAL_UNITS = re.compile(
    r'(\d[\d\s,\.]+)\s*(M€|Md€|M\$|Md\$|%|bp|K€|K\$|billion|million|trillion)',
    re.IGNORECASE,
)


def _clean_cell(cell: str) -> str:
    """Nettoie le contenu d'une cellule de tableau."""
    if not isinstance(cell, str):
        return str(cell)
    cleaned = cell.strip().replace('\n', ' ').replace('  ', ' ')
    return cleaned


def _df_to_markdown(df: pd.DataFrame, title: str = "") -> str:
    """Convertit un DataFrame en tableau markdown avec titre."""
    if df.empty:
        return ""

    lines: List[str] = []
    if title:
        lines.append(f"### {title}")
        lines.append("")

    # FIX ÉLEVÉ : applymap() supprimé en pandas ≥ 2.1, utiliser map() ou applymap selon version
    # Compatibilité pandas 2.0 (applymap) ET pandas 2.1+ (map)
    _apply = df.map if hasattr(df, "map") and callable(getattr(df, "map")) else df.applymap
    df = _apply(lambda x: _clean_cell(str(x)) if pd.notna(x) else "")

    # Header
    headers = [str(col) for col in df.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Rows
    for _, row in df.iterrows():
        cells = [str(v) for v in row.values]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _infer_table_title(page_text: str, table_bbox: Optional[Tuple] = None) -> str:
    """Tente d'inférer le titre d'un tableau depuis le texte de la page."""
    if not page_text:
        return "Tableau Financier"

    title_patterns = [
        re.compile(
            r'((?:Tableau|Table|Compte de résultat|Bilan|Cash[- ]flow|Revenus?|'
            r'Résultats?|Performance|Summary|Overview|Ratios?)[^\n]{0,80})',
            re.I,
        ),
        re.compile(r'^([A-Z][^\n]{10,60})$', re.MULTILINE),
    ]

    for pattern in title_patterns:
        match = pattern.search(page_text[:500])
        if match:
            return match.group(1).strip()

    return "Tableau Financier"


# ─── Main Extractor ───────────────────────────────────────────────────────────

class FinancialTableExtractor:
    """
    Extracteur de tableaux financiers depuis les PDFs.

    Stratégie :
    1. camelot lattice (tableaux avec bordures nettes)
    2. camelot stream (tableaux sans bordures)
    3. pdfplumber extract_tables (fallback)

    Convertit en markdown structuré avec métadonnées de position.
    Gère les tableaux multi-pages.
    """

    def __init__(
        self,
        min_rows: int = 2,
        min_cols: int = 2,
        accuracy_threshold: float = 80.0,
    ) -> None:
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.accuracy_threshold = accuracy_threshold

    def extract_tables_from_pdf(self, pdf_path: str | Path) -> List[Document]:
        """
        Extrait tous les tableaux d'un PDF et retourne des Documents LangChain.
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Extraction tableaux : {pdf_path.name}")

        docs: List[Document] = []

        camelot_docs = self._extract_with_camelot(pdf_path)
        if camelot_docs:
            docs.extend(camelot_docs)
        else:
            logger.info(f"Fallback pdfplumber pour {pdf_path.name}")
            docs.extend(self._extract_with_pdfplumber(pdf_path))

        logger.info(f"Tableaux extraits : {len(docs)} depuis {pdf_path.name}")
        return docs

    def _extract_with_camelot(self, pdf_path: Path) -> List[Document]:
        """Extraction via camelot (lattice puis stream)."""
        try:
            import camelot
        except ImportError:
            logger.warning("camelot non installé, skip")
            return []

        docs: List[Document] = []
        filename = pdf_path.name

        for flavor in ("lattice", "stream"):
            try:
                tables = camelot.read_pdf(
                    str(pdf_path),
                    pages="all",
                    flavor=flavor,
                )
                logger.debug(f"camelot {flavor}: {tables.n} tableaux trouvés")

                for i, table in enumerate(tables):
                    if table.accuracy < self.accuracy_threshold:
                        continue

                    df = table.df
                    if df.shape[0] < self.min_rows or df.shape[1] < self.min_cols:
                        continue

                    if self._looks_like_header(df.iloc[0].tolist()):
                        df.columns = df.iloc[0]
                        df = df[1:].reset_index(drop=True)

                    title = f"Tableau {i+1} (page {table.page})"
                    markdown = _df_to_markdown(df, title=title)

                    if not markdown:
                        continue

                    doc = Document(
                        page_content=markdown,
                        metadata={
                            "source": filename,
                            "filename": filename,
                            "document_type": "financial_table",
                            "table_index": i,
                            "page_number": table.page,
                            "extraction_method": f"camelot_{flavor}",
                            "accuracy": table.accuracy,
                            "contains_table": True,
                            "contains_numbers": True,
                            "shape": f"{df.shape[0]}x{df.shape[1]}",
                            "ticker_symbols": [],
                        },
                    )
                    docs.append(doc)

                if docs:
                    return docs

            except Exception as e:
                logger.debug(f"camelot {flavor} échec: {e}")
                continue

        return docs

    def _extract_with_pdfplumber(self, pdf_path: Path) -> List[Document]:
        """Extraction via pdfplumber (fallback)."""
        try:
            import pdfplumber
        except ImportError:
            logger.error("pdfplumber non installé")
            return []

        docs: List[Document] = []
        filename = pdf_path.name

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    page_text = page.extract_text() or ""

                    for table_idx, table_data in enumerate(tables):
                        if not table_data or len(table_data) < self.min_rows:
                            continue

                        try:
                            df = pd.DataFrame(table_data)
                            if df.shape[1] < self.min_cols:
                                continue

                            if self._looks_like_header(df.iloc[0].tolist()):
                                df.columns = [
                                    str(v) if v else f"Col_{idx}"
                                    for idx, v in enumerate(df.iloc[0])
                                ]
                                df = df[1:].reset_index(drop=True)

                            title = _infer_table_title(page_text)
                            markdown = _df_to_markdown(df, title=f"{title} (p.{page_num})")

                            if not markdown:
                                continue

                            doc = Document(
                                page_content=markdown,
                                metadata={
                                    "source": filename,
                                    "filename": filename,
                                    "document_type": "financial_table",
                                    "table_index": table_idx,
                                    "page_number": page_num,
                                    "extraction_method": "pdfplumber",
                                    "contains_table": True,
                                    "contains_numbers": True,
                                    "shape": f"{df.shape[0]}x{df.shape[1]}",
                                    "ticker_symbols": [],
                                },
                            )
                            docs.append(doc)

                        except Exception as e:
                            logger.debug(f"Table parse error p.{page_num}: {e}")
                            continue

        except Exception as e:
            logger.error(f"pdfplumber échec {filename}: {e}")

        return docs

    def _looks_like_header(self, row: List[Any]) -> bool:
        """Heuristique pour détecter si une ligne est un en-tête."""
        if not row:
            return False
        str_row = [str(c) for c in row if c]
        if not str_row:
            return False
        numeric_count = sum(1 for c in str_row if re.match(r'^[\d\s,\.\-\(\)]+$', c))
        return numeric_count < len(str_row) * 0.5

    def tables_to_context(self, docs: List[Document]) -> str:
        """Agrège les tableaux extraits en un contexte textuel pour le LLM."""
        if not docs:
            return "Aucun tableau financier trouvé."

        parts = [f"**Tableaux financiers extraits ({len(docs)} tableaux)**\n"]
        for doc in docs:
            meta = doc.metadata
            parts.append(
                f"\n---\n*Source: {meta.get('filename', 'unknown')}, "
                f"Page {meta.get('page_number', '?')}*\n"
            )
            parts.append(doc.page_content)

        return "\n".join(parts)