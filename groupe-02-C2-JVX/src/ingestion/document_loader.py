"""
Chargeur de documents financiers multi-format.
Supporte PDF, CSV, XLSX, TXT, JSON, URLs (news JSON).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from langchain_core.documents import Document
from loguru import logger


# ─── Metadata helpers ────────────────────────────────────────────────────────

TICKER_PATTERN = re.compile(r'\b([A-Z]{2,5})\b')
KNOWN_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BRK",
    "JNJ", "UNH", "PFE", "ABBV", "LLY",
    "XOM", "CVX", "COP",
    "WMT", "PG", "KO", "PEP", "MCD",
    "SPY", "QQQ", "DIA", "IWM",
}

DATE_PATTERNS = [
    re.compile(r'\b(20\d{2})\b'),
    re.compile(r'\b(Q[1-4]\s+20\d{2})\b'),
    re.compile(r'\b(FY20\d{2})\b'),
]

DOCUMENT_TYPE_KEYWORDS = {
    "annual_report": ["annual report", "rapport annuel", "10-K", "annual"],
    "quarterly_report": ["quarterly", "trimestriel", "Q1", "Q2", "Q3", "Q4", "10-Q"],
    "news_article": ["article", "news", "press release", "communiqué"],
    "market_overview": ["overview", "market", "marché", "macro"],
    "research_report": ["research", "analyse", "analyst", "rating"],
    "csv_data": ["csv", "portfolio", "données"],
}


def _detect_document_type(content: str, filename: str) -> str:
    """Détecte automatiquement le type de document."""
    content_lower = content.lower()
    filename_lower = filename.lower()
    text = content_lower + " " + filename_lower

    best_type = "unknown"
    best_score = 0
    for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text)
        if score > best_score:
            best_score = score
            best_type = doc_type
    return best_type


def _extract_tickers(text: str) -> List[str]:
    """Extrait les symboles boursiers du texte."""
    found = set(TICKER_PATTERN.findall(text))
    return sorted(found & KNOWN_TICKERS)


def _extract_dates(text: str) -> Optional[str]:
    """Extrait la date/période principale du document."""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


# ─── Main Loader ─────────────────────────────────────────────────────────────

class FinancialDocumentLoader:
    """
    Chargeur universel de documents financiers.

    Supporte : PDF, CSV, XLSX, TXT, JSON, URLs (news JSON).
    Extrait les métadonnées : source, filename, date, document_type,
    ticker_symbols, page_count.
    Retourne des objets langchain.schema.Document enrichis.
    """

    def __init__(self, chunk_on_load: bool = False) -> None:
        self.chunk_on_load = chunk_on_load

    # ── Public API ───────────────────────────────────────────────────────────

    def load(self, source: str | Path) -> List[Document]:
        """
        Charge un document depuis un chemin fichier ou URL.

        Args:
            source: Chemin vers le fichier (str ou Path).

        Returns:
            Liste de Documents LangChain enrichis.
        """
        path = Path(source)
        ext = path.suffix.lower()

        logger.info(f"Chargement : {path.name} (type={ext})")

        loaders = {
            ".pdf": self._load_pdf,
            ".csv": self._load_csv,
            ".xlsx": self._load_xlsx,
            ".xls": self._load_xlsx,
            ".txt": self._load_txt,
            ".json": self._load_json,
        }

        loader_fn = loaders.get(ext)
        if loader_fn is None:
            logger.warning(f"Extension non supportée : {ext} — {path.name}")
            return []

        try:
            docs = loader_fn(path)
            logger.success(f"Chargé {len(docs)} documents depuis {path.name}")
            return docs
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {path.name}: {e}")
            return []

    def load_directory(self, directory: str | Path) -> List[Document]:
        """
        Charge tous les documents supportés depuis un répertoire.

        Args:
            directory: Chemin du répertoire à scanner.

        Returns:
            Liste consolidée de Documents LangChain.
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Répertoire introuvable : {directory}")
            return []

        all_docs: List[Document] = []
        supported_exts = {".pdf", ".csv", ".xlsx", ".xls", ".txt", ".json"}

        files = [f for f in directory.rglob("*") if f.suffix.lower() in supported_exts]
        logger.info(f"Répertoire {directory} : {len(files)} fichiers trouvés")

        for file_path in files:
            docs = self.load(file_path)
            all_docs.extend(docs)

        logger.info(f"Total chargé : {len(all_docs)} documents")
        return all_docs

    # ── Private Loaders ──────────────────────────────────────────────────────

    def _build_metadata(
        self,
        filename: str,
        content: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construit les métadonnées enrichies pour un document."""
        meta: Dict[str, Any] = {
            "source": filename,
            "filename": filename,
            "date": _extract_dates(content) or extra.get("date", "") if extra else _extract_dates(content) or "",
            "document_type": _detect_document_type(content, filename),
            "ticker_symbols": _extract_tickers(content),
            "page_count": 0,
            "ingestion_timestamp": datetime.utcnow().isoformat(),
        }
        if extra:
            meta.update(extra)
        return meta

    def _load_pdf(self, path: Path) -> List[Document]:
        """Charge un PDF avec pdfplumber (fallback PyMuPDF)."""
        docs: List[Document] = []

        # Try pdfplumber first
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                page_count = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue

                    meta = self._build_metadata(
                        filename=path.name,
                        content=text,
                        extra={
                            "page_number": page_num,
                            "page_count": page_count,
                            "document_type": _detect_document_type(text, path.name),
                        },
                    )
                    docs.append(Document(page_content=text, metadata=meta))

            return docs

        except ImportError:
            logger.warning("pdfplumber non disponible, tentative avec PyMuPDF")
        except Exception as e:
            logger.warning(f"pdfplumber échec sur {path.name}: {e}, tentative PyMuPDF")

        # Fallback: PyMuPDF
        try:
            import fitz  # PyMuPDF

            pdf_doc = fitz.open(str(path))
            page_count = len(pdf_doc)

            for page_num in range(page_count):
                page = pdf_doc[page_num]
                text = page.get_text()
                if not text.strip():
                    continue

                meta = self._build_metadata(
                    filename=path.name,
                    content=text,
                    extra={
                        "page_number": page_num + 1,
                        "page_count": page_count,
                    },
                )
                docs.append(Document(page_content=text, metadata=meta))

            pdf_doc.close()
            return docs

        except ImportError:
            logger.error("Ni pdfplumber ni PyMuPDF ne sont installés.")
            return []
        except Exception as e:
            logger.error(f"Impossible de charger le PDF {path.name}: {e}")
            return []

    def _load_csv(self, path: Path) -> List[Document]:
        """Charge un CSV et le convertit en document textuel."""
        try:
            df = pd.read_csv(path, encoding="utf-8", errors="replace")
            content = f"Dataset: {path.stem}\n\n"
            content += f"Colonnes: {', '.join(df.columns.tolist())}\n"
            content += f"Lignes: {len(df)}\n\n"
            content += "Aperçu des données:\n"
            content += df.head(50).to_string(index=False)

            if "ticker" in df.columns:
                tickers = df["ticker"].dropna().unique().tolist()
            else:
                tickers = _extract_tickers(content)

            meta = self._build_metadata(
                filename=path.name,
                content=content,
                extra={
                    "document_type": "csv_data",
                    "row_count": len(df),
                    "columns": df.columns.tolist(),
                    "ticker_symbols": tickers,
                },
            )
            return [Document(page_content=content, metadata=meta)]

        except Exception as e:
            logger.error(f"Erreur CSV {path.name}: {e}")
            return []

    def _load_xlsx(self, path: Path) -> List[Document]:
        """Charge un fichier Excel (toutes les feuilles)."""
        try:
            xl = pd.ExcelFile(path, engine="openpyxl")
            docs: List[Document] = []

            for sheet_name in xl.sheet_names:
                df = xl.parse(sheet_name)
                content = f"Fichier: {path.stem} | Feuille: {sheet_name}\n\n"
                content += f"Colonnes: {', '.join(str(c) for c in df.columns.tolist())}\n"
                content += f"Lignes: {len(df)}\n\n"
                content += df.head(50).to_string(index=False)

                meta = self._build_metadata(
                    filename=path.name,
                    content=content,
                    extra={
                        "document_type": "excel_data",
                        "sheet_name": sheet_name,
                        "row_count": len(df),
                    },
                )
                docs.append(Document(page_content=content, metadata=meta))

            return docs

        except Exception as e:
            logger.error(f"Erreur XLSX {path.name}: {e}")
            return []

    def _load_txt(self, path: Path) -> List[Document]:
        """Charge un fichier texte brut."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            meta = self._build_metadata(filename=path.name, content=content)
            return [Document(page_content=content, metadata=meta)]
        except Exception as e:
            logger.error(f"Erreur TXT {path.name}: {e}")
            return []

    def _load_json(self, path: Path) -> List[Document]:
        """Charge un fichier JSON (article de news ou données)."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # News article format
            if isinstance(data, dict) and "title" in data and "content" in data:
                content = f"Titre: {data.get('title', '')}\n\n"
                content += f"Source: {data.get('source', '')}\n"
                content += f"Date: {data.get('date', '')}\n"
                content += f"Ticker: {data.get('ticker', '')}\n\n"
                content += data.get("content", "")

                meta = self._build_metadata(
                    filename=path.name,
                    content=content,
                    extra={
                        "document_type": "news_article",
                        "date": data.get("date", ""),
                        "ticker_symbols": [data["ticker"]] if data.get("ticker") else [],
                        "news_source": data.get("source", ""),
                        "news_title": data.get("title", ""),
                        "category": data.get("category", ""),
                        "url": data.get("url", ""),
                    },
                )
                return [Document(page_content=content, metadata=meta)]

            # List of articles
            elif isinstance(data, list):
                docs: List[Document] = []
                for item in data:
                    if isinstance(item, dict) and "content" in item:
                        content = json.dumps(item, ensure_ascii=False, indent=2)
                        meta = self._build_metadata(filename=path.name, content=content)
                        docs.append(Document(page_content=content, metadata=meta))
                return docs

            # Generic JSON
            else:
                content = json.dumps(data, ensure_ascii=False, indent=2)
                meta = self._build_metadata(filename=path.name, content=content)
                return [Document(page_content=content, metadata=meta)]

        except Exception as e:
            logger.error(f"Erreur JSON {path.name}: {e}")
            return []
