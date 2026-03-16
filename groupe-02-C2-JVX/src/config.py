"""Configuration centralisée du système FinRAG via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """Configuration globale du système FinRAG.

    Toutes les valeurs peuvent être surchargées via variables d'environnement
    ou le fichier .env à la racine du projet.
    """

    # FIX ÉLEVÉ : utilisation de model_config au lieu de class Config (déprécié en pydantic-settings v2)
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",       # ignore les variables d'env inconnues
    )

    # === LLM ===
    OPENAI_API_KEY: str = Field(default="", description="Clé API OpenAI pour les embeddings")
    ANTHROPIC_API_KEY: str = Field(default="", description="Clé API Anthropic pour la génération")
    LLM_MODEL: str = Field(default="claude-sonnet-4-20250514", description="Modèle LLM pour la génération")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Modèle d'embeddings OpenAI")
    FALLBACK_EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Modèle d'embeddings local (fallback sans API OpenAI)",
    )

    # === RAG ===
    CHUNK_SIZE: int = Field(default=512, description="Taille des chunks en tokens")
    CHUNK_OVERLAP: int = Field(default=51, description="Overlap entre chunks (10%)")
    TOP_K_RETRIEVAL: int = Field(default=10, description="Nombre de chunks récupérés après RRF")
    TOP_K_RERANK: int = Field(default=5, description="Nombre de chunks après re-ranking")
    HYBRID_ALPHA: float = Field(default=0.7, description="Poids dense vs sparse (0=sparse, 1=dense)")
    TIME_DECAY_FACTOR: float = Field(default=0.1, description="Facteur de décroissance temporelle")
    BM25_TOP_K: int = Field(default=20, description="Top-k pour BM25 avant fusion")
    DENSE_TOP_K: int = Field(default=20, description="Top-k pour dense retrieval avant fusion")

    # === BM25 ===
    # FIX ÉLEVÉ : plafond mémoire BM25 pour éviter OOM sur gros corpus
    BM25_MAX_DOCS: int = Field(default=10_000, description="Nombre max de docs chargés en RAM pour BM25")

    # === ChromaDB ===
    CHROMA_PERSIST_DIR: str = Field(default="./data/chroma_db", description="Répertoire de persistance ChromaDB")
    COLLECTION_PREFIX: str = Field(default="finrag", description="Préfixe des collections ChromaDB")

    # === UI ===
    MAX_HISTORY_MESSAGES: int = Field(default=50, description="Nombre max de messages dans l'historique")
    STREAM_RESPONSE: bool = Field(default=True, description="Active le streaming des réponses")

    # === Performance ===
    REQUEST_TIMEOUT: int = Field(default=30, description="Timeout des requêtes en secondes")
    EMBEDDING_CACHE_DIR: str = Field(default="./data/embedding_cache", description="Cache des embeddings")

    # === Logging ===
    LOG_LEVEL: str = Field(default="INFO", description="Niveau de log (DEBUG, INFO, WARNING, ERROR)")
    LOG_FILE: str = Field(default="./logs/finrag.log", description="Fichier de log")

    @property
    def use_openai_embeddings(self) -> bool:
        """Retourne True si la clé OpenAI est disponible."""
        return bool(self.OPENAI_API_KEY)

    @property
    def use_anthropic(self) -> bool:
        """Retourne True si la clé Anthropic est disponible."""
        return bool(self.ANTHROPIC_API_KEY)

    @property
    def chroma_persist_path(self) -> Path:
        """Retourne le chemin absolu du répertoire ChromaDB."""
        return Path(self.CHROMA_PERSIST_DIR).resolve()


settings = Settings()