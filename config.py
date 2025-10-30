from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # V2-style model config
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",          # <â€" ignore env vars not declared below
        case_sensitive=False,    # default anyway; keeps .env keys uppercase-friendly
    )
    # ChromaDB Configuration
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    RAG_API_PORT: int = 8001
    CHROMA_COLLECTION_NAME: str = "org_documents"
    
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-5-chat-latest"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 1536
    LLM_TEMPERATURE: float = 0.3
    LLM_TOP_P: float = 0.9
    LLM_MAX_TOKENS: int = 2048
    
    # Application
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # RAG Configuration - Semantic Chunking Strategy
    # Using semantic chunking based on document structure (headings, paragraphs)
    CHUNKING_STRATEGY: str = "by_title"  # "by_title" for semantic, "basic" for simple combination
    MAX_CHUNK_SIZE: int = 1000  # Maximum characters per chunk
    CHUNK_OVERLAP: int = 200  # Character overlap between chunks for context preservation
    COMBINE_TEXT_UNDER_N_CHARS: int = 500  # Combine small sections to avoid micro-chunks
    MULTIPAGE_SECTIONS: bool = True  # Allow sections to span multiple pages
    INCLUDE_ORIG_ELEMENTS: bool = False  # Include original elements in chunk metadata
    
    # Legacy settings for backwards compatibility (not used with semantic chunking)
    CHUNK_SIZE: int = 512  # Legacy: token-based chunk size
    MAX_CONTEXT_LENGTH: int = 8192
    
    # Retrieval Configuration
    SIMILARITY_THRESHOLD: float = 0.70
    VECTOR_TOP_K: int = 5
    
    # Reranking Configuration
    USE_RERANKER: bool = True
    VECTOR_RERANK_MULTIPLIER: int = 3  # Retrieve 3x candidates for reranking (e.g., 15 candidates â†' rerank â†' top 5)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast, accurate cross-encoder

settings = Settings()
