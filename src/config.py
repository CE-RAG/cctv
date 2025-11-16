from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantConfig(BaseSettings):
    """Centralized application settings loaded from environment variables."""
    
    # Qdrant configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: str | None = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
qdrant_config = QdrantConfig()

__all__ = ["qdrant_config"]