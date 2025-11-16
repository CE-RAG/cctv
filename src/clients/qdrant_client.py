from qdrant_client import QdrantClient as _QdrantClient

from src.config import qdrant_config


class QdrantClient(_QdrantClient):
    """Custom Qdrant client with centralized configuration."""
    
    def __init__(self, **kwargs):
        """Initialize Qdrant client with config defaults."""
        # Use config values as defaults, but allow override via kwargs
        host = kwargs.pop("host", qdrant_config.qdrant_host)
        port = kwargs.pop("port", qdrant_config.qdrant_port)
        grpc_port = kwargs.pop("grpc_port", qdrant_config.qdrant_grpc_port)
        api_key = kwargs.pop("api_key", qdrant_config.qdrant_api_key)
        
        super().__init__(
            host=host,
            port=port,
            grpc_port=grpc_port,
            api_key=api_key,
            **kwargs
        )


# Global client instance (singleton pattern)
_qdrant_client_instance: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client instance."""
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        _qdrant_client_instance = QdrantClient()
    return _qdrant_client_instance
