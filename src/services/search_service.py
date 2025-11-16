from typing import List, Dict, Any

from src.repositories import VectorRepository


class SearchService:
    """Service for semantic vector search operations."""

    def __init__(self, repository: VectorRepository | None = None):
        """Initialize search service with vector repository."""
        self.repository = repository or VectorRepository()

    def search_vehicles(
        self,
        query_embedding: List[float],
        collection_name: str = "car_embeddings",
        similarity_threshold: float = 0.7,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for vehicles by semantic similarity.

        Args:
            query_embedding: The query vector embedding
            collection_name: Name of the Qdrant collection
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            start_date: Optional start date filter (ISO format or YYYY-MM-DD)
            end_date: Optional end date filter (ISO format or YYYY-MM-DD)
            limit: Maximum number of results to return

        Returns:
            List of search results with id, score, and payload
        """
        return self.repository.search_semantic(
            query_embedding=query_embedding,
            collection_name=collection_name,
            similarity_threshold=similarity_threshold,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

