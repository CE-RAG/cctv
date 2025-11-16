from typing import List, Dict, Any
from datetime import datetime

from src.clients import QdrantClient, get_qdrant_client


class VectorRepository:
    """Repository for semantic vector search in Qdrant database."""

    def __init__(self, client: QdrantClient | None = None):
        """Initialize vector repository with Qdrant client."""
        self.client = client or get_qdrant_client()

    def search_semantic(
        self,
        query_embedding: List[float],
        collection_name: str = "car_embeddings",
        similarity_threshold: float = 0.7,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search by semantic similarity.

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
        # Get more results than needed to account for date filtering
        search_limit = limit * 3 if (start_date or end_date) else limit

        print("vector query: ", query_embedding)
        results = self.client.search(
            collection_name=collection_name,
            query_vector=("semantic", query_embedding),
            score_threshold=similarity_threshold,
            limit=search_limit,
        )

        # Post-filter by ISO datetime string
        filtered_results = []
        if start_date or end_date:
            start_dt = None
            end_dt = None

            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                except ValueError:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")

            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                except ValueError:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    end_dt = end_dt.replace(hour=23, minute=59, second=59)

            for hit in results:
                payload = hit.payload
                created_at_str = payload.get("created_at", "")

                if created_at_str:
                    try:
                        created_dt = datetime.fromisoformat(
                            created_at_str.replace("Z", "+00:00")
                        )

                        # Check if within date range
                        if start_dt and created_dt < start_dt:
                            continue
                        if end_dt and created_dt > end_dt:
                            continue

                        filtered_results.append(
                            {
                                "id": hit.id,
                                "score": hit.score,
                                "payload": payload,
                            }
                        )

                        # Stop if we have enough results
                        if len(filtered_results) >= limit:
                            break
                    except (ValueError, AttributeError):
                        # If datetime parsing fails, include the result
                        filtered_results.append(
                            {
                                "id": hit.id,
                                "score": hit.score,
                                "payload": payload,
                            }
                        )
                        if len(filtered_results) >= limit:
                            break
                else:
                    # If no created_at, include the result
                    filtered_results.append(
                        {
                            "id": hit.id,
                            "score": hit.score,
                            "payload": payload,
                        }
                    )
                    if len(filtered_results) >= limit:
                        break
        else:
            # No date filtering needed
            filtered_results = [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
                for hit in results[:limit]
            ]

        return filtered_results

