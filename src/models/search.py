from pydantic import BaseModel, Field
from typing import List, Dict, Any


class SearchQueryRequest(BaseModel):
    """Request model for semantic search query."""

    query_embedding: List[float] = Field(
        ..., description="The query vector embedding for semantic search"
    )
    collection_name: str = Field(
        default="car_embeddings", description="Name of the Qdrant collection"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0 to 1.0)",
    )
    start_date: str | None = Field(
        default=None,
        description="Optional start date filter (ISO format or YYYY-MM-DD)",
    )
    end_date: str | None = Field(
        default=None,
        description="Optional end date filter (ISO format or YYYY-MM-DD)",
    )
    limit: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results to return"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query_embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "collection_name": "car_embeddings",
                "similarity_threshold": 0.7,
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "limit": 10,
            }
        }


class SearchResult(BaseModel):
    """Individual search result model."""

    id: str | int = Field(..., description="Result ID")
    score: float = Field(..., description="Similarity score")
    payload: Dict[str, Any] = Field(..., description="Result payload data")


class SearchQueryResponse(BaseModel):
    """Response model for semantic search query."""

    results: List[SearchResult] = Field(..., description="List of search results")
    count: int = Field(..., description="Number of results returned")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "id": "123",
                        "score": 0.95,
                        "payload": {
                            "car_type": "pickup truck",
                            "car_brand": "Toyota",
                            "created_at": "2024-01-15T10:30:00Z",
                        },
                    }
                ],
                "count": 1,
            }
        }

