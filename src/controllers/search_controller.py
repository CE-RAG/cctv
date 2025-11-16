from typing import List

from fastapi import APIRouter, HTTPException, status

from src.models.search import SearchQueryRequest, SearchQueryResponse, SearchResult
from src.services.search_service import SearchService

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("/query", response_model=SearchQueryResponse)
async def search_query(request: SearchQueryRequest) -> SearchQueryResponse:
    """Search for vehicles by semantic similarity.

    This endpoint performs semantic vector search in Qdrant to find
    vehicles matching the provided query embedding.
    """
    try:
        service = SearchService()
        results = service.search_vehicles(
            query_embedding=request.query_embedding,
            collection_name=request.collection_name,
            similarity_threshold=request.similarity_threshold,
            start_date=request.start_date,
            end_date=request.end_date,
            limit=request.limit,
        )

        # Convert results to SearchResult models
        search_results = [
            SearchResult(
                id=result["id"], score=result["score"], payload=result["payload"]
            )
            for result in results
        ]

        return SearchQueryResponse(results=search_results, count=len(search_results))

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for the search service."""
    return {"status": "healthy", "service": "search"}
