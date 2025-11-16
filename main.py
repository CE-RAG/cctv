from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.controllers import search_router

app = FastAPI(
    title="CCTV Vehicle Search API",
    description="API for semantic search of vehicle detection data",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CCTV Vehicle Search API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
