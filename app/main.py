"""FastAPI application for NVIDIA NIM RAG Demo."""
import os
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv

from . import __version__
from .models import (
    QueryRequest, QueryResponse,
    IngestionRequest, IngestionResponse,
    HealthResponse
)
from .nim_client import NIMClient
from .vector_store import VectorStore
from .rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Global instances
nim_client: NIMClient | None = None
vector_store: VectorStore | None = None
rag_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global nim_client, vector_store, rag_pipeline

    # Startup
    logger.info("Starting NVIDIA NIM RAG Demo...")

    try:
        # Initialize NIM client
        nim_client = NIMClient(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            llm_model=os.getenv("NIM_LLM_MODEL", "meta/llama-3.1-70b-instruct"),
            embed_model=os.getenv("NIM_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
        )

        # Initialize vector store
        vector_store = VectorStore(
            database_url=os.getenv("DATABASE_URL"),
            dimension=int(os.getenv("PGVECTOR_DIMENSION", "1024"))
        )
        await vector_store.connect()

        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            nim_client=nim_client,
            vector_store=vector_store,
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
        )

        logger.info("Application started successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")
    if vector_store:
        await vector_store.close()


# Create FastAPI app
app = FastAPI(
    title="NVIDIA NIM RAG Demo",
    description="RAG application powered by NVIDIA NIM inference microservices",
    version=__version__,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "name": "NVIDIA NIM RAG Demo",
        "version": __version__,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    nim_ok = await nim_client.health_check() if nim_client else False
    db_ok = await vector_store.health_check() if vector_store else False

    return HealthResponse(
        status="healthy" if (nim_ok and db_ok) else "degraded",
        version=__version__,
        nim_connected=nim_ok,
        database_connected=db_ok,
        timestamp=datetime.utcnow()
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Execute a RAG query."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        response = await rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            include_sources=request.include_sources
        )
        return response

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(request: IngestionRequest):
    """Ingest a document into the knowledge base."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    try:
        response = await rag_pipeline.ingest_document(
            content=request.content,
            source=request.source,
            metadata=request.metadata
        )
        return response

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", response_model=IngestionResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a file into the knowledge base."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    # Read file content
    content = await file.read()

    # Handle different file types
    if file.filename.endswith(".txt"):
        text_content = content.decode("utf-8")
    elif file.filename.endswith(".pdf"):
        # Would use pypdf here in production
        raise HTTPException(status_code=400, detail="PDF support coming soon")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        response = await rag_pipeline.ingest_document(
            content=text_content,
            source=file.filename,
            metadata={"filename": file.filename, "content_type": file.content_type}
        )
        return response

    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=dict)
async def get_stats():
    """Get vector store statistics."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    stats = await vector_store.get_stats()
    return stats


@app.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the knowledge base."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    success = await vector_store.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"status": "deleted", "document_id": document_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
