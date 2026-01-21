"""Pydantic models for the RAG API."""
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model for ingestion."""
    id: Optional[str] = None
    content: str
    metadata: dict = Field(default_factory=dict)
    source: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """Chunked document for vector storage."""
    id: Optional[str] = None
    document_id: str
    content: str
    embedding: Optional[List[float]] = None
    chunk_index: int
    metadata: Dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    include_sources: bool = True


class Citation(BaseModel):
    """Citation from source documents."""
    document_id: str
    source: str
    content: str
    similarity_score: float
    chunk_index: int


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    metrics: Dict = Field(default_factory=dict)


class IngestionRequest(BaseModel):
    """Request model for document ingestion."""
    content: str
    source: str
    metadata: dict = Field(default_factory=dict)


class IngestionResponse(BaseModel):
    """Response model for document ingestion."""
    document_id: str
    chunks_created: int
    status: str = "success"
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    nim_connected: bool
    database_connected: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
