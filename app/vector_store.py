"""Vector store operations using PostgreSQL with pgvector."""
import os
import uuid
from datetime import datetime
from typing import Optional
import asyncpg
from loguru import logger


class VectorStore:
    """PostgreSQL + pgvector vector store for document chunks."""

    def __init__(
        self,
        database_url: str | None = None,
        dimension: int = 1024
    ):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")

        self.dimension = dimension
        self.pool: Optional[asyncpg.Pool] = None
        logger.info(f"VectorStore initialized with dimension: {dimension}")

    async def connect(self):
        """Create database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10
            )
            await self._init_schema()
            logger.info("VectorStore connected to database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("VectorStore connection closed")

    async def _init_schema(self):
        """Initialize database schema with pgvector extension."""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content TEXT NOT NULL,
                    source VARCHAR(500),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create chunks table with vector embedding
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    embedding vector({self.dimension}),
                    chunk_index INTEGER,
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Create vector similarity index (IVFFlat for approximate search)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)

            logger.info("Database schema initialized")

    async def insert_document(
        self,
        content: str,
        source: str,
        metadata: dict = None
    ) -> str:
        """Insert a document and return its ID."""
        doc_id = str(uuid.uuid4())
        metadata = metadata or {}

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO documents (id, content, source, metadata)
                VALUES ($1, $2, $3, $4)
                """,
                uuid.UUID(doc_id), content, source, metadata
            )

        logger.info(f"Document inserted: {doc_id}")
        return doc_id

    async def insert_chunk(
        self,
        document_id: str,
        content: str,
        embedding: list[float],
        chunk_index: int,
        metadata: dict = None
    ) -> str:
        """Insert a document chunk with embedding."""
        chunk_id = str(uuid.uuid4())
        metadata = metadata or {}

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chunks (id, document_id, content, embedding, chunk_index, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                uuid.UUID(chunk_id),
                uuid.UUID(document_id),
                content,
                embedding,
                chunk_index,
                metadata
            )

        return chunk_id

    async def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> list[dict]:
        """Search for similar chunks using cosine similarity."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id,
                    c.document_id,
                    c.content,
                    c.chunk_index,
                    c.metadata,
                    d.source,
                    1 - (c.embedding <=> $1::vector) as similarity
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE 1 - (c.embedding <=> $1::vector) >= $3
                ORDER BY c.embedding <=> $1::vector
                LIMIT $2
                """,
                query_embedding,
                top_k,
                similarity_threshold
            )

            results = []
            for row in rows:
                results.append({
                    "id": str(row["id"]),
                    "document_id": str(row["document_id"]),
                    "content": row["content"],
                    "chunk_index": row["chunk_index"],
                    "metadata": row["metadata"],
                    "source": row["source"],
                    "similarity": float(row["similarity"])
                })

            logger.info(f"Found {len(results)} similar chunks")
            return results

    async def get_document(self, document_id: str) -> Optional[dict]:
        """Get a document by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM documents WHERE id = $1",
                uuid.UUID(document_id)
            )

            if row:
                return {
                    "id": str(row["id"]),
                    "content": row["content"],
                    "source": row["source"],
                    "metadata": row["metadata"],
                    "created_at": row["created_at"]
                }
            return None

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM documents WHERE id = $1",
                uuid.UUID(document_id)
            )
            return result == "DELETE 1"

    async def get_stats(self) -> dict:
        """Get vector store statistics."""
        async with self.pool.acquire() as conn:
            doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
            chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")

            return {
                "documents": doc_count,
                "chunks": chunk_count,
                "dimension": self.dimension
            }

    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False
