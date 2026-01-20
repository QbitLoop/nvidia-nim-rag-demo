"""Tests for the RAG pipeline."""
import pytest
from unittest.mock import AsyncMock, patch
import sys
sys.path.insert(0, '..')

from app.models import QueryRequest, IngestionRequest


class TestModels:
    """Test Pydantic models."""

    def test_query_request_validation(self):
        """Test QueryRequest validation."""
        # Valid request
        request = QueryRequest(query="What is NIM?")
        assert request.query == "What is NIM?"
        assert request.top_k == 5
        assert request.similarity_threshold == 0.7

    def test_query_request_custom_params(self):
        """Test QueryRequest with custom parameters."""
        request = QueryRequest(
            query="Test query",
            top_k=10,
            similarity_threshold=0.8,
            include_sources=False
        )
        assert request.top_k == 10
        assert request.similarity_threshold == 0.8
        assert request.include_sources is False

    def test_ingestion_request(self):
        """Test IngestionRequest validation."""
        request = IngestionRequest(
            content="Test content",
            source="test.txt",
            metadata={"key": "value"}
        )
        assert request.content == "Test content"
        assert request.source == "test.txt"
        assert request.metadata == {"key": "value"}


class TestNIMClient:
    """Test NIM client (mocked)."""

    @pytest.mark.asyncio
    async def test_embed_returns_embeddings(self):
        """Test embedding generation."""
        with patch('app.nim_client.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            # Mock embedding response
            mock_response = AsyncMock()
            mock_response.data = [
                AsyncMock(embedding=[0.1] * 1024)
            ]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)

            from app.nim_client import NIMClient
            client = NIMClient(api_key="test-key")
            embeddings = await client.embed(["test text"])

            assert len(embeddings) == 1
            assert len(embeddings[0]) == 1024

    @pytest.mark.asyncio
    async def test_generate_returns_text(self):
        """Test text generation."""
        with patch('app.nim_client.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client

            # Mock completion response
            mock_response = AsyncMock()
            mock_response.choices = [
                AsyncMock(message=AsyncMock(content="Test response"))
            ]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            from app.nim_client import NIMClient
            client = NIMClient(api_key="test-key")
            response = await client.generate("Test prompt")

            assert response == "Test response"


class TestRAGPipeline:
    """Test RAG pipeline (integration-style with mocks)."""

    @pytest.mark.asyncio
    async def test_chunk_text(self):
        """Test text chunking."""
        with patch('app.nim_client.AsyncOpenAI'):
            from app.nim_client import NIMClient
            from app.vector_store import VectorStore
            from app.rag_pipeline import RAGPipeline

            nim = NIMClient(api_key="test-key")
            vs = VectorStore(database_url="postgresql://test:test@localhost/test")
            pipeline = RAGPipeline(nim, vs, chunk_size=100, chunk_overlap=10)

            # Test chunking
            text = "This is a test. " * 100
            chunks = pipeline._chunk_text(text)

            assert len(chunks) > 1
            assert all(len(c) > 0 for c in chunks)


class TestContextFormatting:
    """Test context formatting utilities."""

    def test_format_context_for_rag(self):
        """Test context formatting."""
        from app.nim_client import format_context_for_rag

        chunks = [
            {"source": "doc1.txt", "content": "Content 1"},
            {"source": "doc2.txt", "content": "Content 2"}
        ]

        context = format_context_for_rag(chunks)

        assert "[Document: doc1.txt]" in context
        assert "[Document: doc2.txt]" in context
        assert "Content 1" in context
        assert "Content 2" in context

    def test_build_rag_prompt(self):
        """Test RAG prompt building."""
        from app.nim_client import build_rag_prompt

        prompt = build_rag_prompt("What is NIM?", "NIM is inference microservices.")

        assert "What is NIM?" in prompt
        assert "NIM is inference microservices." in prompt
        assert "Context:" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
