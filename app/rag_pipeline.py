"""RAG pipeline orchestration."""
import time
import tiktoken
from loguru import logger
from .nim_client import NIMClient, RAG_SYSTEM_PROMPT, format_context_for_rag, build_rag_prompt
from .vector_store import VectorStore
from .models import QueryResponse, Citation, IngestionResponse


class RAGPipeline:
    """Orchestrates the RAG workflow."""

    def __init__(
        self,
        nim_client: NIMClient,
        vector_store: VectorStore,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.nim = nim_client
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info("RAG Pipeline initialized")

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks based on token count."""
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Move start position with overlap
            start = end - self.chunk_overlap

        logger.info(f"Text chunked into {len(chunks)} chunks")
        return chunks

    async def ingest_document(
        self,
        content: str,
        source: str,
        metadata: dict = None
    ) -> IngestionResponse:
        """Ingest a document into the vector store."""
        start_time = time.perf_counter()
        metadata = metadata or {}

        # Insert document
        doc_id = await self.vector_store.insert_document(
            content=content,
            source=source,
            metadata=metadata
        )

        # Chunk the document
        chunks = self._chunk_text(content)

        # Generate embeddings for all chunks
        embeddings = await self.nim.embed(chunks)

        # Insert chunks with embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            await self.vector_store.insert_chunk(
                document_id=doc_id,
                content=chunk,
                embedding=embedding,
                chunk_index=i,
                metadata={"source": source, **metadata}
            )

        processing_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"Document ingested: {doc_id} with {len(chunks)} chunks in {processing_time:.2f}ms")

        return IngestionResponse(
            document_id=doc_id,
            chunks_created=len(chunks),
            status="success",
            processing_time_ms=processing_time
        )

    async def query(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        include_sources: bool = True
    ) -> QueryResponse:
        """Execute a RAG query."""
        start_time = time.perf_counter()
        metrics = {}

        # Step 1: Embed the query
        embed_start = time.perf_counter()
        query_embedding = await self.nim.embed_single(query)
        metrics["embed_time_ms"] = (time.perf_counter() - embed_start) * 1000

        # Step 2: Retrieve similar chunks
        retrieve_start = time.perf_counter()
        similar_chunks = await self.vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        metrics["retrieve_time_ms"] = (time.perf_counter() - retrieve_start) * 1000
        metrics["chunks_retrieved"] = len(similar_chunks)

        # Step 3: Build context and prompt
        if not similar_chunks:
            return QueryResponse(
                query=query,
                answer="I don't have enough information in the knowledge base to answer this question.",
                citations=[],
                metrics=metrics
            )

        context = format_context_for_rag(similar_chunks)
        prompt = build_rag_prompt(query, context)

        # Step 4: Generate answer
        generate_start = time.perf_counter()
        answer = await self.nim.generate(
            prompt=prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
            max_tokens=1024,
            temperature=0.3
        )
        metrics["generate_time_ms"] = (time.perf_counter() - generate_start) * 1000

        # Step 5: Build citations
        citations = []
        if include_sources:
            for chunk in similar_chunks:
                citations.append(Citation(
                    document_id=chunk["document_id"],
                    source=chunk["source"],
                    content=chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    similarity_score=chunk["similarity"],
                    chunk_index=chunk["chunk_index"]
                ))

        # Calculate total time
        metrics["total_time_ms"] = (time.perf_counter() - start_time) * 1000

        logger.info(f"RAG query completed in {metrics['total_time_ms']:.2f}ms")

        return QueryResponse(
            query=query,
            answer=answer,
            citations=citations,
            metrics=metrics
        )

    async def batch_ingest(
        self,
        documents: list[dict]
    ) -> list[IngestionResponse]:
        """Ingest multiple documents."""
        results = []
        for doc in documents:
            result = await self.ingest_document(
                content=doc["content"],
                source=doc["source"],
                metadata=doc.get("metadata", {})
            )
            results.append(result)
        return results
