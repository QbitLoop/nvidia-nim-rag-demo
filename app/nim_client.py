"""NVIDIA NIM API client for LLM inference and embeddings."""
import os
import time
from typing import AsyncGenerator
from openai import AsyncOpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class NIMClient:
    """Client for NVIDIA NIM inference microservices."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        llm_model: str = "meta/llama-3.1-70b-instruct",
        embed_model: str = "nvidia/nv-embedqa-e5-v5"
    ):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is required")

        self.base_url = base_url
        self.llm_model = llm_model
        self.embed_model = embed_model

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        logger.info(f"NIM client initialized with model: {llm_model}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        """Generate text using NVIDIA NIM LLM."""
        start_time = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            if stream:
                return self._stream_response(messages, max_tokens, temperature, start_time)
            else:
                response = await self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                elapsed = (time.perf_counter() - start_time) * 1000
                logger.info(f"NIM generation completed in {elapsed:.2f}ms")

                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"NIM generation error: {e}")
            raise

    async def _stream_response(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens."""
        first_token = True

        stream = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                if first_token:
                    ttft = (time.perf_counter() - start_time) * 1000
                    logger.info(f"TTFT: {ttft:.2f}ms")
                    first_token = False
                yield chunk.choices[0].delta.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using NVIDIA NV-Embed-QA."""
        start_time = time.perf_counter()

        try:
            response = await self.client.embeddings.create(
                model=self.embed_model,
                input=texts,
                encoding_format="float"
            )

            elapsed = (time.perf_counter() - start_time) * 1000
            logger.info(f"Generated {len(texts)} embeddings in {elapsed:.2f}ms")

            return [item.embedding for item in response.data]

        except Exception as e:
            logger.error(f"NIM embedding error: {e}")
            raise

    async def embed_single(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        embeddings = await self.embed([text])
        return embeddings[0]

    async def health_check(self) -> bool:
        """Check if NIM API is accessible."""
        try:
            await self.embed(["health check"])
            return True
        except Exception as e:
            logger.warning(f"NIM health check failed: {e}")
            return False


# RAG-specific prompts
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

Instructions:
1. Answer ONLY based on the provided context
2. If the context doesn't contain relevant information, say "I don't have enough information to answer this question."
3. Cite specific parts of the context in your answer
4. Be concise but thorough
5. If multiple documents are relevant, synthesize the information

Context will be provided in the following format:
[Document: source_name]
content...
[End Document]
"""

def format_context_for_rag(chunks: list[dict]) -> str:
    """Format retrieved chunks into context for LLM."""
    context_parts = []
    for chunk in chunks:
        source = chunk.get("source", "Unknown")
        content = chunk.get("content", "")
        context_parts.append(f"[Document: {source}]\n{content}\n[End Document]")

    return "\n\n".join(context_parts)


def build_rag_prompt(query: str, context: str) -> str:
    """Build the final RAG prompt."""
    return f"""Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. Include citations to specific documents when possible."""
