# NVIDIA NIM RAG Demo - Building Guide

## Development Setup

### Prerequisites

```bash
# Python 3.11+
python --version

# Docker (for PostgreSQL)
docker --version

# Git
git --version
```

### Local Setup

```bash
# Clone repository
git clone https://github.com/QbitLoop/nvidia-nim-rag-demo.git
cd nvidia-nim-rag-demo

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your NVIDIA API key from https://build.nvidia.com
```

### Start Dependencies

```bash
# Start PostgreSQL with pgvector
docker-compose up -d postgres

# Verify database is running
docker-compose logs postgres
```

---

## Project Structure

```
nvidia-nim-rag-demo/
├── app/
│   ├── main.py              # FastAPI application entry
│   ├── nim_client.py        # NVIDIA NIM API wrapper
│   ├── vector_store.py      # PgVector operations
│   ├── rag_pipeline.py      # RAG orchestration
│   ├── models.py            # Pydantic models
│   └── streamlit_app.py     # Streamlit frontend
├── data/
│   └── sample_docs/         # Sample documents for demo
├── tests/
│   ├── test_rag.py
│   └── conftest.py
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Key Components

### 1. NIM Client (`nim_client.py`)

Wrapper for NVIDIA NIM API:

```python
class NIMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.build.nvidia.com"

    async def generate(self, prompt: str, context: str) -> str:
        # Call NIM inference endpoint
        pass

    async def embed(self, text: str) -> List[float]:
        # Generate embeddings with NV-Embed-QA
        pass
```

### 2. Vector Store (`vector_store.py`)

PostgreSQL with pgvector:

```python
class VectorStore:
    async def add_document(self, doc: Document) -> str:
        # Insert document with embedding
        pass

    async def search(self, query_embedding: List[float], top_k: int) -> List[Document]:
        # Similarity search
        pass
```

### 3. RAG Pipeline (`rag_pipeline.py`)

Orchestrates retrieval and generation:

```python
class RAGPipeline:
    async def query(self, question: str) -> Response:
        # 1. Embed question
        # 2. Retrieve relevant docs
        # 3. Generate response with context
        # 4. Return with citations
        pass
```

---

## Adding New Features

### Add New Document Type

1. Update `app/models.py`:

```python
class DocumentType(str, Enum):
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"  # Add new type
```

2. Add parser in `app/document_parser.py`:

```python
def parse_markdown(file_path: str) -> List[Chunk]:
    # Implementation
    pass
```

3. Register in pipeline.

### Add New Embedding Model

1. Create wrapper in `app/embeddings/`:

```python
# app/embeddings/openai.py
class OpenAIEmbeddings:
    async def embed(self, text: str) -> List[float]:
        pass
```

2. Register in configuration.

---

## Key Commands

```bash
# Start backend
uvicorn app.main:app --reload --port 8000

# Start frontend
streamlit run app/streamlit_app.py

# Run tests
pytest tests/ -v

# Format code
ruff format app/

# Lint code
ruff check app/
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `streamlit` | Frontend UI |
| `asyncpg` | PostgreSQL async driver |
| `pgvector` | Vector similarity extension |
| `httpx` | Async HTTP client |
| `pydantic` | Data validation |
| `python-dotenv` | Environment variables |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NVIDIA_API_KEY` | Yes | From build.nvidia.com |
| `DATABASE_URL` | Yes | PostgreSQL connection |
| `LOG_LEVEL` | No | Logging level (default: INFO) |

---

*Created: 2026-01-22 | Author: Waseem Habib | QbitLoop*
