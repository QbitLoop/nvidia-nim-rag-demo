# NVIDIA NIM RAG Demo

Demonstrating retrieval-augmented generation using NVIDIA NIM inference microservices.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                            │
│                   (Query Interface)                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                               │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐    │
│  │   Router    │ → │ RAG Pipeline │ → │ Response Builder  │    │
│  └─────────────┘   └──────────────┘   └───────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ NVIDIA NIM API   │ │   PgVector DB    │ │ NVIDIA Embeddings│
│ (LLM Inference)  │ │ (Vector Store)   │ │ (NV-Embed-QA)    │
│ build.nvidia.com │ │                  │ │ build.nvidia.com │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

## Features

- **NIM Integration**: Uses NVIDIA NIM API for fast inference
- **Document Ingestion**: Upload PDFs and text files
- **Semantic Search**: Vector embeddings with NVIDIA NV-Embed-QA
- **Query Interface**: Clean Streamlit UI with latency metrics
- **Response Citations**: Shows source documents used

## Quick Start

### 1. Setup Environment

```bash
# Clone the repo
git clone https://github.com/Qbitloop/nvidia-nim-rag-demo.git
cd nvidia-nim-rag-demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example env
cp .env.example .env

# Edit .env with your NVIDIA API key
# Get your key at: https://build.nvidia.com
```

### 3. Start PostgreSQL with pgvector

```bash
# Using Docker
docker-compose up -d postgres

# Or use local PostgreSQL with pgvector extension
```

### 4. Run the Application

```bash
# Start FastAPI backend
uvicorn app.main:app --reload --port 8000

# In another terminal, start Streamlit frontend
streamlit run app/streamlit_app.py
```

### 5. Access the Demo

- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| TTFT | ~150ms | Time to first token |
| E2E Latency | ~800ms | End-to-end query response |
| Embedding Speed | ~50ms | Document vectorization |
| Throughput | ~20 req/sec | Concurrent query handling |

## Project Structure

```
nvidia-nim-rag-demo/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── nim_client.py        # NVIDIA NIM API wrapper
│   ├── vector_store.py      # PgVector operations
│   ├── rag_pipeline.py      # RAG orchestration
│   ├── models.py            # Pydantic models
│   └── streamlit_app.py     # Streamlit frontend
├── data/
│   └── sample_docs/         # Sample documents for demo
├── tests/
│   └── test_rag.py
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## NVIDIA Technologies Used

| Technology | Purpose |
|------------|---------|
| **NVIDIA NIM** | LLM inference microservices |
| **NV-Embed-QA** | Document embeddings |
| **build.nvidia.com** | API endpoints |

## Use Cases Demonstrated

1. **Technical Documentation Q&A** - Query product docs
2. **Policy Search** - Find relevant policies from corpus
3. **Knowledge Base** - Enterprise knowledge retrieval

## Built With

- **NVIDIA NIM API** - `build.nvidia.com`
- **FastAPI** - Backend framework
- **Streamlit** - Frontend UI
- **PostgreSQL + pgvector** - Vector database
- **Python 3.11+**

## Author

**Waseem Habib**
- GitHub: [@Qbitloop](https://github.com/Qbitloop)
- LinkedIn: [Waseem Habib](https://www.linkedin.com/in/wh2002/)

## License

MIT License - See LICENSE for details

---

*Built to demonstrate NVIDIA NIM capabilities for RAG applications*
