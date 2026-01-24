# NVIDIA NIM RAG Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QbitLoop/nvidia-nim-rag-demo/blob/main/notebooks/nvidia_nim_rag_colab.ipynb)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-NIM-76B900?style=for-the-badge&logo=nvidia)](https://build.nvidia.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-990F3D?style=for-the-badge)](LICENSE)

**Production-ready RAG application** using NVIDIA NIM inference microservices for enterprise knowledge retrieval.

---

## Try It Now

**[Open in Google Colab](https://colab.research.google.com/github/QbitLoop/nvidia-nim-rag-demo/blob/main/notebooks/nvidia_nim_rag_colab.ipynb)** - No setup required, just add your NVIDIA API key!

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA NIM RAG Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         USER INTERFACE                               │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  Streamlit Frontend  │  Query Input  │  Latency Display     │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         FASTAPI BACKEND                              │   │
│  │  ┌────────────┐    ┌────────────────┐    ┌──────────────────┐      │   │
│  │  │   Router   │───▶│  RAG Pipeline  │───▶│ Response Builder │      │   │
│  │  │   /query   │    │  + Citations   │    │  + Latency       │      │   │
│  │  └────────────┘    └────────────────┘    └──────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                      │                      │                     │
│         ▼                      ▼                      ▼                     │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐              │
│  │  NVIDIA NIM    │   │   PgVector     │   │ NV-Embed-QA    │              │
│  │  LLM Inference │   │  Vector Store  │   │  Embeddings    │              │
│  │  ~150ms TTFT   │   │  Similarity    │   │  ~50ms/doc     │              │
│  └────────────────┘   └────────────────┘   └────────────────┘              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  METRICS: TTFT ~150ms  │  E2E ~800ms  │  Throughput ~20 req/sec            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **TTFT** | ~150ms | Time to first token |
| **E2E Latency** | ~800ms | End-to-end query response |
| **Embedding** | ~50ms | Document vectorization |
| **Throughput** | ~20 req/sec | Concurrent handling |

---

## Features

- **NIM Integration** - NVIDIA NIM API for fast LLM inference
- **Document Ingestion** - Upload PDFs and text files
- **Semantic Search** - Vector embeddings with NV-Embed-QA
- **Query Interface** - Clean UI with latency metrics
- **Response Citations** - Source documents displayed

---

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/QbitLoop/nvidia-nim-rag-demo.git
cd nvidia-nim-rag-demo

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your NVIDIA API key from https://build.nvidia.com
```

### 3. Start Database

```bash
docker-compose up -d postgres
```

### 4. Run Application

```bash
# Backend
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
streamlit run app/streamlit_app.py
```

### 5. Access

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| API Docs | http://localhost:8000/docs |

---

## Project Structure

```
nvidia-nim-rag-demo/
├── app/
│   ├── main.py              # FastAPI application
│   ├── nim_client.py        # NVIDIA NIM API wrapper
│   ├── vector_store.py      # PgVector operations
│   ├── rag_pipeline.py      # RAG orchestration
│   ├── models.py            # Pydantic models
│   └── streamlit_app.py     # Streamlit frontend
├── data/
│   └── sample_docs/         # Sample documents
├── tests/
│   └── test_rag.py
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## NVIDIA Technologies

| Technology | Purpose |
|------------|---------|
| **NVIDIA NIM** | LLM inference microservices |
| **NV-Embed-QA** | Document embeddings |
| **build.nvidia.com** | API endpoints |

---

## Use Cases

| Use Case | Description |
|----------|-------------|
| Technical Documentation Q&A | Query product docs |
| Policy Search | Find relevant policies |
| Knowledge Base | Enterprise retrieval |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | NVIDIA NIM API |
| Backend | FastAPI |
| Frontend | Streamlit |
| Vector DB | PostgreSQL + pgvector |
| Language | Python 3.11+ |

---

## Related Projects

| Project | Description |
|---------|-------------|
| [RealtimeVoice](https://github.com/QbitLoop/RealtimeVoice) | ASR Benchmark: Nemotron 21x faster |
| [NemotronVoiceRAG](https://github.com/QbitLoop/NemotronVoiceRAG) | Voice-enabled RAG |
| [ai-infra-advisor](https://github.com/QbitLoop/ai-infra-advisor) | TCO Calculator |

---

## Author

**Waseem Habib** | [GitHub](https://github.com/QbitLoop) | [LinkedIn](https://www.linkedin.com/in/wh2002/)

---

## License

MIT

---

*Built by [QbitLoop](https://github.com/QbitLoop) | Brand-WHFT Design System*
