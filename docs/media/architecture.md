# NVIDIA NIM RAG Demo Architecture

## System Overview

```mermaid
flowchart TB
    subgraph Frontend["Frontend (Streamlit)"]
        UI[Query Interface]
        DISP[Results Display]
    end

    subgraph Backend["Backend (FastAPI)"]
        API[REST API<br/>/query, /ingest]
        RAG[RAG Pipeline]
        EMB[Embedding Service]
    end

    subgraph Storage["Storage Layer"]
        PG[(PostgreSQL<br/>+ pgvector)]
        DOCS[Document Store]
    end

    subgraph NVIDIA["NVIDIA NIM"]
        LLM[LLM Inference<br/>OpenAI-compatible]
        NVEMB[NV-Embed-QA]
    end

    UI --> API
    API --> RAG
    RAG --> EMB
    EMB --> NVEMB
    RAG --> PG
    RAG --> LLM
    LLM --> DISP
    DOCS --> EMB
```

## Document Ingestion Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as API
    participant P as Processor
    participant E as Embeddings
    participant D as pgvector

    U->>A: Upload Document
    A->>P: Parse (PDF/DOCX)
    P->>P: Chunk Text
    P->>E: Generate Embeddings
    Note over E: NV-Embed-QA<br/>~50ms/doc
    E->>D: Store Vectors
    D->>A: Confirm Indexed
    A->>U: Success
```

## Query Flow

```mermaid
sequenceDiagram
    participant U as User
    participant A as API
    participant V as pgvector
    participant L as NIM LLM

    U->>A: Natural Language Query
    A->>V: Vector Similarity Search
    V->>A: Top-K Documents
    A->>L: Query + Context
    Note over L: TTFT: ~150ms
    L->>A: Generated Response
    A->>U: Answer + Citations
    Note over U,A: E2E: ~800ms
```

## Component Stack

```mermaid
block-beta
    columns 3

    block:frontend:1
        columns 1
        f1["Streamlit UI"]
        f2["Query Input"]
        f3["Results View"]
    end

    block:backend:1
        columns 1
        b1["FastAPI"]
        b2["RAG Pipeline"]
        b3["Doc Processor"]
    end

    block:infra:1
        columns 1
        i1["PostgreSQL"]
        i2["pgvector"]
        i3["NVIDIA NIM"]
    end
```

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| TTFT | ~150ms | Time to first token |
| E2E Latency | ~800ms | Full query response |
| Embedding | ~50ms | Per document |
| Throughput | ~20 req/sec | Concurrent handling |

---
*Built by [QbitLoop](https://github.com/QbitLoop)*
