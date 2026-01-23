# NVIDIA NIM RAG Demo - Testing Guide

## Test Strategy

### Test Levels

| Level | Scope | Tools |
|-------|-------|-------|
| Unit | Individual components | pytest |
| Integration | RAG pipeline | pytest + fixtures |
| Performance | Latency measurements | locust |
| E2E | Full user flow | pytest + httpx |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html

# Performance tests
locust -f tests/perf/locustfile.py --headless -u 10 -r 2 -t 60s
```

---

## Test Files

```
tests/
├── unit/
│   ├── test_nim_client.py      # NIM API wrapper tests
│   ├── test_vector_store.py    # Vector operations tests
│   └── test_models.py          # Pydantic model tests
├── integration/
│   ├── test_rag_pipeline.py    # Full pipeline tests
│   └── test_api.py             # API endpoint tests
├── perf/
│   └── locustfile.py           # Load testing
├── fixtures/
│   └── sample_docs.py          # Test documents
└── conftest.py                 # Shared fixtures
```

---

## Test Cases

### Unit Tests

| Test | Description | Expected |
|------|-------------|----------|
| `test_nim_generate` | LLM generation call | Valid response |
| `test_nim_embed` | Embedding generation | 1024-dim vector |
| `test_vector_insert` | Document insertion | Success |
| `test_vector_search` | Similarity search | Top-k results |
| `test_chunk_text` | Text chunking | Correct chunks |

### Integration Tests

| Test | Description | Expected |
|------|-------------|----------|
| `test_rag_query` | Full RAG pipeline | Response + citations |
| `test_document_ingest` | Upload and index | Searchable doc |
| `test_api_health` | Health endpoint | 200 OK |
| `test_api_query` | Query endpoint | JSON response |

### Performance Tests

| Test | Description | Target |
|------|-------------|--------|
| `test_ttft` | Time to first token | <200ms |
| `test_e2e_latency` | Full query latency | <1000ms |
| `test_throughput` | Concurrent requests | 20 req/sec |

---

## Fixtures

### Sample Documents

```python
# tests/fixtures/sample_docs.py
SAMPLE_DOCS = [
    {
        "id": "doc_001",
        "content": "NVIDIA NIM provides fast inference...",
        "metadata": {"source": "nvidia-docs"}
    },
    # Additional test documents
]
```

### Mock NIM Client

```python
# tests/conftest.py
@pytest.fixture
def mock_nim_client():
    client = MagicMock()
    client.generate.return_value = "Mocked response"
    client.embed.return_value = [0.1] * 1024
    return client
```

---

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: pgvector/pgvector:pg16
        env:
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov

      - run: pytest tests/ -v --cov=app
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/test
          NVIDIA_API_KEY: ${{ secrets.NVIDIA_API_KEY }}
```

---

## Test Data

### Query Test Cases

| Query | Expected Behavior |
|-------|-------------------|
| "What is NIM?" | Returns NIM description |
| "How to deploy?" | Returns deployment steps |
| "" (empty) | Returns validation error |
| Long query (>1000 chars) | Handles gracefully |

### Document Test Cases

| Document Type | Test |
|---------------|------|
| PDF | Successful parsing |
| Plain text | Successful parsing |
| Empty file | Graceful error |
| Corrupt file | Graceful error |

---

## Coverage Requirements

| Component | Target Coverage |
|-----------|-----------------|
| `nim_client.py` | 90% |
| `vector_store.py` | 85% |
| `rag_pipeline.py` | 90% |
| `main.py` | 80% |

---

*Created: 2026-01-22 | Author: Waseem Habib | QbitLoop*
