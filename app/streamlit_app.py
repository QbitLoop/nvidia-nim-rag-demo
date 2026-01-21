"""Streamlit frontend for NVIDIA NIM RAG Demo."""
import streamlit as st
import httpx
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="NVIDIA NIM RAG Demo",
    page_icon="🚀",
    layout="wide"
)


def get_health():
    """Check API health."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


def query_rag(query: str, top_k: int, threshold: float):
    """Send query to RAG API."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/query",
            json={
                "query": query,
                "top_k": top_k,
                "similarity_threshold": threshold,
                "include_sources": True
            },
            timeout=60.0
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def ingest_document(content: str, source: str):
    """Ingest document via API."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/ingest",
            json={
                "content": content,
                "source": source,
                "metadata": {"ingested_via": "streamlit"}
            },
            timeout=60.0
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_stats():
    """Get vector store stats."""
    try:
        response = httpx.get(f"{API_BASE_URL}/stats", timeout=5.0)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# Sidebar
with st.sidebar:
    st.image("https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-horiz-500x200-2c50-d.png", width=200)
    st.title("NVIDIA NIM RAG")
    st.markdown("---")

    # Health status
    health = get_health()
    if health.get("status") == "healthy":
        st.success("System Healthy")
    elif health.get("status") == "degraded":
        st.warning("System Degraded")
    else:
        st.error("System Error")

    # Stats
    stats = get_stats()
    if "error" not in stats:
        st.metric("Documents", stats.get("documents", 0))
        st.metric("Chunks", stats.get("chunks", 0))

    st.markdown("---")

    # Configuration
    st.subheader("Query Settings")
    top_k = st.slider("Top K Results", 1, 20, 5)
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.2)

    st.markdown("---")
    st.markdown("**Built with:**")
    st.markdown("- NVIDIA NIM API")
    st.markdown("- FastAPI")
    st.markdown("- PostgreSQL + pgvector")


# Main content
st.title("NVIDIA NIM RAG Demo")
st.markdown("Retrieval-Augmented Generation powered by NVIDIA NIM inference microservices")

# Tabs
tab1, tab2, tab3 = st.tabs(["Query", "Ingest Documents", "About"])

with tab1:
    st.header("Query Knowledge Base")

    query = st.text_area(
        "Enter your question:",
        placeholder="What is the main topic of the documents?",
        height=100
    )

    if st.button("Submit Query", type="primary"):
        if query:
            with st.spinner("Processing query..."):
                start_time = time.time()
                result = query_rag(query, top_k, threshold)
                elapsed = (time.time() - start_time) * 1000

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                # Display answer
                st.subheader("Answer")
                st.markdown(result.get("answer", "No answer generated"))

                # Display metrics
                st.subheader("Performance Metrics")
                metrics = result.get("metrics", {})
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Time", f"{metrics.get('total_time_ms', 0):.0f}ms")
                with col2:
                    st.metric("Embed Time", f"{metrics.get('embed_time_ms', 0):.0f}ms")
                with col3:
                    st.metric("Retrieve Time", f"{metrics.get('retrieve_time_ms', 0):.0f}ms")
                with col4:
                    st.metric("Generate Time", f"{metrics.get('generate_time_ms', 0):.0f}ms")

                # Display citations
                citations = result.get("citations", [])
                if citations:
                    st.subheader(f"Sources ({len(citations)} citations)")
                    for i, citation in enumerate(citations, 1):
                        with st.expander(f"Citation {i}: {citation.get('source', 'Unknown')} (Score: {citation.get('similarity_score', 0):.3f})"):
                            st.markdown(f"**Document ID:** `{citation.get('document_id', 'N/A')}`")
                            st.markdown(f"**Chunk Index:** {citation.get('chunk_index', 'N/A')}")
                            st.markdown("**Content:**")
                            st.text(citation.get("content", "No content"))
        else:
            st.warning("Please enter a query")

with tab2:
    st.header("Ingest Documents")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Text Input")
        doc_source = st.text_input("Document Source/Name:", placeholder="my_document.txt")
        doc_content = st.text_area(
            "Document Content:",
            placeholder="Paste your document content here...",
            height=300
        )

        if st.button("Ingest Document"):
            if doc_content and doc_source:
                with st.spinner("Ingesting document..."):
                    result = ingest_document(doc_content, doc_source)

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"Document ingested successfully!")
                    st.json(result)
            else:
                st.warning("Please provide both source name and content")

    with col2:
        st.subheader("File Upload")
        uploaded_file = st.file_uploader(
            "Upload a text file:",
            type=["txt"],
            help="Currently supports .txt files"
        )

        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            st.text_area("File Preview:", content[:1000] + "..." if len(content) > 1000 else content, height=200)

            if st.button("Ingest File"):
                with st.spinner("Ingesting file..."):
                    result = ingest_document(content, uploaded_file.name)

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.success(f"File ingested successfully!")
                    st.json(result)

with tab3:
    st.header("About This Demo")

    st.markdown("""
    ## Architecture

    This demo showcases a production-ready RAG (Retrieval-Augmented Generation) system using NVIDIA NIM inference microservices.

    ### Components

    | Component | Technology | Purpose |
    |-----------|------------|---------|
    | **LLM Inference** | NVIDIA NIM (Llama 3.1 70B) | Answer generation |
    | **Embeddings** | NVIDIA NV-Embed-QA | Semantic search |
    | **Vector Store** | PostgreSQL + pgvector | Document storage |
    | **Backend** | FastAPI | API server |
    | **Frontend** | Streamlit | User interface |

    ### Key Features

    - **Sub-second TTFT**: Fast time-to-first-token with NIM
    - **Semantic Search**: Find relevant context using embeddings
    - **Citation Tracking**: See exactly which documents informed the answer
    - **Performance Metrics**: Real-time latency measurements

    ### NVIDIA Technologies

    - **NVIDIA NIM**: Inference microservices for optimized LLM deployment
    - **NV-Embed-QA**: State-of-the-art embedding model for question answering
    - **build.nvidia.com**: API access to NVIDIA AI foundation models

    ---

    **Author:** Waseem Habib
    **GitHub:** [github.com/Qbitloop](https://github.com/Qbitloop)
    """)


# Footer
st.markdown("---")
st.markdown(
    f"*NVIDIA NIM RAG Demo v1.0 | Built with NVIDIA NIM | {datetime.now().strftime('%Y-%m-%d')}*",
    unsafe_allow_html=True
)
