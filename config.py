"""Configuration for Budget RAG Chatbot."""

import os

# Try Streamlit secrets first, then fall back to .env
try:
    import streamlit as st
    _secrets = st.secrets
except Exception:
    _secrets = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get(key: str, default: str = "") -> str:
    """Get config value from Streamlit secrets or environment variables."""
    if _secrets:
        try:
            return _secrets[key]
        except (KeyError, Exception):
            pass
    return os.getenv(key, default)

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY = _get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = _get("PINECONE_INDEX_NAME", "budget-rag")
PINECONE_HOST_URL = _get("PINECONE_HOST_URL")
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "multilingual-e5-large")

# ── Cerebras LLM ─────────────────────────────────────────────────────────────
CEREBRAS_API_KEY = _get("CEREBRAS_API_KEY")
CEREBRAS_MODEL = _get("CEREBRAS_MODEL", "zai-glm-4.7")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(DATA_DIR, "docs")
PARENT_STORE_PATH = os.path.join(DATA_DIR, "parent_store.json")
CHAT_HISTORY_PATH = os.path.join(DATA_DIR, "chat_history.json")

# ── PDF files ─────────────────────────────────────────────────────────────────
PDF_FILES = {
    "Annual_Budget_Statement.pdf": "english",
    "Budget_in_Brief.pdf": "english",
    "FM_Speech_Urdu.pdf": "urdu",
}

# ── Retrieval config ──────────────────────────────────────────────────────────
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 80
TOP_K_PER_QUERY = 8
HISTORY_SIMILARITY_THRESHOLD = 0.92
PINECONE_BATCH_SIZE = 50

# ── Pinecone namespaces ───────────────────────────────────────────────────────
NS_DOCS = "budget_docs"
NS_HISTORY = "qa_history"
