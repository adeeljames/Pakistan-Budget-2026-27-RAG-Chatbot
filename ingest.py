"""One-time document ingestion: PDFs → chunk → embed → Pinecone."""

import json
import os
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from tqdm import tqdm

import config


def init_pinecone() -> Pinecone:
    return Pinecone(api_key=config.PINECONE_API_KEY)


def load_pdf(filepath: str, source: str, language: str):
    """Load a PDF and tag each page with metadata."""
    loader = PyPDFLoader(filepath)
    pages = loader.load()
    for page in pages:
        page.metadata.update({
            "source": source,
            "language": language,
            "page": page.metadata.get("page", 0),
        })
    return pages


def create_parent_child_chunks(documents):
    """Split documents into parent (large) and child (small) chunks."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_docs = parent_splitter.split_documents(documents)
    # Assign stable IDs to parents
    for doc in parent_docs:
        doc.metadata["parent_id"] = str(uuid.uuid4())

    child_docs = []
    for parent in parent_docs:
        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent.metadata["parent_id"]
            child.metadata["source"] = parent.metadata["source"]
            child.metadata["language"] = parent.metadata["language"]
            child.metadata["page"] = parent.metadata.get("page", 0)
            child.metadata["type"] = "budget_doc"
        child_docs.extend(children)

    return parent_docs, child_docs


def embed_texts(pc: Pinecone, texts: list[str]) -> list[list[float]]:
    """Embed texts using Pinecone hosted inference API."""
    all_embeddings = []
    for i in range(0, len(texts), config.PINECONE_BATCH_SIZE):
        batch = texts[i : i + config.PINECONE_BATCH_SIZE]
        result = pc.inference.embed(
            model=config.EMBEDDING_MODEL,
            inputs=batch,
            parameters={"input_type": "passage", "truncate": "END"},
        )
        # result.data is a list of Embedding objects with .values attribute
        for emb in result.data:
            all_embeddings.append(emb.values)
    return all_embeddings


def upsert_to_pinecone(pc: Pinecone, child_docs, embeddings):
    """Upsert embedded child chunks into Pinecone."""
    idx = pc.Index(name=config.PINECONE_INDEX_NAME, host=config.PINECONE_HOST_URL)

    vectors = []
    for doc, emb in zip(child_docs, embeddings):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": emb,
            "metadata": {
                "parent_id": doc.metadata["parent_id"],
                "source": doc.metadata["source"],
                "language": doc.metadata["language"],
                "page": int(doc.metadata.get("page", 0)),
                "type": "budget_doc",
                "text_preview": doc.page_content[:200],
            },
        })

    for i in tqdm(range(0, len(vectors), config.PINECONE_BATCH_SIZE), desc="Upserting to Pinecone"):
        batch = vectors[i : i + config.PINECONE_BATCH_SIZE]
        idx.upsert(vectors=batch, namespace=config.NS_DOCS)


def save_parent_store(parent_docs):
    """Persist parent documents to a local JSON file."""
    store = {}
    for doc in parent_docs:
        store[doc.metadata["parent_id"]] = {
            "content": doc.page_content,
            "metadata": {
                "source": doc.metadata["source"],
                "language": doc.metadata["language"],
                "page": int(doc.metadata.get("page", 0)),
            },
        }
    with open(config.PARENT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)
    print(f"💾  Saved {len(store)} parent documents to {config.PARENT_STORE_PATH}")


def main():
    print("🚀 Budget RAG – Document Ingestion Pipeline\n")

    pc = init_pinecone()

    # Check if index already has data
    idx = pc.Index(name=config.PINECONE_INDEX_NAME, host=config.PINECONE_HOST_URL)
    stats = idx.describe_index_stats()
    ns_stats = stats.get("namespaces", {}).get(config.NS_DOCS, {})
    existing_count = ns_stats.get("record_count", 0)

    if existing_count > 0:
        print(f"⚠️  Index already has {existing_count} vectors in '{config.NS_DOCS}' namespace.")
        print("   Skipping ingestion. Delete vectors manually to re-ingest.\n")
        return

    # ── Step 1: Load PDFs ─────────────────────────────────────────────────
    print("📄 Loading PDF files...")
    all_docs = []
    for filename, language in config.PDF_FILES.items():
        filepath = os.path.join(config.DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"   ⚠️  {filename} not found, skipping.")
            continue
        pages = load_pdf(filepath, filename, language)
        all_docs.extend(pages)
        print(f"   ✅ {filename} – {len(pages)} pages ({language})")

    if not all_docs:
        print("❌ No documents found. Aborting.")
        return

    # ── Step 2: Create parent/child chunks ────────────────────────────────
    print("\n✂️  Creating parent/child chunks...")
    parent_docs, child_docs = create_parent_child_chunks(all_docs)
    print(f"   Parents: {len(parent_docs)} | Children: {len(child_docs)}")

    # ── Step 3: Embed child chunks ────────────────────────────────────────
    print("\n🔢 Embedding child chunks via Pinecone Inference API...")
    texts = [doc.page_content for doc in child_docs]
    embeddings = embed_texts(pc, texts)
    print(f"   ✅ Embedded {len(embeddings)} chunks")

    # ── Step 4: Upsert to Pinecone ────────────────────────────────────────
    print("\n📤 Upserting vectors to Pinecone...")
    upsert_to_pinecone(pc, child_docs, embeddings)

    # ── Step 5: Save parent store ─────────────────────────────────────────
    print("\n💾 Saving parent documents locally...")
    save_parent_store(parent_docs)

    # ── Final stats ───────────────────────────────────────────────────────
    final_stats = idx.describe_index_stats()
    total = final_stats.get("total_vector_count", 0)
    print(f"\n✅ Ingestion complete! Total vectors in index: {total}")


if __name__ == "__main__":
    main()
