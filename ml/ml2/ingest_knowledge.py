"""
ingest_knowledge.py
-------------------
One-time script to load all knowledge base documents into ChromaDB.
Run from project root: python -m ml.ml2.ingest_knowledge
"""

import os
import sys
import glob

# Ensure project root is in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import chromadb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "wellness_knowledge"
CHUNK_SIZE = 500         # characters per chunk
CHUNK_OVERLAP = 50       # overlap between chunks for continuity


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping chunks of approximately chunk_size characters.
    Splits on paragraph boundaries when possible to maintain coherence.
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph would exceed chunk_size, save current chunk and start new
        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep last `overlap` characters for continuity
            if len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _get_category_from_filename(filename: str) -> str:
    """Infer a category tag from the filename."""
    name = os.path.splitext(filename)[0].lower()
    category_map = {
        "vital_signs_guide": "vital_signs",
        "stress_management": "stress",
        "fatigue_and_drowsiness": "fatigue",
        "breathing_exercises": "breathing",
        "when_to_seek_help": "safety",
    }
    return category_map.get(name, "general")


def ingest_all_documents():
    """
    Reads all .md files from the knowledge base directory,
    chunks them, and upserts into ChromaDB.
    """
    print("=" * 60)
    print("  ML-2 Knowledge Base Ingestion")
    print("=" * 60)

    # Initialize ChromaDB
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Delete existing collection if it exists (clean re-ingest)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[INFO] Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Wellness knowledge base for RAG guidance layer"},
    )
    print(f"[INFO] Created collection '{COLLECTION_NAME}'")

    # Find all .md files in knowledge base
    md_files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.md"))

    if not md_files:
        print(f"[ERROR] No .md files found in {KNOWLEDGE_BASE_DIR}")
        return

    print(f"[INFO] Found {len(md_files)} document(s) to ingest\n")

    total_chunks = 0

    for filepath in sorted(md_files):
        filename = os.path.basename(filepath)
        category = _get_category_from_filename(filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = _chunk_text(content)
        print(f"  [DOC] {filename} -> {len(chunks)} chunks (category: {category})")

        # Prepare batch data
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{os.path.splitext(filename)[0]}_chunk_{i:03d}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": filename,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        # Upsert into ChromaDB
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)

    print(f"\n[SUCCESS] Ingested {total_chunks} total chunks from {len(md_files)} documents")
    print(f"[INFO] ChromaDB stored at: {CHROMA_DB_DIR}")

    # Quick verification query
    print("\n--- Verification Query ---")
    results = collection.query(
        query_texts=["How to reduce stress quickly?"],
        n_results=2,
    )
    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]
        print(f"  Result {i+1} (from {source}): {doc[:100]}...")

    print("\n[DONE] Knowledge base ready for RAG queries!")


if __name__ == "__main__":
    ingest_all_documents()
