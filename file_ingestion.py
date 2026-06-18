"""
file_ingestion.py — QuickShip Customer File Upload & RAG Ingestion

Supports: PDF, TXT files uploaded by customers.
Chunks the content, embeds it, and stores it in the existing Chroma vector DB.
Each document is tagged with a unique upload_id so it can be retrieved or removed later.
"""

import os
import uuid
import logging
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Optional PDF support (graceful degradation if not installed) ---
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("pdfplumber not installed. PDF uploads will be rejected. Run: pip install pdfplumber")

# --- Config ---
CHROMA_DIR = "./chroma_db"
MAX_FILE_SIZE_MB = 5
ALLOWED_EXTENSIONS = {".txt", ".pdf"}

# Chunk settings: balance between context richness and retrieval precision
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

# --- Setup ---
logging.basicConfig(
    filename="quickship_agent.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ─────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────

def _get_vector_db() -> Chroma:
    """Returns the shared Chroma instance."""
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def _file_hash(file_bytes: bytes) -> str:
    """SHA-256 fingerprint to detect duplicate uploads."""
    return hashlib.sha256(file_bytes).hexdigest()


def _extract_text_from_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_text_from_pdf(file_path: str) -> str:
    if not PDF_SUPPORT:
        raise RuntimeError("PDF support not available. Install pdfplumber: pip install pdfplumber")
    
    text_parts = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"[Page {page_num}]\n{page_text}")
    
    if not text_parts:
        raise ValueError("PDF appears to be scanned/image-only. No extractable text found.")
    
    return "\n\n".join(text_parts)


def _sanitize_content(text: str) -> str:
    """
    Guardrail: strip prompt injection attempts from user-uploaded content
    before it enters the vector store.
    """
    injection_patterns = [
        "ignore previous",
        "ignore all instructions",
        "system prompt",
        "as an agent",
        "you must now",
        "disregard your",
        "new instructions:",
        "override:",
        "forget everything",
    ]
    lower = text.lower()
    for pattern in injection_patterns:
        if pattern in lower:
            logging.warning(f"[UPLOAD GUARDRAIL] Injection pattern detected in upload: '{pattern}'. Content flagged.")
            # Replace just the offending sentence rather than rejecting the whole file
            text = text.replace(pattern, "[REDACTED]")
    return text


def _chunk_text(text: str, source_name: str, upload_id: str) -> list[Document]:
    """
    Splits text into overlapping chunks and wraps each as a LangChain Document
    with rich metadata for later filtering/attribution.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "upload_id": upload_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "type": "customer_upload"   # distinguishes from built-in policies
            }
        ))
    return docs


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def ingest_file(file_path: str, original_filename: str) -> dict:
    """
    Main entry point. Validates, extracts, chunks, and stores a customer file.

    Args:
        file_path:         Absolute or relative path to the saved file on disk.
        original_filename: The filename as provided by the customer (used for metadata).

    Returns:
        dict with keys: upload_id, filename, chunks_stored, message
    
    Raises:
        ValueError: on unsupported type, oversized file, or empty content.
        RuntimeError: on PDF support missing.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # --- Validation ---
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    file_bytes = path.read_bytes()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB.")
    
    # --- Duplicate detection ---
    fingerprint = _file_hash(file_bytes)
    vector_db = _get_vector_db()
    existing = vector_db.get(where={"source": original_filename})
    if existing and existing.get("ids"):
        logging.info(f"[UPLOAD] Duplicate detected for '{original_filename}'. Skipping re-ingestion.")
        return {
            "upload_id": existing["metadatas"][0].get("upload_id", "unknown"),
            "filename": original_filename,
            "chunks_stored": 0,
            "message": f"'{original_filename}' was already uploaded. Using existing version."
        }

    # --- Text Extraction ---
    if ext == ".txt":
        raw_text = _extract_text_from_txt(file_path)
    elif ext == ".pdf":
        raw_text = _extract_text_from_pdf(file_path)
    
    if not raw_text or not raw_text.strip():
        raise ValueError("File appears to be empty or contains no readable text.")

    # --- Sanitize & Chunk ---
    clean_text = _sanitize_content(raw_text)
    upload_id = str(uuid.uuid4())
    docs = _chunk_text(clean_text, source_name=original_filename, upload_id=upload_id)

    # --- Store in Chroma ---
    vector_db.add_documents(docs)
    
    logging.info(f"[UPLOAD] Ingested '{original_filename}' → upload_id={upload_id}, chunks={len(docs)}")
    
    return {
        "upload_id": upload_id,
        "filename": original_filename,
        "chunks_stored": len(docs),
        "message": f"'{original_filename}' successfully uploaded and indexed ({len(docs)} chunks)."
    }


def list_uploaded_files() -> list[dict]:
    """
    Returns metadata for all customer-uploaded files currently in the vector store.
    Useful for showing customers what files are on record.
    """
    vector_db = _get_vector_db()
    results = vector_db.get(where={"type": "customer_upload"})
    
    seen = {}
    for meta in results.get("metadatas", []):
        uid = meta.get("upload_id")
        if uid and uid not in seen:
            seen[uid] = {
                "upload_id": uid,
                "filename": meta.get("source", "unknown"),
                "chunks": meta.get("total_chunks", "?")
            }
    return list(seen.values())


def remove_uploaded_file(upload_id: str) -> dict:
    """
    Removes all chunks for a given upload_id from the vector store.
    Allows customers (or admins) to delete previously uploaded documents.
    """
    vector_db = _get_vector_db()
    results = vector_db.get(where={"upload_id": upload_id})
    ids_to_delete = results.get("ids", [])
    
    if not ids_to_delete:
        return {"success": False, "message": f"No document found with upload_id '{upload_id}'."}
    
    vector_db.delete(ids=ids_to_delete)
    logging.info(f"[UPLOAD] Removed upload_id={upload_id}, deleted {len(ids_to_delete)} chunks.")
    
    return {
        "success": True,
        "upload_id": upload_id,
        "chunks_removed": len(ids_to_delete),
        "message": f"Document removed successfully ({len(ids_to_delete)} chunks deleted)."
    }