"""
chat_api.py — QuickShip Chatbot REST API
=========================================
A FastAPI gateway that exposes the QuickShip AI agent over HTTP.

Endpoints:
  POST /chat            — Send a text message, get an agent reply
  POST /chat/upload     — Send a file + optional message; agent ingests & replies
  GET  /files           — List all customer-uploaded files in the vector store
  DELETE /files/{uid}   — Remove a previously uploaded file by upload_id
  GET  /health          — Liveness check

Usage:
  uvicorn chat_api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import shutil
import tempfile
import logging
from typing import Annotated

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ─── Local agent imports ────────────────────────────────────────────────────
# agent1.py and file_ingestion.py must be on the Python path (same directory)
from agent1 import run_quickship_agent
from file_ingestion import list_uploaded_files, remove_uploaded_file

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="chat_api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="QuickShip Chat API",
    description="REST interface for the QuickShip AI support agent.",
    version="1.0.0",
)

# Allow browser-based clients (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp directory for uploaded files during request processing
UPLOAD_TEMP_DIR = tempfile.mkdtemp(prefix="quickship_uploads_")

# Allowed MIME types → file extensions
ALLOWED_CONTENT_TYPES: dict[str, str] = {
    "text/plain": ".txt",
    "application/pdf": ".pdf",
}

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


# ─── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Customer message")
    has_upload_context: bool = Field(
        False,
        description="Pass true if the customer already uploaded a file in a previous turn",
    )
    session_id: str | None = Field(None, description="Optional opaque session identifier for logging")


class ChatResponse(BaseModel):
    reply: str
    session_id: str | None = None


class UploadChatResponse(BaseModel):
    reply: str
    upload_id: str | None = None
    chunks_stored: int | None = None
    filename: str | None = None


class FileEntry(BaseModel):
    upload_id: str
    filename: str
    chunks: int | str


class FilesResponse(BaseModel):
    files: list[FileEntry]


class DeleteResponse(BaseModel):
    success: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str


# ─── Exception handler ────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
def health_check():
    """Liveness probe — returns 200 when the service is up."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(body: ChatRequest):
    """
    Send a plain-text message to the QuickShip agent.

    - **message**: The customer's question or request.
    - **has_upload_context**: Set to `true` if the customer previously uploaded
      a file and wants to ask follow-up questions about it.
    - **session_id**: Optional opaque string forwarded back in the response for
      client-side session tracking (not used internally).
    """
    logger.info(f"[CHAT] session={body.session_id} message={body.message!r}")

    reply = run_quickship_agent(
        user_input=body.message,
        has_upload_context=body.has_upload_context,
    )

    return ChatResponse(reply=reply, session_id=body.session_id)


@app.post("/chat/upload", response_model=UploadChatResponse, tags=["Chat"])
async def chat_with_upload(
    file: Annotated[UploadFile, File(description="PDF or TXT file to attach")],
    message: Annotated[
        str,
        Form(description="Optional customer message about the file (can be empty)"),
    ] = "",
):
    """
    Upload a file and (optionally) ask a question about it in one request.

    - **file**: A `.txt` or `.pdf` file (max 5 MB).
    - **message**: Anything the customer wants to say alongside the upload.
      Leave empty to just store the file.

    The agent ingests the file into the RAG store and returns a natural-language
    reply. The `upload_id` in the response can be used with `DELETE /files/{uid}`
    to remove the file later.
    """
    # ── Content-type guard ──────────────────────────────────────────────────
    content_type = file.content_type or ""
    # Strip charset suffix, e.g. "text/plain; charset=utf-8"
    base_ct = content_type.split(";")[0].strip()
    if base_ct not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Allowed: PDF and plain text.",
        )

    # ── Size guard (stream to temp file) ───────────────────────────────────
    ext = ALLOWED_CONTENT_TYPES[base_ct]
    safe_name = os.path.basename(file.filename or f"upload{ext}")
    tmp_path = os.path.join(UPLOAD_TEMP_DIR, safe_name)

    bytes_written = 0
    try:
        with open(tmp_path, "wb") as out:
            while chunk := await file.read(65536):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the 5 MB limit.",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[UPLOAD] Write error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    # ── Delegate to agent ───────────────────────────────────────────────────
    logger.info(f"[UPLOAD] file={safe_name} size={bytes_written} msg={message!r}")
    try:
        reply = run_quickship_agent(
            user_input=message,
            uploaded_file_path=tmp_path,
            uploaded_filename=safe_name,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    finally:
        # Clean up temp file regardless of outcome
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # ── Retrieve upload metadata from the store for the response ───────────
    upload_id = None
    chunks_stored = None
    try:
        all_files = list_uploaded_files()
        match = next((f for f in all_files if f.get("filename") == safe_name), None)
        if match:
            upload_id = match.get("upload_id")
            chunks = match.get("chunks")
            chunks_stored = int(chunks) if isinstance(chunks, (int, str)) else None
    except Exception:
        pass  # Non-critical; reply is the important part

    return UploadChatResponse(
        reply=reply,
        upload_id=upload_id,
        chunks_stored=chunks_stored,
        filename=safe_name,
    )


@app.get("/files", response_model=FilesResponse, tags=["Files"])
def get_uploaded_files():
    """
    List all customer-uploaded files currently stored in the vector database.
    Returns metadata: upload_id, filename, and chunk count.
    """
    try:
        files = list_uploaded_files()
    except Exception as exc:
        logger.error(f"[FILES] list error: {exc}")
        raise HTTPException(status_code=500, detail="Could not retrieve file list.")

    entries = [
        FileEntry(
            upload_id=f.get("upload_id", ""),
            filename=f.get("filename", "unknown"),
            chunks=f.get("chunks", "?"),
        )
        for f in files
    ]
    return FilesResponse(files=entries)


@app.delete("/files/{upload_id}", response_model=DeleteResponse, tags=["Files"])
def delete_uploaded_file(upload_id: str):
    """
    Remove a previously uploaded file from the vector store by its `upload_id`.
    All embedded chunks for that file are permanently deleted.
    """
    try:
        result = remove_uploaded_file(upload_id)
    except Exception as exc:
        logger.error(f"[FILES] delete error upload_id={upload_id}: {exc}")
        raise HTTPException(status_code=500, detail="Could not delete file.")

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message", "File not found."))

    return DeleteResponse(success=True, message=result.get("message", "Deleted."))