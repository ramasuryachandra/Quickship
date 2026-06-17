import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

load_dotenv()

from agent import run_agent
from scheduler import start_scheduler

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield


app = FastAPI(title="Calendar Agent", lifespan=lifespan)


# ── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    response: str
    history: list[dict]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY is not set in .env")

    reply = await run_agent(req.message, req.history)

    updated_history = req.history + [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": reply},
    ]
    return ChatResponse(response=reply, history=updated_history)


@app.post("/upload-calendar")
async def upload_calendar(file: UploadFile = File(...)):
    if not file.filename.endswith(".ics"):
        raise HTTPException(status_code=400, detail="Only .ics files are accepted.")

    calendar_path = os.getenv("CALENDAR_PATH", "calendar.ics")
    with open(calendar_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"message": f"Calendar '{file.filename}' uploaded successfully."}


@app.get("/calendar-status")
async def calendar_status():
    path = Path(os.getenv("CALENDAR_PATH", "calendar.ics"))
    return {"loaded": path.exists(), "path": str(path)}


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
