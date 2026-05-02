from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from core.agent import AgentService
from core.db import init_db, session_scope
from core.ingestion import IngestionService
from core.repository import Repository
from core.schemas import ChatRequest, ChatResponse, IngestResult
from core.google_drive import GoogleDriveConnector

app = FastAPI(title="BuddyAi", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AgentService()
ingestion = IngestionService()


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return agent.run(request.session_id, request.message)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    async def event_stream():
        async for token in agent.stream(request.session_id, request.message):
            yield token

    return StreamingResponse(
        event_stream(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ingest/upload", response_model=IngestResult)
async def upload(file: UploadFile = File(...), session_id: str | None = Form(default=None)) -> IngestResult:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    result = ingestion.ingest_upload(file.filename or "uploaded_file", content)
    if session_id:
        with session_scope() as session:
            Repository(session).bind_document_to_conversation(
                session_id,
                result.document_id,
                topic=f"Uploaded {file.filename or 'uploaded_file'}",
            )
    return result


@app.post("/ingest/reindex")
def reindex_documents(force: bool = True) -> dict:
    return ingestion.reindex_active_documents(force=force)


@app.post("/google-drive/sync")
def google_drive_sync() -> dict:
    return GoogleDriveConnector().sync()


@app.get("/documents")
def documents() -> list[dict]:
    with session_scope() as session:
        return [doc.model_dump(mode="json") for doc in Repository(session).list_documents()]


@app.get("/documents/{document_id}/versions")
def versions(document_id: str) -> list[dict]:
    with session_scope() as session:
        return [
            version.model_dump(mode="json")
            for version in Repository(session).list_versions(document_id)
        ]


@app.get("/conversations")
def conversations() -> list[dict]:
    with session_scope() as session:
        return Repository(session).list_conversations()


@app.get("/conversations/{session_id}/messages")
def conversation_messages(session_id: str) -> list[dict]:
    with session_scope() as session:
        return [
            {
                "role": message.role,
                "content": message.content,
                "metadata": message.message_metadata,
                "created_at": message.created_at.isoformat(),
            }
            for message in Repository(session).list_messages(session_id)
        ]


@app.delete("/conversations/{session_id}")
def delete_conversation(session_id: str) -> dict:
    with session_scope() as session:
        deleted = Repository(session).delete_conversation(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"deleted": True, "session_id": session_id}
