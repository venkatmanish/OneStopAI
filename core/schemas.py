from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


IntentName = Literal[
    "general_qa",
    "kb_qa",
    "excel_analysis",
    "web_search",
    "weather",
    "version_compare",
    "cross_source",
    "clarify",
]


class SourceRef(BaseModel):
    document_id: str | None = None
    version_id: str | None = None
    file_name: str | None = None
    source_type: str | None = None
    page: int | None = None
    sheet: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    excerpt: str | None = None


class RouteDecision(BaseModel):
    intent: IntentName
    sub_intents: list[str] = Field(default_factory=list)
    needs_memory: bool = True
    needs_rag: bool = False
    needs_web: bool = False
    tool_name: str = "general_llm"
    is_follow_up: bool = False
    is_topic_shift: bool = False
    target_source: str | None = None
    target_version: str = "latest_active"
    confidence: float = 0.0
    reason: str = ""


class AuditEvent(BaseModel):
    stage: str
    detail: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[SourceRef] = Field(default_factory=list)
    audit_trace: list[AuditEvent] = Field(default_factory=list)
    route: RouteDecision
    fallback: str | None = None
    tool_data: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    document_id: str
    file_name: str
    source_type: str
    active_version_id: str | None = None
    created_at: datetime
    updated_at: datetime


class VersionRecord(BaseModel):
    version_id: str
    document_id: str
    content_hash: str
    storage_path: str
    active: bool
    created_at: datetime


class IngestResult(BaseModel):
    document_id: str
    version_id: str
    reused_existing: bool
    chunks_indexed: int
    summary: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: SourceRef
    metadata: dict[str, Any] = Field(default_factory=dict)
