from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.db import (
    Chunk,
    Conversation,
    Document,
    DocumentVersion,
    GoogleDriveFile,
    Message,
    SourceTrace,
    ToolRun,
)
from core.schemas import DocumentRecord, SourceRef, VersionRecord


class Repository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_document_version(
        self, file_name: str, source_type: str, content_hash: str, storage_path: Path
    ) -> tuple[Document, DocumentVersion, bool]:
        document_id = self._document_id_for(file_name)
        existing = self.session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.content_hash == content_hash,
            )
        )
        document = self.session.get(Document, document_id)
        if existing and document:
            return document, existing, True

        now = datetime.utcnow()
        if document is None:
            document = Document(
                document_id=document_id,
                file_name=file_name,
                source_type=source_type,
                created_at=now,
                updated_at=now,
            )
            self.session.add(document)
        else:
            self.session.query(DocumentVersion).filter(
                DocumentVersion.document_id == document_id
            ).update({"active": False})
            document.updated_at = now

        version = DocumentVersion(
            version_id=uuid.uuid4().hex,
            document_id=document_id,
            content_hash=content_hash,
            storage_path=str(storage_path),
            active=True,
            created_at=now,
        )
        self.session.add(version)
        document.active_version_id = version.version_id
        return document, version, False

    def replace_chunks(self, version_id: str, chunks: list[dict]) -> None:
        self.session.query(Chunk).filter(Chunk.version_id == version_id).delete()
        for chunk in chunks:
            self.session.add(Chunk(**chunk))

    def list_documents(self) -> list[DocumentRecord]:
        rows = self.session.scalars(select(Document).order_by(Document.updated_at.desc())).all()
        return [
            DocumentRecord(
                document_id=row.document_id,
                file_name=row.file_name,
                source_type=row.source_type,
                active_version_id=row.active_version_id,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ]

    def list_versions(self, document_id: str) -> list[VersionRecord]:
        rows = self.session.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.created_at.desc())
        ).all()
        return [
            VersionRecord(
                version_id=row.version_id,
                document_id=row.document_id,
                content_hash=row.content_hash,
                storage_path=row.storage_path,
                active=row.active,
                created_at=row.created_at,
            )
            for row in rows
        ]

    def chunks_for_active_versions(self) -> list[Chunk]:
        return self.session.scalars(
            select(Chunk)
            .join(DocumentVersion, Chunk.version_id == DocumentVersion.version_id)
            .where(DocumentVersion.active.is_(True))
        ).all()

    def chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        return self.session.scalars(select(Chunk).where(Chunk.chunk_id.in_(chunk_ids))).all()

    def get_or_create_conversation(self, session_id: str) -> Conversation:
        conversation = self.session.get(Conversation, session_id)
        if conversation is None:
            conversation = Conversation(session_id=session_id)
            self.session.add(conversation)
            self.session.flush()
        return conversation

    def list_conversations(self) -> list[dict]:
        rows = self.session.scalars(select(Conversation).order_by(Conversation.updated_at.desc())).all()
        conversations = []
        for row in rows:
            message_count = (
                self.session.query(Message).filter(Message.session_id == row.session_id).count()
            )
            last_document = (
                self.session.get(Document, row.last_document_id) if row.last_document_id else None
            )
            if message_count == 0 and last_document is None:
                continue

            last_message = self.session.scalar(
                select(Message)
                .where(Message.session_id == row.session_id)
                .order_by(Message.created_at.desc())
                .limit(1)
            )
            title = (
                row.topic
                or (last_message.content[:120] if last_message else None)
                or (f"Uploaded {last_document.file_name}" if last_document else None)
                or "Untitled chat"
            )
            conversations.append(
                {
                    "session_id": row.session_id,
                    "topic": row.topic,
                    "title": title,
                    "last_tool": row.last_tool,
                    "last_document_id": row.last_document_id,
                    "last_document_name": last_document.file_name if last_document else None,
                    "message_count": message_count,
                    "last_message": last_message.content[:120] if last_message else None,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
            )
        return conversations

    def list_messages(self, session_id: str) -> list[Message]:
        return self.session.scalars(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        ).all()

    def delete_conversation(self, session_id: str) -> bool:
        conversation = self.session.get(Conversation, session_id)
        if conversation is None:
            return False
        self.session.query(SourceTrace).filter(SourceTrace.session_id == session_id).delete()
        self.session.query(ToolRun).filter(ToolRun.session_id == session_id).delete()
        self.session.query(Message).filter(Message.session_id == session_id).delete()
        self.session.delete(conversation)
        return True

    def bind_document_to_conversation(self, session_id: str, document_id: str, topic: str | None = None) -> None:
        conversation = self.get_or_create_conversation(session_id)
        conversation.last_document_id = document_id
        conversation.topic = topic or conversation.topic
        conversation.updated_at = datetime.utcnow()

    def recent_messages(self, session_id: str, limit: int = 8) -> list[Message]:
        rows = self.session.scalars(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        ).all()
        return list(reversed(rows))

    def add_message(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        self.get_or_create_conversation(session_id)
        self.session.add(
            Message(
                session_id=session_id,
                role=role,
                content=content,
                message_metadata=metadata or {},
            )
        )

    def update_memory(
        self,
        session_id: str,
        topic: str | None,
        last_tool: str | None,
        last_document_id: str | None,
    ) -> None:
        conversation = self.get_or_create_conversation(session_id)
        conversation.topic = topic or conversation.topic
        conversation.last_tool = last_tool
        conversation.last_document_id = last_document_id or conversation.last_document_id
        conversation.updated_at = datetime.utcnow()

    def add_tool_run(self, session_id: str, tool_name: str, input_text: str, output_summary: str) -> None:
        self.session.add(
            ToolRun(
                session_id=session_id,
                tool_name=tool_name,
                input_text=input_text,
                output_summary=output_summary[:4000],
            )
        )

    def add_source_traces(self, session_id: str, sources: list[SourceRef]) -> None:
        for source in sources:
            self.session.add(SourceTrace(session_id=session_id, source=source.model_dump()))

    def upsert_google_drive_file(self, item: dict) -> GoogleDriveFile:
        row = self.session.get(GoogleDriveFile, item["file_id"])
        now = datetime.utcnow()
        if row is None:
            row = GoogleDriveFile(created_at=now, **item)
            self.session.add(row)
        else:
            for key, value in item.items():
                setattr(row, key, value)
            row.updated_at = now
        return row

    @staticmethod
    def _document_id_for(file_name: str) -> str:
        from core.storage import stable_document_id

        return stable_document_id(file_name)
