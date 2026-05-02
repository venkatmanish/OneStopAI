from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from core.db import session_scope
from core.ingestion import IngestionService
from core.repository import Repository
from core.settings import get_settings


GOOGLE_EXPORTS = {
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
}

SUPPORTED_BINARY_PREFIXES = ("application/pdf", "image/")
SUPPORTED_BINARY_MIMES = {
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


class GoogleDriveConnector:
    def configured(self) -> bool:
        settings = get_settings()
        return bool(settings.google_drive_folder_id and settings.google_service_account_file)

    def sync(self) -> dict:
        settings = get_settings()
        if not self.configured():
            return {
                "status": "not_configured",
                "message": (
                    "Set GOOGLE_DRIVE_FOLDER_ID and GOOGLE_SERVICE_ACCOUNT_FILE. "
                    "Share the Drive folder with the service account email."
                ),
                "changed": 0,
            }
        service_account_path = Path(settings.google_service_account_file or "")
        if not service_account_path.exists():
            return {
                "status": "not_configured",
                "message": f"Service account file not found: {service_account_path}",
                "changed": 0,
            }

        try:
            service = self._service(service_account_path)
        except ModuleNotFoundError:
            return {
                "status": "missing_dependency",
                "message": "Install google-api-python-client and google-auth to enable Drive sync.",
                "changed": 0,
            }

        files = self._list_files(service)
        changed = 0
        skipped = 0
        unsupported = 0
        errors: list[str] = []

        for item in files:
            if not self._supported(item):
                unsupported += 1
                continue
            if not self._changed(item):
                skipped += 1
                continue
            try:
                file_name, content = self._download(service, item)
                result = IngestionService().ingest_upload(file_name, content, source_type="google_drive")
                self._record_file(item, result.document_id)
                changed += 1
            except Exception as exc:
                errors.append(f"{item.get('name', item.get('id'))}: {exc}")

        return {
            "status": "ok" if not errors else "partial",
            "message": "Google Drive folder sync completed.",
            "folder_id": settings.google_drive_folder_id,
            "shared_drive_id": settings.google_drive_shared_drive_id,
            "seen": len(files),
            "changed": changed,
            "skipped": skipped,
            "unsupported": unsupported,
            "errors": errors,
        }

    def _service(self, service_account_path: Path):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        credentials = service_account.Credentials.from_service_account_file(
            str(service_account_path),
            scopes=scopes,
        )
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    def _list_files(self, service) -> list[dict[str, Any]]:
        settings = get_settings()
        files: list[dict[str, Any]] = []
        page_token = None
        while True:
            params = {
                "q": f"'{settings.google_drive_folder_id}' in parents and trashed = false",
                "fields": (
                    "nextPageToken, files("
                    "id,name,mimeType,md5Checksum,modifiedTime,webViewLink,version)"
                ),
                "pageSize": 100,
                "pageToken": page_token,
                "supportsAllDrives": True,
                "includeItemsFromAllDrives": True,
            }
            if settings.google_drive_shared_drive_id:
                params.update(
                    {
                        "corpora": "drive",
                        "driveId": settings.google_drive_shared_drive_id,
                    }
                )
            response = service.files().list(**params).execute()
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return files

    def _changed(self, item: dict[str, Any]) -> bool:
        from core.db import Chunk, DocumentVersion, GoogleDriveFile

        file_id = item["id"]
        checksum = item.get("md5Checksum")
        revision_id = str(item.get("version") or "")
        modified = item.get("modifiedTime")
        with session_scope() as session:
            row = session.get(GoogleDriveFile, file_id)
            if row is None:
                return True
            active_chunks = (
                session.query(Chunk)
                .join(DocumentVersion, Chunk.version_id == DocumentVersion.version_id)
                .filter(
                    Chunk.document_id == row.document_id,
                    DocumentVersion.active.is_(True),
                )
                .all()
            )
            if not active_chunks or IngestionService._needs_reindex(active_chunks):
                return True
            if checksum and row.md5_checksum != checksum:
                return True
            if revision_id and row.revision_id != revision_id:
                return True
            return row.last_modified != modified

    def _download(self, service, item: dict[str, Any]) -> tuple[str, bytes]:
        from googleapiclient.http import MediaIoBaseDownload

        mime_type = item.get("mimeType", "")
        file_name = item.get("name", item["id"])
        if mime_type in GOOGLE_EXPORTS:
            export_mime, suffix = GOOGLE_EXPORTS[mime_type]
            request = service.files().export_media(fileId=item["id"], mimeType=export_mime)
            if not Path(file_name).suffix:
                file_name = f"{file_name}{suffix}"
        else:
            request = service.files().get_media(fileId=item["id"], supportsAllDrives=True)

        buffer = BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return file_name, buffer.getvalue()

    def _record_file(self, item: dict[str, Any], document_id: str) -> None:
        with session_scope() as session:
            Repository(session).upsert_google_drive_file(
                {
                    "file_id": item["id"],
                    "document_id": document_id,
                    "md5_checksum": item.get("md5Checksum"),
                    "revision_id": str(item.get("version") or ""),
                    "last_modified": item.get("modifiedTime"),
                    "name": item.get("name", item["id"]),
                    "web_url": item.get("webViewLink"),
                    "mime_type": item.get("mimeType"),
                }
            )

    @staticmethod
    def _supported(item: dict[str, Any]) -> bool:
        mime_type = item.get("mimeType", "")
        return (
            mime_type in GOOGLE_EXPORTS
            or mime_type in SUPPORTED_BINARY_MIMES
            or mime_type.startswith(SUPPORTED_BINARY_PREFIXES)
        )
