from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

from core.settings import get_settings


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_document_id(file_name: str) -> str:
    stem = Path(file_name).stem.lower().strip().replace(" ", "-")
    stem = "".join(ch for ch in stem if ch.isalnum() or ch in {"-", "_"})
    return stem[:48] or uuid.uuid4().hex


class LocalObjectStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or get_settings().local_storage_dir
        self.uploads_dir = self.root / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, file_name: str, content: bytes, content_hash: str) -> Path:
        suffix = Path(file_name).suffix
        path = self.uploads_dir / f"{content_hash}{suffix}"
        if not path.exists():
            path.write_bytes(content)
        return path

    def copy_into_store(self, source_path: Path, content_hash: str) -> Path:
        suffix = source_path.suffix
        target = self.uploads_dir / f"{content_hash}{suffix}"
        if not target.exists():
            shutil.copy2(source_path, target)
        return target
