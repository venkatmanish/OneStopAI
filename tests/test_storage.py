from core.storage import sha256_bytes, stable_document_id
from core.vector_store import VectorStore


def test_sha256_is_stable():
    assert sha256_bytes(b"abc") == sha256_bytes(b"abc")
    assert sha256_bytes(b"abc") != sha256_bytes(b"abcd")


def test_document_id_is_filename_based():
    assert stable_document_id("Revenue Report 2025.xlsx") == "revenue-report-2025"


def test_vector_collection_name_includes_embedding_model_fingerprint():
    first = VectorStore.collection_name("chunks", "BAAI/bge-base-en-v1.5")
    second = VectorStore.collection_name("chunks", "sentence-transformers/all-MiniLM-L6-v2")

    assert first.startswith("chunks_baai_bge_base_en_v1_5_")
    assert first != second
