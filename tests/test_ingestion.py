import importlib.util
import tempfile
from pathlib import Path

import pytest

from src.knowledge.chunking import ChunkConfig, ChunkingStrategy
from src.knowledge.ingestion import (
    MAX_CONTENT_LENGTH,
    MAX_FILE_SIZE_BYTES,
    DocumentIngester,
)

HAS_CHROMADB = importlib.util.find_spec("langchain_chroma") is not None

requires_chromadb = pytest.mark.skipif(not HAS_CHROMADB, reason="langchain_chroma not installed")


class TestDocumentIngesterValidation:
    def test_validate_content_length_exceeds_max(self) -> None:
        ingester = DocumentIngester(max_content_length=100)
        with pytest.raises(ValueError, match=r"Content length .* exceeds maximum"):
            ingester._validate_content_length("x" * 101)

    def test_validate_content_length_within_limit(self) -> None:
        ingester = DocumentIngester(max_content_length=100)
        ingester._validate_content_length("x" * 100)

    def test_validate_file_size_exceeds_max(self) -> None:
        ingester = DocumentIngester(max_file_size=100)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"x" * 101)
            f.flush()
            path = Path(f.name)
        try:
            with pytest.raises(ValueError, match=r"File size .* exceeds maximum"):
                ingester._validate_file_size(path)
        finally:
            path.unlink()

    def test_validate_file_size_within_limit(self) -> None:
        ingester = DocumentIngester(max_file_size=100)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"x" * 50)
            f.flush()
            path = Path(f.name)
        try:
            ingester._validate_file_size(path)
        finally:
            path.unlink()


class TestDocumentIngesterTextIngestion:
    def test_ingest_empty_content_fails(self) -> None:
        ingester = DocumentIngester()
        result = ingester.ingest_text("")
        assert result.success is False
        assert result.error == "Empty content"

    def test_ingest_whitespace_only_fails(self) -> None:
        ingester = DocumentIngester()
        result = ingester.ingest_text("   \n\t  ")
        assert result.success is False
        assert result.error == "Empty content"

    def test_ingest_text_exceeds_max_content_length(self) -> None:
        ingester = DocumentIngester(max_content_length=100)
        result = ingester.ingest_text("x" * 200)
        assert result.success is False
        assert "exceeds maximum" in result.error

    def test_generate_doc_id_deterministic(self) -> None:
        ingester = DocumentIngester()
        content_hash = ingester._content_hash("content")
        id1 = ingester._generate_doc_id(content_hash)
        id2 = ingester._generate_doc_id(content_hash)
        assert id1 == id2

    def test_generate_doc_id_different_for_different_content(self) -> None:
        ingester = DocumentIngester()
        hash1 = ingester._content_hash("content1")
        hash2 = ingester._content_hash("content2")
        id1 = ingester._generate_doc_id(hash1)
        id2 = ingester._generate_doc_id(hash2)
        assert id1 != id2

    def test_content_hash_deterministic(self) -> None:
        ingester = DocumentIngester()
        hash1 = ingester._content_hash("test content")
        hash2 = ingester._content_hash("test content")
        assert hash1 == hash2

    def test_content_hash_different_for_different_content(self) -> None:
        ingester = DocumentIngester()
        hash1 = ingester._content_hash("content1")
        hash2 = ingester._content_hash("content2")
        assert hash1 != hash2


class TestDocumentIngesterFileIngestion:
    def test_ingest_file_not_found(self) -> None:
        ingester = DocumentIngester()
        result = ingester.ingest_file("/nonexistent/path/file.txt")
        assert result.success is False
        assert "File not found" in result.error

    @requires_chromadb
    def test_ingest_text_file(self) -> None:
        ingester = DocumentIngester()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as f:
            f.write("This is test content for ingestion.")
            path = Path(f.name)
        try:
            result = ingester.ingest_file(path)
            assert result.success is True
            assert result.chunks_created > 0
            assert result.filename == path.name
        finally:
            path.unlink()

    @requires_chromadb
    def test_ingest_markdown_file(self) -> None:
        ingester = DocumentIngester()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as f:
            f.write("# Title\n\nThis is markdown content.\n\n## Section\n\nMore content.")
            path = Path(f.name)
        try:
            result = ingester.ingest_file(path)
            assert result.success is True
            assert result.chunks_created > 0
        finally:
            path.unlink()

    @requires_chromadb
    def test_ingest_python_file(self) -> None:
        ingester = DocumentIngester()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as f:
            f.write("def hello():\n    print('Hello')\n\nclass Test:\n    pass")
            path = Path(f.name)
        try:
            result = ingester.ingest_file(path)
            assert result.success is True
            assert result.chunks_created > 0
        finally:
            path.unlink()

    @requires_chromadb
    def test_ingest_csv_file(self) -> None:
        ingester = DocumentIngester()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as f:
            f.write("name,age,city\nAlice,30,NYC\nBob,25,LA")
            path = Path(f.name)
        try:
            result = ingester.ingest_file(path)
            assert result.success is True
        finally:
            path.unlink()

    @requires_chromadb
    def test_ingest_html_file(self) -> None:
        ingester = DocumentIngester()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
            f.write("<html><body><h1>Title</h1><p>Content here</p></body></html>")
            path = Path(f.name)
        try:
            result = ingester.ingest_file(path)
            assert result.success is True
        finally:
            path.unlink()


class TestDocumentIngesterUpload:
    @requires_chromadb
    def test_ingest_upload_text(self) -> None:
        ingester = DocumentIngester()
        data = b"This is uploaded text content."
        result = ingester.ingest_upload(data, "upload.txt")
        assert result.success is True
        assert result.filename == "upload.txt"

    def test_ingest_upload_exceeds_size_limit(self) -> None:
        ingester = DocumentIngester(max_file_size=100)
        data = b"x" * 200
        result = ingester.ingest_upload(data, "large.txt")
        assert result.success is False
        assert "exceeds maximum" in result.error

    def test_ingest_upload_binary_unsupported(self) -> None:
        ingester = DocumentIngester()
        data = bytes([0x00, 0x01, 0x02, 0xFF, 0xFE])
        result = ingester.ingest_upload(data, "binary.bin")
        assert result.success is False
        assert "Cannot decode" in result.error


class TestDocumentIngesterDocumentManagement:
    def test_list_documents_empty(self) -> None:
        ingester = DocumentIngester()
        docs = ingester.list_documents()
        assert docs == []

    def test_get_document_nonexistent(self) -> None:
        ingester = DocumentIngester()
        doc = ingester.get_document("nonexistent")
        assert doc is None

    def test_delete_document_nonexistent(self) -> None:
        ingester = DocumentIngester()
        result = ingester.delete_document("nonexistent")
        assert result is False


class TestChunkConfig:
    def test_default_config(self) -> None:
        config = ChunkConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.strategy == ChunkingStrategy.AUTO

    def test_custom_config(self) -> None:
        config = ChunkConfig(
            chunk_size=500,
            chunk_overlap=100,
            strategy=ChunkingStrategy.MARKDOWN,
        )
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.strategy == ChunkingStrategy.MARKDOWN


class TestConstants:
    def test_max_file_size_bytes(self) -> None:
        assert MAX_FILE_SIZE_BYTES == 50 * 1024 * 1024

    def test_max_content_length(self) -> None:
        assert MAX_CONTENT_LENGTH == 10_000_000
