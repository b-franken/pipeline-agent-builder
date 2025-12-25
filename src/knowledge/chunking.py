from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    Language,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)


class ChunkingStrategy(Enum):
    AUTO = "auto"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    CODE = "code"


@dataclass
class ChunkConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    strategy: ChunkingStrategy = ChunkingStrategy.AUTO


def _detect_content_type(content: str, filename: str | None = None) -> ChunkingStrategy:
    if filename:
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        if ext in ("py", "js", "ts", "java", "go", "rs", "cpp", "c", "rb"):
            return ChunkingStrategy.CODE
        if ext in ("md", "markdown"):
            return ChunkingStrategy.MARKDOWN

    lines = content.split("\n")[:50]

    code_indicators = sum(
        1
        for line in lines
        if any(kw in line for kw in ["def ", "class ", "import ", "function ", "const ", "let ", "var "])
    )
    md_indicators = sum(1 for line in lines if line.startswith(("#", "-", "*", ">")))

    if code_indicators > 5:
        return ChunkingStrategy.CODE
    if md_indicators > 5:
        return ChunkingStrategy.MARKDOWN

    return ChunkingStrategy.RECURSIVE


def _get_splitter(strategy: ChunkingStrategy, config: ChunkConfig, language: str | None = None) -> Any:
    if strategy == ChunkingStrategy.MARKDOWN:
        return MarkdownTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    if strategy == ChunkingStrategy.CODE:
        if language:
            try:
                lang = Language(language.lower())
                return RecursiveCharacterTextSplitter.from_language(
                    language=lang,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
            except ValueError:
                pass
        return PythonCodeTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_document(
    content: str,
    metadata: dict[str, Any] | None = None,
    config: ChunkConfig | None = None,
    filename: str | None = None,
    language: str | None = None,
) -> list[Document]:
    config = config or ChunkConfig()
    metadata = metadata or {}

    strategy = config.strategy
    if strategy == ChunkingStrategy.AUTO:
        strategy = _detect_content_type(content, filename)

    splitter = _get_splitter(strategy, config, language)

    base_doc = Document(
        page_content=content,
        metadata={
            **metadata,
            "source": filename or "unknown",
            "chunking_strategy": strategy.value,
        },
    )

    chunks: list[Document] = splitter.split_documents([base_doc])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


def estimate_chunks(content: str, config: ChunkConfig | None = None) -> int:
    config = config or ChunkConfig()
    effective_size = config.chunk_size - config.chunk_overlap
    return max(1, len(content) // effective_size)
