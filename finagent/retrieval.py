"""Provider-neutral knowledge-store adapters.

The first production backend is still OpenAI Files + Vector Stores because
that is what existing deployments use. The important architectural change is
that uploads and hosted file-search tool creation now live behind this module,
so future pgvector/Qdrant/knowledge-MCP backends do not require API changes.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


DEFAULT_VECTOR_STORE_ID = os.environ.get(
    "OPENAI_VECTOR_STORE_ID", "vs_69a81b0197a481919e14c2d66197af7d"
)


@dataclass(frozen=True)
class UploadResult:
    filename: str
    file_id: str
    vector_store_id: str | None
    vector_store_file_id: str | None
    status: str
    bytes: int
    backend: str

    def as_response(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_id": self.file_id,
            "vector_store_id": self.vector_store_id,
            "vector_store_file_id": self.vector_store_file_id,
            "status": self.status,
            "bytes": self.bytes,
            "backend": self.backend,
        }


class KnowledgeStore(Protocol):
    backend: str

    async def upload_pdf(self, *, filename: str, data: bytes) -> UploadResult: ...

    def hosted_agent_tools(self) -> list[Any]: ...


class OpenAIVectorStoreKnowledgeStore:
    backend = "openai_vector_store"

    def __init__(self, vector_store_id: str = DEFAULT_VECTOR_STORE_ID) -> None:
        self.vector_store_id = vector_store_id

    async def upload_pdf(self, *, filename: str, data: bytes) -> UploadResult:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        uploaded = await client.files.create(
            file=(filename, data, "application/pdf"),
            purpose="assistants",
        )
        vsf = await client.vector_stores.files.create(
            vector_store_id=self.vector_store_id,
            file_id=uploaded.id,
        )
        return UploadResult(
            filename=filename,
            file_id=uploaded.id,
            vector_store_id=self.vector_store_id,
            vector_store_file_id=vsf.id,
            status=getattr(vsf, "status", "queued"),
            bytes=len(data),
            backend=self.backend,
        )

    def hosted_agent_tools(self) -> list[Any]:
        from agents import FileSearchTool

        return [FileSearchTool(vector_store_ids=[self.vector_store_id])]


class LocalPdfArchiveKnowledgeStore:
    """Minimal local backend for non-OpenAI development.

    It persists uploads and returns stable ids, but intentionally exposes no
    hosted agent file-search tool. A later pgvector/Qdrant implementation can
    add semantic search without changing the upload route contract.
    """

    backend = "local_pdf_archive"

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root or os.environ.get("KNOWLEDGE_STORE_PATH", "outputs/knowledge"))
        self.root.mkdir(parents=True, exist_ok=True)

    async def upload_pdf(self, *, filename: str, data: bytes) -> UploadResult:
        digest = hashlib.sha256(data).hexdigest()[:16]
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in filename)
        path = self.root / f"{digest}__{safe_name}"
        path.write_bytes(data)
        return UploadResult(
            filename=filename,
            file_id=f"local_{digest}",
            vector_store_id=None,
            vector_store_file_id=None,
            status="stored",
            bytes=len(data),
            backend=self.backend,
        )

    def hosted_agent_tools(self) -> list[Any]:
        return []


def get_knowledge_store() -> KnowledgeStore:
    backend = os.environ.get("KNOWLEDGE_STORE_BACKEND", "openai").strip().lower()
    if backend in {"openai", "openai_vector_store", "vector_store"}:
        return OpenAIVectorStoreKnowledgeStore()
    if backend in {"local", "local_pdf_archive"}:
        return LocalPdfArchiveKnowledgeStore()
    raise ValueError(
        f"Unsupported KNOWLEDGE_STORE_BACKEND={backend!r}. "
        "Supported values: openai, local."
    )


def hosted_file_search_tools() -> list[Any]:
    """Tools consumable by legacy OpenAI Agents SDK agents."""
    return get_knowledge_store().hosted_agent_tools()
