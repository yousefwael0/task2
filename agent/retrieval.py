"""
Document ingestion and retrieval for the Researcher node, built on
LlamaIndex.

Design notes:
- Input is (filename, bytes) tuples — what backend/routes.py's /upload
  handler produces via `await file.read()` — not a Streamlit/FastAPI
  upload type. Keeps this module usable from any caller.
- Finding is imported from agent.state, not redefined here — one
  canonical shape (a TypedDict, so it's JSON-serializable for the SSE
  stream) instead of two definitions that can drift apart.
- Embeddings run locally via a HuggingFace sentence-transformers model.
  Left unconfigured, LlamaIndex defaults to OpenAI embeddings and
  crashes on a missing OPENAI_API_KEY — a second, unrelated key this
  project has no other use for, since the LLM side is all Groq.
"""

import os
import tempfile

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import Document, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from agent.state import Finding

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings_configured = False


def _ensure_embeddings_configured() -> None:
    """Point LlamaIndex's global Settings at a local embedding model,
    once per process — guarded so repeated calls don't reload it."""
    global _embeddings_configured
    if not _embeddings_configured:
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        _embeddings_configured = True


class Retriever:
    """Thin wrapper around a LlamaIndex VectorStoreIndex, so callers
    (session_store, nodes.py) import one stable type regardless of
    which indexing library sits behind it."""

    def __init__(self, index: VectorStoreIndex):
        self.index = index


def _files_to_documents(files: list[tuple[str, bytes]]) -> list[Document]:
    """Turn (filename, bytes) pairs into LlamaIndex Documents.
    SimpleDirectoryReader needs a real file on disk to pick a parser by
    extension, so bytes are written to a temp file, read, then cleaned
    up — the returned Documents hold the extracted text, not a
    reference to the temp file."""
    documents: list[Document] = []

    for filename, content in files:
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            reader = SimpleDirectoryReader(input_files=[tmp_path])
            loaded = reader.load_data()
        finally:
            os.unlink(tmp_path)

        # Restore the real filename — SimpleDirectoryReader otherwise
        # tags metadata with the temp path's basename, and this is what
        # should show up as a Finding's source later.
        for doc in loaded:
            doc.metadata["file_name"] = filename
        documents.extend(loaded)

    return documents


def build_retriever(files: list[tuple[str, bytes]]) -> Retriever:
    """Chunk + embed uploaded files, return a queryable retriever object."""
    _ensure_embeddings_configured()
    documents = _files_to_documents(files)
    index = VectorStoreIndex.from_documents(documents)
    return Retriever(index=index)


def add_documents(retriever: Retriever, files: list[tuple[str, bytes]]) -> Retriever:
    """Extend an existing retriever with more uploaded files."""
    _ensure_embeddings_configured()
    documents = _files_to_documents(files)
    for doc in documents:
        retriever.index.insert(doc)
    return retriever


def retrieve(retriever: Retriever | None, task: str, k: int = 5) -> list[Finding]:
    """Query for a given research task; returns Finding objects with
    task and source set. Returns an empty list, not an error, if
    retriever is None — the Researcher node falls back to model
    knowledge when no documents were uploaded."""
    if retriever is None:
        return []

    llama_retriever = retriever.index.as_retriever(similarity_top_k=k)
    nodes: list[NodeWithScore] = llama_retriever.retrieve(task)

    findings: list[Finding] = []
    for node in nodes:
        source_name = node.node.metadata.get("file_name", "unknown")
        findings.append(
            Finding(task=task, content=node.node.get_content(), source=source_name)
        )
    return findings
