import os
import tempfile
from dataclasses import dataclass

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import NodeWithScore


@dataclass
class Finding:
    """Represents a retrieved context snippet with its file source."""

    text: str
    source: str


class Retriever:
    """A wrapper around LlamaIndex's VectorStoreIndex to maintain state."""

    def __init__(self, index: VectorStoreIndex):
        self.index = index


def _process_files_to_llama_docs(files: list) -> list[Document]:
    """Helper to convert various file objects into LlamaIndex Documents."""
    from llama_index.core import SimpleDirectoryReader

    llama_docs = []

    for file_obj in files:
        # Handle file paths (strings)
        if isinstance(file_obj, str) and os.path.exists(file_obj):
            reader = SimpleDirectoryReader(input_files=[file_obj])
            llama_docs.extend(reader.load_data())

        # Handle uploaded file-like objects (e.g., Streamlit, FastAPI, or bytes)
        elif hasattr(file_obj, "read"):
            original_name = getattr(file_obj, "name", "uploaded_file.txt")
            suffix = os.path.splitext(original_name)[1]

            # Write to a temporary file so LlamaIndex readers can parse it natively
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = file_obj.read()
                if isinstance(content, str):
                    content = content.encode("utf-8")
                temp_file.write(content)
                temp_path = temp_file.name

            try:
                reader = SimpleDirectoryReader(input_files=[temp_path])
                loaded = reader.load_data()
                # Restore the original filename in the metadata
                for doc in loaded:
                    doc.metadata["file_name"] = original_name
                llama_docs.extend(loaded)
            finally:
                os.unlink(temp_path)  # Clean up temp file

    return llama_docs


def build_retriever(files: list) -> Retriever:
    """Chunk + embed uploaded files, return a queryable retriever object."""
    llama_docs = _process_files_to_llama_docs(files)

    # Creates an in-memory vector store, chunks text, and generates embeddings
    index = VectorStoreIndex.from_documents(llama_docs)
    return Retriever(index=index)


def add_documents(retriever: Retriever, files: list) -> Retriever:
    """Extend an existing retriever with more uploaded files."""
    llama_docs = _process_files_to_llama_docs(files)

    # Insert new documents dynamically into the existing index
    for doc in llama_docs:
        retriever.index.insert(doc)

    return retriever


def retrieve(retriever: Retriever, task: str, k: int = 5) -> list[Finding]:
    """Query for a given research task; returns Finding objects with source set."""
    # Convert index into a low-level retriever object
    llama_retriever = retriever.index.as_retriever(similarity_top_k=k)

    # Fetch top nodes
    nodes: list[NodeWithScore] = llama_retriever.retrieve(task)

    findings = []
    for node in nodes:
        # Extract filename from metadata map safely
        source_name = node.node.metadata.get("file_name", "Unknown Source")

        findings.append(Finding(text=node.node.get_content(), source=source_name))

    return findings
