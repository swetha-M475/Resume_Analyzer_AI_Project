"""
Embeddings Module
Handles HuggingFace embeddings and ChromaDB vector store operations.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "resume_collection"


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and return the HuggingFace embeddings model.
    Uses all-MiniLM-L6-v2 which is lightweight and runs locally (no API needed).
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return embeddings


def create_vector_store(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Create a ChromaDB vector store from document chunks.
    
    Args:
        chunks: List of Document objects from text splitting
        embeddings: HuggingFace embeddings model
    
    Returns:
        Chroma vector store instance
    """
    # Clear existing collection to avoid duplicates on re-upload
    try:
        existing_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        existing_store.delete_collection()
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    """
    Create a retriever from the vector store for RAG queries.
    
    Args:
        vector_store: Chroma vector store instance
        k: Number of relevant chunks to retrieve
    
    Returns:
        Retriever instance
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever


def load_existing_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.
    
    Args:
        embeddings: HuggingFace embeddings model
    
    Returns:
        Chroma vector store instance or None if not found
    """
    try:
        vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        # Check if the collection has any documents
        if vector_store._collection.count() > 0:
            return vector_store
        return None
    except Exception:
        return None
