"""
Resume Parser Module
Handles PDF text extraction and text chunking for vector storage.
"""

import re
from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text content from an uploaded PDF file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object (PDF)
    
    Returns:
        Extracted text as a single string
    """
    try:
        pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
        text_parts = []

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n".join(text_parts)
        return full_text

    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def clean_resume_text(text: str) -> str:
    """
    Clean extracted resume text by removing artifacts and normalizing whitespace.
    
    Args:
        text: Raw extracted text from PDF
    
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special unicode characters that PDF extraction sometimes produces
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    # Normalize bullet points
    text = re.sub(r'[•●■◆▪]', '- ', text)

    # Fix common PDF extraction issues
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenated words split across lines

    # Normalize line breaks for readability
    text = re.sub(r'\s*\n\s*', '\n', text)

    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100, source: str = None) -> list:
    """
    Split text into overlapping chunks for vector storage.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
        source: Optional source file name to store in metadata
    
    Returns:
        List of Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )

    metadatas = [{"source": source}] if source else None
    chunks = text_splitter.create_documents([text], metadatas=metadatas)
    return chunks


def get_resume_sections(text: str) -> dict:
    """
    Attempt to identify common resume sections from the text.
    
    Args:
        text: Cleaned resume text
    
    Returns:
        Dictionary with identified sections
    """
    section_headers = [
        "objective", "summary", "experience", "work experience",
        "education", "skills", "technical skills", "projects",
        "certifications", "achievements", "awards", "languages",
        "interests", "hobbies", "references", "contact",
        "professional summary", "career objective"
    ]

    sections = {}
    text_lower = text.lower()

    for header in section_headers:
        pattern = re.compile(
            rf'(?:^|\n)\s*{re.escape(header)}[\s:]*\n(.*?)(?=\n\s*(?:{"|".join(section_headers)})\s*[\s:]|\Z)',
            re.IGNORECASE | re.DOTALL
        )
        match = pattern.search(text_lower)
        if match:
            sections[header] = match.group(1).strip()

    return sections
