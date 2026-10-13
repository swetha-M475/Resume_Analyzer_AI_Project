"""
Resume Parser Module
Handles multi-format resume text extraction (PDF, DOCX, TXT) and text chunking for vector storage.
"""

import re
from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─── Supported File Extensions ────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = ["pdf", "docx", "txt"]


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


def extract_text_from_docx(uploaded_file) -> str:
    """
    Extract text content from an uploaded DOCX file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object (DOCX)
    
    Returns:
        Extracted text as a single string
    """
    try:
        from docx import Document

        doc = Document(BytesIO(uploaded_file.read()))
        text_parts = []

        # Extract paragraph text
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        # Extract text from tables (resumes often use tables for layout)
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        row_text.append(cell_text)
                if row_text:
                    text_parts.append(" | ".join(row_text))

        full_text = "\n".join(text_parts)
        return full_text

    except ImportError:
        raise ValueError(
            "python-docx is required for DOCX support. "
            "Install it with: pip install python-docx"
        )
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")


def extract_text_from_txt(uploaded_file) -> str:
    """
    Extract text content from an uploaded TXT file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object (TXT)
    
    Returns:
        Extracted text as a single string
    """
    try:
        raw_bytes = uploaded_file.read()

        # Try UTF-8 first, fall back to latin-1 (which never fails)
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue

        return raw_bytes.decode("utf-8", errors="replace")

    except Exception as e:
        raise ValueError(f"Failed to read TXT file: {str(e)}")


def extract_text_from_file(uploaded_file) -> str:
    """
    Unified text extraction dispatcher — detects file type by extension
    and routes to the appropriate extractor.
    
    Supported formats: PDF, DOCX, TXT
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Extracted text as a single string
    
    Raises:
        ValueError: If file type is unsupported or extraction fails
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif filename.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"Unsupported file type: '{filename.split('.')[-1]}'. "
            f"Supported formats: {supported}"
        )


def clean_resume_text(text: str) -> str:
    """
    Clean extracted resume text by removing artifacts and normalizing whitespace.
    
    Args:
        text: Raw extracted text from a resume file
    
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
