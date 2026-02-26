"""
PDF utility functions for the SteeringLLM RAG demo tab.

Handles PDF text extraction, text chunking, and building contrast pairs
that allow the steering vector to ground the model on document content.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# PDF text extraction (PyPDF2 with pdfplumber fallback)
# ---------------------------------------------------------------------------

def _extract_with_pypdf2(file_bytes: bytes) -> str:
    """Extract text using PyPDF2."""
    import io
    import PyPDF2  # type: ignore[import]

    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def _extract_with_pdfplumber(file_bytes: bytes) -> str:
    """Extract text using pdfplumber (better layout preservation)."""
    import io
    import pdfplumber  # type: ignore[import]

    pages: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n\n".join(pages)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from PDF bytes.

    Tries pdfplumber first (better quality), falls back to PyPDF2.

    Args:
        file_bytes: Raw PDF bytes (from st.file_uploader or open()).

    Returns:
        Extracted text as a single string.

    Raises:
        ImportError: If neither PyPDF2 nor pdfplumber is installed.
        ValueError: If text extraction fails or returns empty text.
    """
    text = ""
    last_error: Optional[Exception] = None

    # Try pdfplumber first
    try:
        text = _extract_with_pdfplumber(file_bytes)
    except ImportError:
        pass
    except Exception as exc:
        last_error = exc

    # Fall back to PyPDF2
    if not text.strip():
        try:
            text = _extract_with_pypdf2(file_bytes)
        except ImportError as imp:
            raise ImportError(
                "PDF extraction requires PyPDF2. Install it with:\n"
                "  pip install PyPDF2\n"
                "or for better results:\n"
                "  pip install pdfplumber"
            ) from imp
        except Exception as exc:
            last_error = exc

    if not text.strip():
        msg = "Could not extract any text from the PDF."
        if last_error:
            msg += f" Last error: {last_error}"
        raise ValueError(msg)

    return text


def extract_text_from_pdf_path(path: str | Path) -> str:
    """
    Extract text from a PDF file on disk.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return extract_text_from_pdf(path.read_bytes())


# ---------------------------------------------------------------------------
# Text cleaning & chunking
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Basic cleanup: collapse whitespace, strip headers/footers patterns."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that look like page numbers (e.g., "- 42 -" or just "42")
    text = re.sub(r"(?m)^\s*-?\s*\d+\s*-?\s*$", "", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 60,
    min_chunk_size: int = 80,
) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text: Input text to chunk.
        chunk_size: Approximate number of words per chunk.
        overlap: Number of words to overlap between consecutive chunks.
        min_chunk_size: Minimum words for a chunk to be kept.

    Returns:
        List of text chunks.
    """
    text = _clean_text(text)
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if len(chunk_words) >= min_chunk_size:
            chunks.append(" ".join(chunk_words))
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Contrast-pair builder for RAG steering
# ---------------------------------------------------------------------------

# Generic vague/ungrounded negative statements.
# These represent "not knowing about the document" -- useful for all topics.
_GENERIC_NEGATIVES = [
    "I don't have specific information about that topic.",
    "That's a general area. I can only provide basic background.",
    "Without access to the details, I can only speculate.",
    "This topic is complex and I lack the specifics to give a precise answer.",
    "I'm not sure about the exact details. My information is very general.",
    "There are many possibilities, but I cannot be specific here.",
    "I don't have context about the particular situation you are describing.",
    "My knowledge here is limited to very general concepts.",
    "I lack the granular details needed to give a concrete answer.",
    "Without knowing the specific content, I can only give a vague response.",
]


def build_rag_contrast_pairs(
    chunks: List[str],
    max_positive: int = 8,
    max_negative: int = 8,
) -> Tuple[List[str], List[str]]:
    """
    Build positive (grounded) and negative (vague) contrast pairs for RAG steering.

    Positive examples = actual document chunks (factual, specific, grounded).
    Negative examples = generic vague statements (no document knowledge).

    Args:
        chunks: Text chunks extracted from the PDF.
        max_positive: Maximum number of positive examples to use.
        max_negative: Maximum number of negative examples to use.

    Returns:
        Tuple of (positive_examples, negative_examples).

    Raises:
        ValueError: If no usable chunks are provided.
    """
    if not chunks:
        raise ValueError("No text chunks provided. Extract PDF text first.")

    # Select a diverse spread of chunks from the document
    if len(chunks) <= max_positive:
        positives = chunks[:]
    else:
        # Sample evenly across the document for broad coverage
        step = len(chunks) / max_positive
        positives = [chunks[int(i * step)] for i in range(max_positive)]

    # Truncate very long positive examples to ~200 words
    positives = [" ".join(p.split()[:200]) for p in positives]

    negatives = _GENERIC_NEGATIVES[:max_negative]

    return positives, negatives


# ---------------------------------------------------------------------------
# Quick summarizer for display
# ---------------------------------------------------------------------------

def summarize_chunks(chunks: List[str], max_display: int = 3) -> List[str]:
    """
    Return a display-ready subset of chunks for the UI source expander.

    Args:
        chunks: List of all chunks.
        max_display: How many to show.

    Returns:
        Truncated list of chunks with display-friendly lengths.
    """
    selected = chunks[:max_display]
    return [textwrap.shorten(c, width=300, placeholder="...") for c in selected]
