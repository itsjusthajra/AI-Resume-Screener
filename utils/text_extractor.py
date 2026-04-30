import io
import logging
from pathlib import Path

import pandas as pd
import re

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Normalize text for TF-IDF.
    Keeps letters, digits, and meaningful punctuation.
    Does NOT strip hyphens, slashes, underscores — critical for tech terms
    like 'machine-learning', 'c++', 'node.js', 'full-stack'.
    """
    text = text.lower()
    # Normalize unicode dashes to ASCII hyphen
    text = re.sub(r'[\u2013\u2014\u2212]', '-', text)
    # Collapse newlines and tabs into spaces
    text = re.sub(r'[\n\r\t]+', ' ', text)
    # Keep letters, digits, and: + # . / - _ ( ) ,
    text = re.sub(r'[^a-zA-Z0-9+#./\-_(),\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _try_pymupdf(raw_bytes: bytes) -> str:
    """Primary PDF extractor using PyMuPDF (fitz)."""
    try:
        import fitz
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        return "\n".join(pages)
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {e}")
        return ""


def _try_pdfplumber(raw_bytes: bytes) -> str:
    """
    Fallback PDF extractor using pdfplumber.
    Uses extract_text() — NOT extract_words() which merges words without spaces.
    """
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                # extract_text() preserves spacing correctly
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if text:
                    pages.append(text)
            return "\n".join(pages)
    except Exception as e:
        logger.error(f"pdfplumber extraction failed: {e}")
        return ""


def extract_from_pdf(file_obj) -> str:
    """
    Extract and clean text from a PDF.
    Tries PyMuPDF first falls back to pdfplumber if result is too short.
    """
    if hasattr(file_obj, 'read'):
        raw_bytes = file_obj.read()
    else:
        raw_bytes = open(file_obj, 'rb').read()

    text = _try_pymupdf(raw_bytes)

    if len(text.strip()) < 100:
        logger.warning("PyMuPDF returned short text — trying pdfplumber fallback")
        text = _try_pdfplumber(raw_bytes)

    if len(text.strip()) < 50:
        logger.warning("Both extractors returned near-empty text. PDF may be image-based (needs OCR).")

    return clean_text(text)


def extract_from_docx(file_obj) -> str:
    """Extract plain text from a DOCX file object."""
    try:
        from docx import Document
        doc = Document(file_obj)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return clean_text("\n".join(paragraphs))
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return ""


def extract_from_txt(file_obj) -> str:
    """Read and clean plain text from a TXT file object."""
    try:
        if hasattr(file_obj, "read"):
            raw = file_obj.read()
            text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        else:
            text = Path(file_obj).read_text(encoding="utf-8", errors="replace")
        return clean_text(text)
    except Exception as e:
        logger.error(f"TXT extraction failed: {e}")
        return ""


def extract_from_csv(file_obj) -> pd.DataFrame:
    """Load a resume CSV into a DataFrame. Returns empty DF on failure."""
    try:
        df = pd.read_csv(file_obj)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as e:
        logger.error(f"CSV load failed: {e}")
        return pd.DataFrame()


def extract_text_from_upload(uploaded_file) -> str:
    """
    Dispatch to the right extractor based on file extension.
    Works with Streamlit UploadedFile objects.
    """
    name = uploaded_file.name.lower()
    file_bytes = io.BytesIO(uploaded_file.read())

    if name.endswith(".pdf"):
        return extract_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        return extract_from_docx(file_bytes)
    elif name.endswith(".txt"):
        return extract_from_txt(file_bytes)
    elif name.endswith(".csv"):
        df = extract_from_csv(file_bytes)
        skip_cols = {"id", "resume_html", "resume_html_"}
        priority = ["resume_str", "resume", "text", "description"]
        text_col = next((c for c in priority if c in df.columns), None)
        if text_col:
            return clean_text(" ".join(df[text_col].fillna("").astype(str).tolist()))
        text_cols = [c for c in df.columns if df[c].dtype == object and c not in skip_cols]
        if text_cols:
            return clean_text(" ".join(df[text_cols[0]].fillna("").astype(str).tolist()))
        return ""
    else:
        logger.warning(f"Unsupported file type: {name}")
        return ""


def load_pdf_dataset(data_dir: str) -> pd.DataFrame:
    """
    Walk category subdirectories under data_dir, extract PDF text,
    and return a labeled DataFrame with columns [text, category].
    """
    data_dir = Path(data_dir)
    records = []

    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        label = category_dir.name
        pdf_files = list(category_dir.glob("*.pdf"))

        if not pdf_files:
            logger.info(f"No PDFs in {label}/")
            continue

        for pdf_path in pdf_files:
            text = extract_from_pdf(str(pdf_path))
            if text.strip():
                records.append({"text": text, "category": label})

    if not records:
        return pd.DataFrame(columns=["text", "category"])

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} resumes across {df['category'].nunique()} categories")
    return df


def load_csv_dataset(csv_path: str, text_col: str = None, label_col: str = None) -> pd.DataFrame:
    """
    Load a structured resume CSV. Handles the Kaggle resume dataset format
    (ID, Resume_str, Resume_html, Category) as well as generic formats.
    Auto-detects text and label columns if not specified.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"CSV read failed: {e}")
        return pd.DataFrame(columns=["text", "category"])

    col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df_norm = df.rename(columns=col_map)
    norm_cols = list(df_norm.columns)

    TEXT_CANDIDATES = ["resume_str", "resume", "resume_text", "text", "description", "content", "skills"]
    LABEL_CANDIDATES = ["category", "label", "job_role", "role", "job_title", "position"]

    if text_col is None:
        for c in TEXT_CANDIDATES:
            if c in norm_cols:
                text_col = c
                break
        if text_col is None:
            skip = {"id", "resume_html", "resume_html_"}
            object_cols = [c for c in norm_cols if df_norm[c].dtype == object and c not in skip]
            text_col = object_cols[0] if object_cols else None

    if label_col is None:
        for c in LABEL_CANDIDATES:
            if c in norm_cols:
                label_col = c
                break

    if text_col is None or label_col is None:
        logger.error(f"Could not detect text/label columns. Columns found: {norm_cols}")
        return pd.DataFrame(columns=["text", "category"])

    logger.info(f"CSV loader using text='{text_col}', label='{label_col}'")

    out = pd.DataFrame({
        "text": df_norm[text_col].fillna("").astype(str),
        "category": df_norm[label_col].fillna("Unknown").astype(str),
    })

    out = out[out["text"].str.strip().str.len() > 20]
    return out.reset_index(drop=True)
