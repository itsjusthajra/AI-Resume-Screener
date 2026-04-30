import re
import string
import logging

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Download required NLTK data on first import
def _ensure_nltk_data():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Words that carry signal in resume contexts — don't remove these
_resume_stopword_exceptions = {
    "python", "java", "sql", "r", "c", "excel", "aws", "gcp", "ml", "ai",
    "not", "no", "nor", "but", "however"
}

_CLEAN_PATTERNS = [
    (re.compile(r"http\S+|www\.\S+"), " "),
    (re.compile(r"\S+@\S+"), " "),
    (re.compile(r"\b\d{10,}\b"), " "),
    (re.compile(r"[^\w\s]"), " "),
    (re.compile(r"\d+"), " "),
    (re.compile(r"\s{2,}"), " "),
]


def clean_text(text: str) -> str:
    """Full NLP cleaning pipeline for resume text."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()

    for pattern, replacement in _CLEAN_PATTERNS:
        text = pattern.sub(replacement, text)

    tokens = word_tokenize(text)

    processed = []
    for token in tokens:
        if len(token) < 2:
            continue
        if token in _resume_stopword_exceptions:
            processed.append(token)
            continue
        if token in _stop_words:
            continue
        lemma = _lemmatizer.lemmatize(token)
        processed.append(lemma)

    return " ".join(processed).strip()


def extract_skills_from_text(text: str) -> list[str]:
    """Pull out recognizable technical and soft skills from raw resume text."""
    skill_keywords = {
        # Technical
        "python", "java", "javascript", "typescript", "sql", "r", "scala",
        "c++", "c#", "golang", "ruby", "php", "swift", "kotlin", "rust",
        "html", "css", "react", "angular", "vue", "django", "flask", "fastapi",
        "spring", "node", "express", "tensorflow", "pytorch", "keras", "sklearn",
        "pandas", "numpy", "spark", "hadoop", "kafka", "airflow", "dbt",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "git",
        "linux", "bash", "powerbi", "tableau", "excel", "sap", "salesforce",
        "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
        # Domain
        "machine learning", "deep learning", "nlp", "computer vision",
        "data engineering", "data analysis", "data science", "etl",
        "agile", "scrum", "devops", "ci/cd", "rest api", "microservices",
        # Finance / HR
        "accounting", "auditing", "tally", "quickbooks", "ifrs", "gaap",
        "payroll", "recruitment", "talent acquisition", "hris", "workday",
        "performance management", "training", "compliance",
    }

    text_lower = text.lower()
    found = []
    for skill in sorted(skill_keywords):
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.append(skill)
    return found


def get_top_keywords(text: str, top_n: int = 15) -> list[tuple[str, int]]:
    """Return the most frequent meaningful words in cleaned text."""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    freq: dict[str, int] = {}
    for t in tokens:
        if len(t) > 3:
            freq[t] = freq.get(t, 0) + 1
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
