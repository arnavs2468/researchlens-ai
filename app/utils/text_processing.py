"""Text cleaning, section detection, and hashing utilities."""

import hashlib
import re
from typing import Optional

# Common research paper section header patterns
COMMON_SECTION_PATTERNS = [
    (r"^(?:(?:\d+\.?)?\s*)?abstract\b", "Abstract"),
    (r"^(?:(?:\d+\.?)?\s*)?introduction\b", "Introduction"),
    (r"^(?:(?:\d+\.?)?\s*)?(?:background|related\s+work)\b", "Related Work"),
    (r"^(?:(?:\d+\.?)?\s*)?(?:methodology|methods|method|proposed\s+approach|architecture)\b", "Methodology"),
    (r"^(?:(?:\d+\.?)?\s*)?(?:experiments|experimental\s+setup|evaluation|implementation)\b", "Experiments"),
    (r"^(?:(?:\d+\.?)?\s*)?(?:results|findings|empirical\s+results)\b", "Results"),
    (r"^(?:(?:\d+\.?)?\s*)?(?:discussion|analysis|ablation\s+study)\b", "Discussion"),
    (r"^(?:(?:\d+\.?)?\s*)?(?:limitations|ethical\s+considerations)\b", "Limitations"),
    (r"^(?:(?:\d+\.?)?\s*)?(?:conclusion|conclusions|concluding\s+remarks)\b", "Conclusion"),
    (r"^(?:(?:\d+\.?)?\s*)?(?:future\s+work|future\s+directions)\b", "Future Work"),
    (r"^(?:(?:\d+\.?)?\s*)?references\b", "References"),
]


def clean_extracted_text(text: str) -> str:
    """Clean extracted PDF text: fix hyphenations, normalize unicode, strip noise."""
    if not text:
        return ""

    # Replace null bytes
    text = text.replace("\x00", "")

    # De-hyphenate words broken across lines (e.g., "algo-\nrithm" -> "algorithm")
    text = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)

    # Normalize newlines
    text = re.sub(r"\r\n|\r", "\n", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize whitespace inside paragraphs while retaining newlines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines).strip()

    return cleaned


def detect_section_header(line: str) -> Optional[str]:
    """
    Best-effort heuristic detection of academic paper section headers.
    Returns standardized section name if line matches a known heading pattern.
    """
    stripped = line.strip().lower()
    if not stripped or len(stripped) > 70:
        return None

    # Ignore lines that end in periods (likely sentences, not headers)
    if stripped.endswith("."):
        return None

    for pattern, section_name in COMMON_SECTION_PATTERNS:
        if re.search(pattern, stripped):
            return section_name

    return None


def compute_file_hash(content: bytes) -> str:
    """Compute SHA-256 hash of raw file bytes for deduplication."""
    return hashlib.sha256(content).hexdigest()
