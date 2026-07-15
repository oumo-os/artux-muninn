"""
text_utils.py — Text loading utilities for demos.

Standalone module with no heavy dependencies.
Can be imported independently of memory_module or llama-cpp.
"""

import re
from pathlib import Path


def load_paragraphs(filepath: str, min_length: int = 50) -> list[dict]:
    """
    Load a text file and split into paragraphs.

    Paragraphs are split on double newlines (\\n\\s*\\n).
    Very short paragraphs (< min_length chars) are skipped
    (these are usually section headers, page numbers, or blank lines).

    Parameters
    ----------
    filepath : str
        Path to the text file.
    min_length : int
        Minimum paragraph length in characters (default 50).

    Returns
    -------
    list of dict
        Each dict has keys: id (int), text (str), offset (int).
    """
    raw = Path(filepath).read_text(encoding="utf-8", errors="replace")

    # Strip Gutenberg header/footer if present
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker   = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = raw.find(start_marker)
    end   = raw.find(end_marker)
    if start != -1:
        raw = raw[start + len(start_marker):]
    if end != -1:
        raw = raw[:end]
    raw = raw.strip()

    # Split into paragraphs on double newlines
    raw_paragraphs = re.split(r'\n\s*\n', raw)

    paragraphs = []
    pid = 0
    offset = 0
    for para in raw_paragraphs:
        para = para.strip()
        if len(para) >= min_length:
            paragraphs.append({
                "id":     pid,
                "text":   para,
                "offset": offset,
            })
            pid += 1
        offset += len(para) + 2  # +2 for the \n\n

    return paragraphs


def detect_section(text: str) -> str:
    """
    Detect the book section from paragraph text.
    Returns a topic tag like 'book-1', 'book-2', 'introduction', etc.
    """
    lower = text.lower()

    # Book detection
    book_patterns = [
        (r'\bbook\s*i\b(?!\s*v)', "book-1"),
        (r'\bbook\s*ii\b',        "book-2"),
        (r'\bbook\s*iii\b',       "book-3"),
        (r'\bbook\s*iv\b',        "book-4"),
        (r'\bbook\s*v\b',         "book-5"),
        (r'\bbook\s*vi\b',        "book-6"),
        (r'\bbook\s*vii\b',       "book-7"),
        (r'\bbook\s*viii\b',      "book-8"),
        (r'\bbook\s*ix\b',        "book-9"),
        (r'\bbook\s*x\b',         "book-10"),
    ]

    # Check from most specific to least (Book VIII before Book IV, etc.)
    for pattern, tag in book_patterns:
        if re.search(pattern, lower):
            return tag

    # Introduction/analysis
    if "introduction" in lower or "analysis" in lower:
        return "introduction"

    return "general"


def detect_people(text: str) -> list[str]:
    """
    Detect mentions of known people from the Republic.
    Returns a list of person names found in the text.
    """
    known_people = {
        "Socrates":   ["socrates", "socratic"],
        "Glaucon":    ["glaucon"],
        "Adeimantus": ["adeimantus"],
        "Thrasymachus": ["thrasy machus", "thrasy-machus", "thrasymachus"],
        "Polemarchus": ["polemarchus"],
        "Cephalus":   ["cephalus"],
        "Plato":      ["plato"],
        "Niceratus":  ["niceratus"],
        "Cleitophon": ["cleitophon"],
        "Herodotus":  ["herodotus"],
        "Solon":      ["solon"],
        "Aristotle":  ["aristotle"],
        "Pythagoras": ["pythagoras"],
        "Homer":      ["homer"],
        "Bacon":      ["bacon"],
        "Cicero":     ["cicero"],
        "Augustine":  ["augustine"],
        "More":       ["thomas more", "sir thomas more"],
    }

    found = []
    lower = text.lower()
    for name, patterns in known_people.items():
        for pat in patterns:
            if pat in lower:
                found.append(name)
                break

    return found


def load_chunks(
    filepath: str,
    chunk_size: int = 5,
    min_length: int = 50,
) -> list[dict]:
    """
    Load a text file and group paragraphs into chunks for chat simulation.

    Each chunk becomes one simulated user message.
    Chunks are consecutive groups of *chunk_size* paragraphs, concatenated
    with double newlines.

    Parameters
    ----------
    filepath : str
        Path to the text file.
    chunk_size : int
        Number of paragraphs per chunk (default 5).
    min_length : int
        Minimum paragraph length in characters (passed to ``load_paragraphs``).

    Returns
    -------
    list of dict
        Each dict has keys:
          id        — chunk index (0-based)
          text      — concatenated paragraphs
          start_pid — first paragraph id
          end_pid   — last paragraph id
    """
    paras = load_paragraphs(filepath, min_length=min_length)
    chunks = []
    for i in range(0, len(paras), chunk_size):
        group = paras[i : i + chunk_size]
        text = "\n\n".join(p["text"] for p in group)
        chunks.append({
            "id": i // chunk_size,
            "text": text,
            "start_pid": group[0]["id"],
            "end_pid": group[-1]["id"],
        })
    return chunks
