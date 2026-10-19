from __future__ import annotations

"""Small text helpers used by chunking, retrieval, and answer cleanup."""

import re
from collections import Counter
from math import sqrt

STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "car",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "model",
    "of",
    "on",
    "or",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "vehicle",
    "what",
    "which",
    "with",
}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str, keep_stop_words: bool = False) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    if keep_stop_words:
        return words
    return [word for word in words if word not in STOP_WORDS and len(word) > 1]


def sentence_split(text: str) -> list[str]:
    clean = normalize_space(text)
    if not clean:
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?<=:)\s+(?=[A-Z0-9])", clean)
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= 280:
            sentences.append(part)
            continue
        words = part.split()
        for window in word_windows(words, size=34, overlap=0):
            sentences.append(" ".join(window))
    return sentences


def word_windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if not words:
        return []
    if len(words) <= size:
        return [words]

    windows: list[list[str]] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(words):
        window = words[start : start + size]
        if window:
            windows.append(window)
        if start + size >= len(words):
            break
        start += step
    return windows


def term_counter(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0

    common = set(left).intersection(right)
    numerator = sum(left[term] * right[term] for term in common)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def overlap_ratio(question: str, text: str) -> float:
    q_terms = set(tokenize(question))
    if not q_terms:
        return 0.0
    text_terms = set(tokenize(text))
    return len(q_terms.intersection(text_terms)) / len(q_terms)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"
