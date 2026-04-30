
LABELS = {
    # Binary label pairs; first entry treated as "positive" for score sign
    "toxicity": ["toxic", "non-toxic"],
    "hate": ["hateful", "not hateful"],
    "offense": ["offensive", "not offensive"],
}

__all__ = ["LABELS"]
