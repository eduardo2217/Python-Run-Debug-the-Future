"""BBC News dataset adapter (Repository pattern).

This module isolates dataset loading behind a small, stable interface so the
API and CLI layers never depend on the concrete source of the data (CSV file,
database, live URL, ...). The BBC News dataset used is the well-known
``bbc-text.csv`` published by Susan Li and widely mirrored on GitHub:

    https://github.com/susanli2016/NLP-with-Python/blob/master/data/bbc-text.csv
    (original source: D. Greene, P. Cunningham, "Practical Solutions to the
     Problem of Diagonal Dominance in Kernel Document Clustering", ICML 2006)

It contains BBC news articles — a ``category`` (business, entertainment,
politics, sport, tech) and a ``text`` column with the full article body,
which is a good fit for text summarization.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class NewsArticle:
    """A single news article loaded from the dataset."""

    title: str
    text: str
    category: str


class DatasetRepository:
    """Loads :class:`NewsArticle` records from a CSV file."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def load(self, limit: int | None = None) -> list[NewsArticle]:
        """Return an optional number of articles from the dataset.

        Args:
            limit: Maximum number of articles to return.

        Raises:
            FileNotFoundError: If the CSV does not exist.
            ValueError: If the CSV is missing the expected columns.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.csv_path}. "
                "Run `python scripts/download_dataset.py` first."
            )

        articles: list[NewsArticle] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "text" not in reader.fieldnames:
                raise ValueError(
                    "Dataset is missing the required 'text' column. "
                    f"Found columns: {reader.fieldnames}"
                )
            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                articles.append(
                    NewsArticle(
                        title=(row.get("title") or text[:80]).strip(),
                        text=text,
                        category=(row.get("category") or "unknown").strip(),
                    )
                )
                if limit is not None and len(articles) >= limit:
                    break
        return articles

    def preview(self, count: int = 3) -> Iterable[NewsArticle]:
        """Yield a few sample articles without loading the whole file."""
        return self.load(limit=count)