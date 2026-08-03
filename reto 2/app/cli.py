"""Command-line interface to summarise dataset articles without the API.

Demonstrates the same cached/retried provider path as the web endpoint, but
from the terminal. Reads the OpenAI-compatible provider from settings and
prints a summary per article.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import get_settings
from app.core.cache import TTLCache
from app.core.exceptions import SummarizationError
from app.dataset import DatasetRepository
from app.services.provider_factory import build_provider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bbq-summarize",
        description="Summarise BBC News articles from a local CSV via the API.",
    )
    parser.add_argument(
        "--csv",
        default="data/bbc-text-sample.csv",
        help="Path to the dataset CSV (default: data/bbc-text-sample.csv).",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=3,
        help="Number of articles to summarise (default: 3).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    cache = TTLCache(ttl_seconds=settings.cache_ttl_seconds)

    try:
        repo = DatasetRepository(args.csv)
        articles = repo.load(limit=args.limit)
        if not articles:
            print("No articles found in the dataset.", file=sys.stderr)
            return 1

        provider = build_provider(settings)
        for article in articles:
            cached = cache.get(article.text)
            if cached is not None:
                summary, used_cache = cached, True
            else:
                summary = provider.summarize(article.text)
                cache.set(article.text, summary)
                used_cache = False

            print("\n" + "=" * 60)
            print(f"[{article.category}] {article.title}")
            print(f"(cache hit: {used_cache})")
            print(summary)
        return 0
    except SummarizationError as exc:
        print(f"Error: {exc.message}", file=sys.stderr)
        return exc.exit_code
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())