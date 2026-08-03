"""Download the BBC News dataset (bbc-text.csv) into the ``data/`` folder.

The dataset is sourced from the well-known ``bbc-text.csv`` file mirrored at:

    https://raw.githubusercontent.com/susanli2016/NLP-with-Python/master/data/bbc-text.csv

It is only requested over the network to avoid committing a large file; the
application can also run against the small committed sample at
``data/bbc-text-sample.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

DATASET_URL = (
    "https://raw.githubusercontent.com/susanli2016/NLP-with-Python/"
    "master/data/bbc-text.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "bbc-text.csv"


def download_dataset(url: str = DATASET_URL, output: Path = DEFAULT_OUTPUT) -> Path:
    """Download the dataset to disk.

    Args:
        url: Full URL of the CSV to download.
        output: Destination path.

    Returns:
        The path the dataset was written to.

    Raises:
        requests.HTTPError: If the download fails.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=30) as response:
        response.raise_for_status()
        content = response.content
    output.write_bytes(content)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the BBC News dataset.")
    parser.add_argument(
        "--url", default=DATASET_URL, help="Alternative dataset CSV URL."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination file path.",
    )
    args = parser.parse_args()

    target = download_dataset(args.url, args.output)
    print(f"Saved dataset to {target}")