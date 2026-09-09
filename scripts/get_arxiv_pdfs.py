"""Download PDFs from a JSON file containing a list of arXiv PDF URLs."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

try:
    import certifi

    _ssl_context = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _ssl_context = ssl.create_default_context()


def load_pdf_urls(json_path: str | os.PathLike[str]) -> Dict[str, str]:
    """Load a JSON list of arXiv PDF URLs.

    Each item in the list should be a URL string.
    """
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of PDF URLs, got {type(payload).__name__}")

    cleaned: Dict[str, str] = {}
    for pdf_url in payload:
        if not isinstance(pdf_url, str):
            continue
        paper_id = pdf_url.rstrip("/").split("/")[-1]
        cleaned[paper_id] = pdf_url

    return cleaned


def download_pdf(pdf_url: str, destination: str | os.PathLike[str], retries: int = 3) -> bool:
    """Download a single PDF URL to the destination path with retries."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(pdf_url, context=_ssl_context, timeout=30) as response, open(destination, "wb") as output_file:
                shutil.copyfileobj(response, output_file)
            print(f"Downloaded: {destination.name}")
            return True
        except Exception as exc:
            if attempt == retries:
                print(f"Failed to download {pdf_url} to {destination}: {exc}")
                return False

            wait_time = (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"Download failed for {pdf_url} (attempt {attempt}/{retries}). Retrying in {wait_time:.2f}s...")
            time.sleep(wait_time)

    return False


def download_all_pdfs(
    pdf_urls: Dict[str, str],
    output_dir: str | os.PathLike[str],
    overwrite: bool = False,
    limit: int | None = None,
) -> Tuple[int, int]:
    """Download all PDFs from the URL mapping and return (downloaded, skipped)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    items = list(pdf_urls.items())

    if limit is not None:
        items = items[:limit]

    for paper_id, pdf_url in items:
        file_name = f"{paper_id}.pdf" if not paper_id.lower().endswith(".pdf") else paper_id
        file_path = output_dir / file_name

        if file_path.exists() and not overwrite:
            print(f"Skipping existing file: {file_path.name}")
            skipped += 1
            continue

        if download_pdf(pdf_url, file_path):
            downloaded += 1

    return downloaded, skipped


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_json = repo_root / "arxiv_pdfs" / "pdf_urls.json"
    default_output = repo_root / "arxiv_pdfs" / "pdfs"

    parser = argparse.ArgumentParser(description="Download PDFs from a JSON list of arXiv PDF URLs.")
    parser.add_argument("--input", type=str, default=str(default_json), help="Path to a JSON file containing a list of PDF URLs.")
    parser.add_argument("--output", type=str, default=str(default_output), help="Directory where PDFs will be saved.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of PDFs to download.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite PDFs even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"PDF URL file not found: {input_path}")

    pdf_urls = load_pdf_urls(input_path)
    downloaded, skipped = download_all_pdfs(pdf_urls, output_dir, overwrite=args.overwrite, limit=args.limit)

    print(f"\nCompleted: {downloaded} downloaded, {skipped} skipped, {len(pdf_urls)} total entries in {input_path}.")


if __name__ == "__main__":
    main()

