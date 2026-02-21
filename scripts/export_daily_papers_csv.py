"""Export daily paper reports to a CSV file.

Reads markdown files in a daily report directory and writes a CSV with
selected fields for all papers.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Dict, List

from collect_llm_judge_arxiv import parse_daily_papers


DEFAULT_DAILY_DIR = "reports/daily"
DEFAULT_OUTPUT = "reports/daily_papers.csv"
FIELDNAMES = [
    "Title",
    "Subcategory",
    "Link",
    "Published",
    "Authors",
    "Summary",
]


def build_summary(paper: Dict[str, str]) -> str:
    """Build a summary string from available fields.

    Args:
        paper: Parsed paper dict.

    Returns:
        Summary string for CSV.
    """
    parts: List[str] = []
    purpose = paper.get("purpose", "").strip()
    method = paper.get("method", "").strip()
    results = paper.get("results", "").strip()
    if purpose:
        parts.append(f"Purpose: {purpose}")
    if method:
        parts.append(f"Method: {method}")
    if results:
        parts.append(f"Results: {results}")
    if parts:
        return " ".join(parts)
    abstract = paper.get("abstract", "").strip()
    if abstract:
        return f"Abstract: {abstract}"
    return ""


def collect_papers(daily_dir: pathlib.Path) -> List[Dict[str, str]]:
    """Collect papers from daily markdown reports.

    Args:
        daily_dir: Directory containing daily reports.

    Returns:
        List of parsed paper dicts.
    """
    papers: List[Dict[str, str]] = []
    if not daily_dir.exists():
        return papers
    for entry in sorted(daily_dir.glob("*.md")):
        if entry.name == "index.md":
            continue
        content = entry.read_text(encoding="utf-8")
        papers.extend(parse_daily_papers(content))
    return papers


def write_csv(papers: List[Dict[str, str]], output_path: pathlib.Path) -> None:
    """Write papers to CSV.

    Args:
        papers: Parsed papers.
        output_path: CSV output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Use UTF-8 BOM so Excel on Windows opens the CSV cleanly.
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for paper in papers:
            writer.writerow(
                {
                    "Title": paper.get("title", ""),
                    "Subcategory": paper.get("tags", ""),
                    "Link": paper.get("link", ""),
                    "Published": paper.get("published", ""),
                    "Authors": paper.get("authors", ""),
                    "Summary": build_summary(paper),
                }
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed args.
    """
    parser = argparse.ArgumentParser(
        description="Export daily report papers to CSV."
    )
    parser.add_argument(
        "--daily-dir",
        default=DEFAULT_DAILY_DIR,
        help="Directory with daily markdown reports.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CSV export."""
    args = parse_args()
    daily_dir = pathlib.Path(args.daily_dir)
    output_path = pathlib.Path(args.output)
    papers = collect_papers(daily_dir)
    write_csv(papers, output_path)


if __name__ == "__main__":
    main()
