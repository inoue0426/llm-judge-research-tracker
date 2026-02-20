"""Run the LLM-as-a-judge collector at most once per local day."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from typing import List

from diskcache import Cache

CACHE_DIR = ".cache/llm_judge_arxiv"
DEFAULT_DAYS_BACK = 1
DEFAULT_MAX_RESULTS = 200
DEFAULT_OUTPUT_PATH = "reports/llm-as-a-judge.md"
DEFAULT_TAGS_JSON_PATH = "reports/llm-as-a-judge_tags.json"
DEFAULT_TAGS_CSV_PATH = "reports/llm-as-a-judge_tags.csv"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.3"
DEFAULT_RUN_AFTER_HOUR = 8
DEFAULT_BACKFILL_DAYS = 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-judge collector at most once per local day."
    )
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK)
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--tags-json", type=str, default=DEFAULT_TAGS_JSON_PATH)
    parser.add_argument("--tags-csv", type=str, default=DEFAULT_TAGS_CSV_PATH)
    parser.add_argument("--ollama-endpoint", type=str, default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument(
        "--run-after-hour",
        type=int,
        default=DEFAULT_RUN_AFTER_HOUR,
        help="Only run after this local hour (0-23).",
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=DEFAULT_BACKFILL_DAYS,
        help="Generate daily reports for the last N days.",
    )
    parser.add_argument("--force", action="store_true", help="Run even if already ran today.")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> List[str]:
    """Build the collector command.

    Args:
        args: Parsed arguments.

    Returns:
        Command list.
    """
    return [
        "uv",
        "run",
        "python",
        "scripts/collect_llm_judge_arxiv.py",
        "--days-back",
        str(args.days_back),
        "--max-results",
        str(args.max_results),
        "--output",
        args.output,
        "--tags-json",
        args.tags_json,
        "--tags-csv",
        args.tags_csv,
        "--ollama-endpoint",
        args.ollama_endpoint,
        "--ollama-model",
        args.ollama_model,
        "--require-llm",
        "--skip-failed",
        "--backfill-days",
        str(args.backfill_days),
    ]


def should_run_today(cache: Cache, today: dt.date) -> bool:
    """Check whether the collector should run today.

    Args:
        cache: DiskCache cache.
        today: Local date.

    Returns:
        True if the job should run.
    """
    last_run = cache.get("last_success_date_local")
    return last_run != today.isoformat()


def record_success(cache: Cache, now: dt.datetime) -> None:
    """Record a successful run.

    Args:
        cache: DiskCache cache.
        now: Current datetime.
    """
    cache.set("last_success_date_local", now.date().isoformat())
    cache.set("last_success_timestamp_local", now.isoformat())


def main() -> None:
    """Run the daily collector if needed."""
    args = parse_args()
    now = dt.datetime.now()
    if not args.force and now.hour < args.run_after_hour:
        return
    with Cache(CACHE_DIR) as cache:
        if not args.force and not should_run_today(cache, now.date()):
            return
        command = build_command(args)
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        record_success(cache, now)


if __name__ == "__main__":
    main()
