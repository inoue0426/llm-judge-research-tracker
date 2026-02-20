"""Collect and summarize LLM-as-a-judge papers from arXiv."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import re
from typing import Dict
from typing import Iterable, List, Optional, Sequence

import arxiv
from diskcache import Cache
import requests

QUERY_TERMS: Sequence[str] = (
    "LLM as a judge",
    "LLM-as-a-judge",
    "large language model as a judge",
    "LLM judge",
    "LLM-based evaluation",
    "LLM evaluation",
)
DEFAULT_DAYS_BACK = 1
DEFAULT_MAX_RESULTS = 200
DEFAULT_OUTPUT_PATH = "reports/llm-as-a-judge.md"
DEFAULT_TAGS_JSON_PATH = "reports/llm-as-a-judge_tags.json"
DEFAULT_TAGS_CSV_PATH = "reports/llm-as-a-judge_tags.csv"
DEFAULT_DAILY_DIR = "reports/daily"
DEFAULT_DAILY_INDEX_PATH = "reports/daily/index.md"
DEFAULT_BACKFILL_DAYS = 0
DEFAULT_BACKFILL_SKIP_EXISTING = True
DEFAULT_README_PATH = "README.md"
TAG_STATS_START = "<!-- TAG_STATS_START -->"
TAG_STATS_END = "<!-- TAG_STATS_END -->"
CACHE_DIR = ".cache/llm_judge_arxiv"
OLLAMA_ENDPOINT = "http://localhost:11434"
OLLAMA_MODEL = "llama3.3"
OLLAMA_TIMEOUT_SECONDS = 60
OLLAMA_RETRY_DELAY_SECONDS = 2
OLLAMA_MAX_RETRIES = 1
TAG_MAX_PER_PAPER = 2
TAG_MIN_SCORE = 1

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

SECTION_KEYWORDS = {
    "purpose": ("we propose", "we present", "this paper", "we introduce", "goal", "aim"),
    "method": (
        "we use",
        "we develop",
        "we design",
        "method",
        "approach",
        "model",
        "framework",
        "pipeline",
    ),
    "results": (
        "we show",
        "we demonstrate",
        "results",
        "we achieve",
        "we outperform",
        "experiments",
    ),
}

SUBCLASS_TAGS: Sequence[Dict[str, object]] = (
    {
        "key": "judge_prompting_protocols",
        "label": "Judge Prompting Protocols",
        "keywords": (
            "rubric",
            "criteria",
            "pairwise",
            "chain-of-thought",
            "calibration prompt",
            "prompting protocol",
            "prompt template",
        ),
    },
    {
        "key": "robustness_and_sensitivity",
        "label": "Robustness And Sensitivity",
        "keywords": (
            "lexical",
            "syntactic",
            "paraphrase",
            "perturbation",
            "prompt sensitivity",
            "adversarial",
            "robustness",
        ),
    },
    {
        "key": "benchmarks_and_datasets",
        "label": "Benchmark And Dataset Creation",
        "keywords": (
            "benchmark",
            "dataset",
            "curated",
            "collection",
            "taxonomy",
            "evaluation suite",
        ),
    },
    {
        "key": "judge_reliability_and_calibration",
        "label": "Judge Reliability And Calibration",
        "keywords": (
            "inter-annotator",
            "agreement",
            "human",
            "bias",
            "calibration",
            "reliability",
            "consistency",
        ),
    },
    {
        "key": "domain_specific_judging",
        "label": "Domain-Specific Judging",
        "keywords": (
            "medical",
            "clinical",
            "legal",
            "education",
            "software",
            "history",
            "policy",
            "finance",
        ),
    },
    {
        "key": "multi_judge_ensembles",
        "label": "Multi-Judge Or Ensemble Methods",
        "keywords": (
            "ensemble",
            "multi-judge",
            "committee",
            "majority vote",
            "consensus",
            "self-consistency",
        ),
    },
    {
        "key": "metrics_and_scoring",
        "label": "Metrics And Scoring Methods",
        "keywords": (
            "metric",
            "score",
            "scoring",
            "evaluation metric",
            "calibration curve",
            "probabilistic",
        ),
    },
)


@dataclasses.dataclass(frozen=True)
class PaperSummary:
    """Summary of a single arXiv paper."""

    arxiv_id: str
    title: str
    authors: str
    published: dt.date
    updated: dt.date
    categories: str
    link: str
    abstract: str
    purpose: str
    method: str
    results: str
    tag_keys: Sequence[str]
    tags: Sequence[str]


def build_arxiv_query(terms: Sequence[str]) -> str:
    """Build an arXiv query string from query terms.

    Args:
        terms: Sequence of text queries.

    Returns:
        arXiv API query string.
    """
    fragments: List[str] = []
    for term in terms:
        quoted = f'"{term}"'
        fragments.append(f'ti:{quoted}')
        fragments.append(f'abs:{quoted}')
    return " OR ".join(fragments)


def split_sentences(text: str) -> List[str]:
    """Split a text block into sentences.

    Args:
        text: Input text.

    Returns:
        List of sentences.
    """
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []
    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(cleaned) if sentence.strip()]


def score_sentence(sentence: str, keywords: Iterable[str]) -> int:
    """Score a sentence by keyword matches.

    Args:
        sentence: Sentence text.
        keywords: Keywords to look for.

    Returns:
        Integer score.
    """
    lowered = sentence.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def pick_sentence(sentences: Sequence[str], keywords: Iterable[str]) -> str:
    """Pick the best sentence for a section based on keywords.

    Args:
        sentences: Candidate sentences.
        keywords: Section keywords.

    Returns:
        Selected sentence.
    """
    if not sentences:
        return "Not stated."
    scored = sorted(
        ((score_sentence(sentence, keywords), index, sentence) for index, sentence in enumerate(sentences)),
        key=lambda item: (-item[0], item[1]),
    )
    if scored[0][0] == 0:
        return sentences[0]
    return scored[0][2]


def summarize_abstract_sections(abstract: str) -> tuple[str, str, str]:
    """Generate sectioned summary from an abstract.

    Args:
        abstract: Paper abstract.

    Returns:
        Purpose, method, and results sentences.
    """
    sentences = split_sentences(abstract)
    purpose = pick_sentence(sentences, SECTION_KEYWORDS["purpose"])
    method = pick_sentence(sentences[1:] or sentences, SECTION_KEYWORDS["method"])
    results = pick_sentence(sentences[2:] or sentences, SECTION_KEYWORDS["results"])
    return purpose, method, results


def normalize_text(*parts: str) -> str:
    """Normalize text for keyword scanning.

    Args:
        parts: Text components to join.

    Returns:
        Lowercased, whitespace-normalized string.
    """
    joined = " ".join(part.strip() for part in parts if part)
    return " ".join(joined.lower().split())


def score_keywords(text: str, keywords: Iterable[str]) -> int:
    """Score text by keyword matches.

    Args:
        text: Normalized text to scan.
        keywords: Keyword phrases to match.

    Returns:
        Integer score.
    """
    return sum(1 for keyword in keywords if keyword in text)


def classify_tags(
    title: str, abstract: str, purpose: str, method: str, results: str
) -> tuple[List[str], List[str]]:
    """Assign subclass tags based on keyword matches.

    Args:
        title: Paper title.
        abstract: Paper abstract.
        purpose: Purpose sentence.
        method: Method sentence.
        results: Results sentence.

    Returns:
        Tuple of tag keys and tag labels.
    """
    text = normalize_text(title, abstract, purpose, method, results)
    scored: List[tuple[int, int, str, str]] = []
    for index, entry in enumerate(SUBCLASS_TAGS):
        keywords = entry["keywords"]
        if not isinstance(keywords, tuple):
            continue
        score = score_keywords(text, keywords)
        if score >= TAG_MIN_SCORE:
            scored.append((score, index, entry["key"], entry["label"]))
    if not scored:
        return (["unclassified"], ["Unclassified"])
    scored.sort(key=lambda item: (-item[0], item[1]))
    keys = [key for _, _, key, _ in scored[:TAG_MAX_PER_PAPER]]
    labels = [label for _, _, _, label in scored[:TAG_MAX_PER_PAPER]]
    return keys, labels


def build_ollama_prompt(title: str, abstract: str) -> str:
    """Build prompt for LLM summarization.

    Args:
        title: Paper title.
        abstract: Paper abstract.

    Returns:
        Prompt string.
    """
    return (
        "Summarize the paper in English using exactly three lines. "
        "Each line must start with 'Purpose:', 'Method:', and 'Results:' respectively. "
        "Each line should be a single sentence and must stay faithful to the abstract.\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
    )


def parse_llm_summary(text: str) -> Optional[tuple[str, str, str]]:
    """Parse LLM summary into purpose/method/results.

    Args:
        text: Raw LLM response text.

    Returns:
        Tuple of purpose/method/results or None if parse failed.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 3:
        return None
    purpose_line = next((line for line in lines if line.lower().startswith("purpose:")), None)
    method_line = next((line for line in lines if line.lower().startswith("method:")), None)
    results_line = next((line for line in lines if line.lower().startswith("results:")), None)
    if not (purpose_line and method_line and results_line):
        return None
    return (
        purpose_line.split(":", 1)[1].strip() or "Not stated.",
        method_line.split(":", 1)[1].strip() or "Not stated.",
        results_line.split(":", 1)[1].strip() or "Not stated.",
    )


def call_ollama(prompt: str, endpoint: str, model: str) -> Optional[str]:
    """Call Ollama to generate a summary.

    Args:
        prompt: Prompt text.
        endpoint: Ollama endpoint.
        model: Model name.

    Returns:
        Response text or None if failed.
    """
    url = f"{endpoint.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("response")


def summarize_with_ollama(
    title: str, abstract: str, endpoint: str, model: str
) -> Optional[tuple[str, str, str]]:
    """Summarize using Ollama with a retry on failure.

    Args:
        title: Paper title.
        abstract: Paper abstract.
        endpoint: Ollama endpoint.
        model: Model name.

    Returns:
        Parsed summary or None if failed.
    """
    prompt = build_ollama_prompt(title=title, abstract=abstract)
    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        try:
            response_text = call_ollama(prompt, endpoint=endpoint, model=model)
        except requests.RequestException:
            response_text = None
        if response_text:
            parsed = parse_llm_summary(response_text)
            if parsed:
                return parsed
        if attempt < OLLAMA_MAX_RETRIES:
            import time

            time.sleep(OLLAMA_RETRY_DELAY_SECONDS)
    return None


def format_authors(authors: Sequence[arxiv.Result.Author]) -> str:
    """Format author list for display.

    Args:
        authors: arXiv author list.

    Returns:
        Formatted author string.
    """
    names = [author.name for author in authors]
    if len(names) <= 3:
        return ", ".join(names)
    return f"{', '.join(names[:3])}, et al."


def ensure_utc(value: dt.datetime) -> dt.datetime:
    """Normalize datetime to UTC (aware).

    Args:
        value: Datetime to normalize.

    Returns:
        UTC-aware datetime.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def should_include(published: dt.datetime, cutoff: dt.datetime) -> bool:
    """Check if a paper is within the cutoff window.

    Args:
        published: Paper published datetime.
        cutoff: Earliest allowed datetime.

    Returns:
        True if within window.
    """
    published_utc = ensure_utc(published)
    cutoff_utc = ensure_utc(cutoff)
    return published_utc >= cutoff_utc


def summarize_results(
    results: Iterable[arxiv.Result],
    cutoff: dt.datetime,
    use_llm: bool,
    endpoint: str,
    model: str,
    require_llm: bool,
    skip_failed: bool,
) -> List[PaperSummary]:
    """Summarize arXiv results into structured records.

    Args:
        results: arXiv results iterator.
        cutoff: Earliest allowed datetime.

    Returns:
        List of summarized papers.
    """
    summaries: List[PaperSummary] = []
    for result in results:
        if not should_include(result.published, cutoff):
            continue
        if use_llm:
            llm_summary = summarize_with_ollama(
                title=result.title,
                abstract=result.summary,
                endpoint=endpoint,
                model=model,
            )
        else:
            llm_summary = None
        if llm_summary:
            purpose, method, results_sentence = llm_summary
        else:
            if require_llm and use_llm:
                if skip_failed:
                    continue
                raise RuntimeError(
                    "Ollama summarization failed while --require-llm is enabled "
                    f"for paper {result.get_short_id()}."
                )
            purpose, method, results_sentence = summarize_abstract_sections(result.summary)
        tag_keys, tags = classify_tags(
            title=result.title,
            abstract=result.summary,
            purpose=purpose,
            method=method,
            results=results_sentence,
        )
        summary = PaperSummary(
            arxiv_id=result.get_short_id(),
            title=" ".join(result.title.split()),
            authors=format_authors(result.authors),
            published=result.published.date(),
            updated=result.updated.date(),
            categories=", ".join(result.categories),
            link=result.entry_id,
            abstract=" ".join(result.summary.split()),
            purpose=purpose,
            method=method,
            results=results_sentence,
            tag_keys=tag_keys,
            tags=tags,
        )
        summaries.append(summary)
    return summaries


def render_markdown(summaries: Sequence[PaperSummary], query: str, days_back: int) -> str:
    """Render collected summaries to Markdown.

    Args:
        summaries: Summarized papers.
        query: arXiv query used.
        days_back: Days back window.

    Returns:
        Markdown string.
    """
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    header_lines = [
        "# LLM as a Judge — arXiv Monitor",
        "",
        f"Last updated: {today} (UTC)",
        f"Coverage window: last {days_back} days",
        "",
        "Query:",
        f"`{query}`",
        "",
        f"Total papers: {len(summaries)}",
        "",
        "## Papers",
        "",
    ]
    sections: List[str] = []
    for paper in summaries:
        sections.extend(
            [
                f"### {paper.title} ({paper.arxiv_id})",
                "",
                f"Authors: {paper.authors}",
                f"Published: {paper.published.isoformat()}",
                f"Updated: {paper.updated.isoformat()}",
                f"Categories: {paper.categories}",
                f"Link: [arXiv]({paper.link})",
                "",
                f"Abstract: {paper.abstract}",
                "",
                "Summary:",
                f"Purpose: {paper.purpose}",
                f"Method: {paper.method}",
                f"Results: {paper.results}",
                f"Tags: {', '.join(paper.tags)}",
                "",
            ]
        )
    return "\n".join(header_lines + sections).strip() + "\n"


def render_daily_markdown(
    summaries: Sequence[PaperSummary],
    report_date: dt.date,
    query: str,
    days_back: int,
) -> str:
    """Render a daily Markdown report.

    Args:
        summaries: Summarized papers.
        report_date: Report date (UTC).
        query: arXiv query used.
        days_back: Days back window.

    Returns:
        Markdown string.
    """
    header_lines = [
        "# LLM as a Judge — Daily Report",
        "",
        f"Report date: {report_date.isoformat()} (UTC)",
        f"Coverage window: last {days_back} days",
        "",
        "Query:",
        f"`{query}`",
        "",
        f"Total papers: {len(summaries)}",
        "",
        "## Papers",
        "",
    ]
    sections: List[str] = []
    for paper in summaries:
        sections.extend(
            [
                f"### {paper.title} ({paper.arxiv_id})",
                "",
                f"Authors: {paper.authors}",
                f"Published: {paper.published.isoformat()}",
                f"Updated: {paper.updated.isoformat()}",
                f"Categories: {paper.categories}",
                f"Link: [arXiv]({paper.link})",
                "",
                f"Abstract: {paper.abstract}",
                "",
                "Summary:",
                f"Purpose: {paper.purpose}",
                f"Method: {paper.method}",
                f"Results: {paper.results}",
                f"Tags: {', '.join(paper.tags)}",
                "",
            ]
        )
    return "\n".join(header_lines + sections).strip() + "\n"


def render_daily_index(daily_dir: str) -> str:
    """Render an index of daily reports.

    Args:
        daily_dir: Directory containing daily reports.

    Returns:
        Markdown string.
    """
    path = pathlib.Path(daily_dir)
    reports: List[dt.date] = []
    if path.exists():
        for entry in path.glob("*.md"):
            if entry.name == "index.md":
                continue
            try:
                reports.append(dt.date.fromisoformat(entry.stem))
            except ValueError:
                continue
    reports.sort(reverse=True)
    lines = [
        "# Daily Reports Index",
        "",
        f"Total days: {len(reports)}",
        "",
    ]
    for report_date in reports:
        filename = f"{report_date.isoformat()}.md"
        lines.append(f"- [{report_date.isoformat()}]({filename})")
    return "\n".join(lines).strip() + "\n"


def filter_by_days_back(
    summaries: Sequence[PaperSummary], report_date: dt.date, days_back: int
) -> List[PaperSummary]:
    """Filter summaries by a days-back window.

    Args:
        summaries: All summaries.
        report_date: Report date (UTC).
        days_back: Days back window.

    Returns:
        Filtered summaries.
    """
    cutoff = report_date - dt.timedelta(days=days_back)
    return [summary for summary in summaries if summary.published >= cutoff]


def filter_by_date(
    summaries: Sequence[PaperSummary], target_date: dt.date
) -> List[PaperSummary]:
    """Filter summaries by a specific published date.

    Args:
        summaries: All summaries.
        target_date: Target published date (UTC).

    Returns:
        Filtered summaries.
    """
    return [summary for summary in summaries if summary.published == target_date]


def collect_papers(
    days_back: int,
    max_results: int,
    use_llm: bool,
    endpoint: str,
    model: str,
    require_llm: bool,
    skip_failed: bool,
) -> List[PaperSummary]:
    """Collect papers from arXiv and summarize them.

    Args:
        days_back: Days back window.
        max_results: Max number of results to fetch.

    Returns:
        List of summaries.
    """
    query = build_arxiv_query(QUERY_TERMS)
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_back)
    results = client.results(search)
    summaries = summarize_results(
        results,
        cutoff,
        use_llm=use_llm,
        endpoint=endpoint,
        model=model,
        require_llm=require_llm,
        skip_failed=skip_failed,
    )
    summaries.sort(key=lambda item: item.published, reverse=True)
    return summaries


def load_seen_ids(cache: Cache) -> set[str]:
    """Load seen IDs from cache.

    Args:
        cache: DiskCache cache.

    Returns:
        Set of seen arXiv IDs.
    """
    return set(cache.get("seen_ids", set()))


def update_seen_ids(cache: Cache, summaries: Sequence[PaperSummary]) -> None:
    """Update cache with new seen IDs.

    Args:
        cache: DiskCache cache.
        summaries: Summaries to add.
    """
    seen = load_seen_ids(cache)
    seen.update(summary.arxiv_id for summary in summaries)
    cache.set("seen_ids", seen)
    cache.set("last_updated", dt.datetime.now(dt.timezone.utc).isoformat())


def render_tags_csv(summaries: Sequence[PaperSummary]) -> str:
    """Render paper tags to CSV.

    Args:
        summaries: Summaries to serialize.

    Returns:
        CSV string.
    """
    headers = [
        "arxiv_id",
        "title",
        "published",
        "updated",
        "categories",
        "link",
        "tag_keys",
        "tags",
    ]
    lines = [",".join(headers)]
    for summary in summaries:
        row = [
            summary.arxiv_id,
            summary.title.replace('"', '""'),
            summary.published.isoformat(),
            summary.updated.isoformat(),
            summary.categories.replace('"', '""'),
            summary.link,
            ";".join(summary.tag_keys),
            ";".join(summary.tags),
        ]
        escaped = [f'"{value}"' for value in row]
        lines.append(",".join(escaped))
    return "\n".join(lines).strip() + "\n"


def render_tags_json(summaries: Sequence[PaperSummary]) -> str:
    """Render paper tags to JSON.

    Args:
        summaries: Summaries to serialize.

    Returns:
        JSON string.
    """
    payload = [
        {
            "arxiv_id": summary.arxiv_id,
            "title": summary.title,
            "published": summary.published.isoformat(),
            "updated": summary.updated.isoformat(),
            "categories": summary.categories,
            "link": summary.link,
            "tag_keys": list(summary.tag_keys),
            "tags": list(summary.tags),
        }
        for summary in summaries
    ]
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def build_tag_counts(summaries: Sequence[PaperSummary]) -> Dict[str, int]:
    """Build tag counts from summaries.

    Args:
        summaries: Summaries to count.

    Returns:
        Mapping of tag label to count.
    """
    counts: Dict[str, int] = {}
    for summary in summaries:
        tags = summary.tags or ["Unclassified"]
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def render_tag_pie_chart(tag_counts: Dict[str, int]) -> str:
    """Render a Mermaid pie chart for tag counts.

    Args:
        tag_counts: Tag label to count.

    Returns:
        Mermaid chart block.
    """
    lines = [
        "```mermaid",
        "pie title LLM-as-a-Judge Subclass Counts",
    ]
    if not tag_counts:
        tag_counts = {"Unclassified": 0}
    for label, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f'    "{label}" : {count}')
    lines.append("```")
    return "\n".join(lines)


def update_readme_tag_chart(readme_path: str, tag_counts: Dict[str, int]) -> None:
    """Update README with tag chart.

    Args:
        readme_path: README path.
        tag_counts: Tag counts for the chart.
    """
    path = pathlib.Path(readme_path)
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    start_index = content.find(TAG_STATS_START)
    end_index = content.find(TAG_STATS_END)
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        return
    chart = render_tag_pie_chart(tag_counts)
    replacement = f"{TAG_STATS_START}\n{chart}\n{TAG_STATS_END}"
    updated = (
        content[:start_index]
        + replacement
        + content[end_index + len(TAG_STATS_END) :]
    )
    path.write_text(updated, encoding="utf-8")


def write_text(path: str, content: str) -> None:
    """Write text to disk, creating directories if needed.

    Args:
        path: Output path.
        content: Text content.
    """
    output_dir = path.rsplit("/", 1)[0] if "/" in path else "."
    if output_dir and output_dir != ".":
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Collect LLM-as-a-judge papers from arXiv.")
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK)
    parser.add_argument("--max-results", type=int, default=DEFAULT_MAX_RESULTS)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--tags-json", type=str, default=DEFAULT_TAGS_JSON_PATH)
    parser.add_argument("--tags-csv", type=str, default=DEFAULT_TAGS_CSV_PATH)
    parser.add_argument("--daily-dir", type=str, default=DEFAULT_DAILY_DIR)
    parser.add_argument("--daily-index", type=str, default=DEFAULT_DAILY_INDEX_PATH)
    parser.add_argument("--backfill-days", type=int, default=DEFAULT_BACKFILL_DAYS)
    parser.add_argument(
        "--backfill-skip-existing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_BACKFILL_SKIP_EXISTING,
        help="Skip existing daily files when backfilling.",
    )
    parser.add_argument("--readme-path", type=str, default=DEFAULT_README_PATH)
    parser.add_argument("--ollama-endpoint", type=str, default=OLLAMA_ENDPOINT)
    parser.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--require-llm",
        action="store_true",
        help="Fail if Ollama is unreachable or summarization fails.",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip papers when Ollama summarization fails.",
    )
    return parser.parse_args()


def ensure_ollama_available(endpoint: str) -> None:
    """Ensure Ollama is reachable.

    Args:
        endpoint: Ollama endpoint.
    """
    url = f"{endpoint.rstrip('/')}/api/tags"
    response = requests.get(url, timeout=OLLAMA_TIMEOUT_SECONDS)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama check failed with status {response.status_code}.")


def main() -> None:
    """Run the collection pipeline."""
    args = parse_args()
    if args.require_llm and args.no_llm:
        raise RuntimeError("Cannot use --require-llm with --no-llm.")
    if args.require_llm:
        ensure_ollama_available(args.ollama_endpoint)
    report_date = dt.datetime.now(dt.timezone.utc).date()
    max_days = max(args.days_back, args.backfill_days)
    summaries_all = collect_papers(
        days_back=max_days,
        max_results=args.max_results,
        use_llm=not args.no_llm,
        endpoint=args.ollama_endpoint,
        model=args.ollama_model,
        require_llm=args.require_llm,
        skip_failed=args.skip_failed,
    )
    summaries = filter_by_days_back(summaries_all, report_date, args.days_back)
    with Cache(CACHE_DIR) as cache:
        update_seen_ids(cache, summaries)
    query = build_arxiv_query(QUERY_TERMS)
    markdown = render_markdown(summaries, query=query, days_back=args.days_back)
    write_text(args.output, markdown)
    write_text(args.tags_json, render_tags_json(summaries))
    write_text(args.tags_csv, render_tags_csv(summaries))
    update_readme_tag_chart(args.readme_path, build_tag_counts(summaries_all))
    if args.backfill_days > 0:
        for offset in range(args.backfill_days):
            day = report_date - dt.timedelta(days=offset)
            daily_path = f"{args.daily_dir}/{day.isoformat()}.md"
            if args.backfill_skip_existing and pathlib.Path(daily_path).exists():
                continue
            daily_summaries = filter_by_date(summaries_all, day)
            daily_markdown = render_daily_markdown(
                daily_summaries,
                report_date=day,
                query=query,
                days_back=1,
            )
            write_text(daily_path, daily_markdown)
        write_text(args.daily_index, render_daily_index(args.daily_dir))
    else:
        daily_markdown = render_daily_markdown(
            summaries,
            report_date=report_date,
            query=query,
            days_back=args.days_back,
        )
        daily_path = f"{args.daily_dir}/{report_date.isoformat()}.md"
        write_text(daily_path, daily_markdown)
        write_text(args.daily_index, render_daily_index(args.daily_dir))


if __name__ == "__main__":
    main()
