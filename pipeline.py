"""
pipeline.py — step functions for the AIM26 content pipeline.

Each function takes an anthropic client, relevant inputs, and a stats dict
(mutated in place) for tracking token usage and API call counts.
"""

import re
import time
from typing import Optional

import anthropic
import requests
from bs4 import BeautifulSoup

import prompts
from brief_parser import format_criteria_list
from screener import is_editorial_by_url

MODEL = "claude-sonnet-4-6"


def _call_claude(
    client: anthropic.Anthropic,
    system: str,
    user: str,
    max_tokens: int,
    stats: dict,
    messages: list | None = None,
) -> anthropic.types.Message:
    """
    Claude API call with retry on timeout, updating stats in place.
    Pass `messages` to use a custom multi-turn history; otherwise a single
    user turn is constructed from `user`.
    """
    msg_list = messages if messages is not None else [{"role": "user", "content": user}]
    for attempt in range(2):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=msg_list,
            )
            stats["total_input_tokens"] += response.usage.input_tokens
            stats["total_output_tokens"] += response.usage.output_tokens
            stats["total_api_calls"] += 1
            return response
        except anthropic.APITimeoutError:
            if attempt == 0:
                time.sleep(10)
                continue
            raise
    raise RuntimeError("Unexpected exit from retry loop")


def _text(response: anthropic.types.Message) -> str:
    return response.content[0].text


# ── Step 0: Competitor fetch ───────────────────────────────────────────────────

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _search_urls(
    client: anthropic.Anthropic, keyword: str, max_results: int, stats: dict
) -> list[str]:
    """Return a list of URLs from a web search for the keyword."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{
            "role": "user",
            "content": (
                f"Search for: {keyword}\n\n"
                f"Return a plain numbered list of the top {max_results} result URLs, "
                "one per line. No descriptions, no commentary — just URLs."
            ),
        }],
    )
    stats["total_api_calls"] += 1
    stats["total_input_tokens"] += response.usage.input_tokens
    stats["total_output_tokens"] += response.usage.output_tokens

    text = "".join(block.text for block in response.content if hasattr(block, "text"))
    urls = re.findall(r"https?://[^\s\)\]\,\"\'<>]+", text)
    urls = [u.rstrip(".,;:!?") for u in urls]
    return list(dict.fromkeys(urls))[:max_results]


def _fetch_page_text(url: str) -> str:
    """Fetch a URL and return cleaned visible text (max 6 000 chars)."""
    resp = requests.get(url, headers=_FETCH_HEADERS, timeout=15, allow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()]
    return "\n".join(lines)[:6000]


def _extract_content(
    client: anthropic.Anthropic, url: str, page_text: str, stats: dict
) -> str:
    """Use Claude to extract structured marketing copy from raw page text."""
    user_msg = prompts.EXTRACTION_PROMPT.format(url=url, page_text=page_text)
    response = _call_claude(client, "", user_msg, max_tokens=2000, stats=stats)
    return _text(response)


def _classify_page(
    client: anthropic.Anthropic, url: str, content: str, stats: dict
) -> str:
    """Classify a page as BRAND or EDITORIAL. Returns 'BRAND' or 'EDITORIAL'."""
    user_msg = prompts.CLASSIFIER_USER.format(url=url, content_preview=content[:500])
    response = _call_claude(client, prompts.CLASSIFIER_SYSTEM, user_msg, max_tokens=10, stats=stats)
    return _text(response).strip().upper()


def fetch_competitor_pages(
    client: anthropic.Anthropic,
    keyword: str,
    stats: dict,
    target_count: int = 5,
    max_attempts: int = 15,
    on_url_processed=None,
) -> tuple[list[dict], list[dict]]:
    """
    Search for and fetch brand competitor pages for a keyword.

    on_url_processed: optional callable(screening_log: list[dict]) called after
    each URL is processed — use to update a live display in the UI.

    Returns (competitor_pages, screening_log).
    competitor_pages: list of {"url": str, "content": str}
    screening_log: list of {"url", "stage", "result", "reason", "included"}
    """
    results: list[dict] = []
    screening_log: list[dict] = []

    urls = _search_urls(client, keyword, max_attempts, stats)

    for url in urls:
        if len(results) >= target_count:
            break

        short_url = url[:70] + "…" if len(url) > 70 else url

        # Stage 1 — URL screen
        is_editorial, reason = is_editorial_by_url(url)
        if is_editorial:
            screening_log.append({
                "URL": short_url,
                "Stage": "URL screen",
                "Result": "skip",
                "Reason": reason,
                "Included": False,
            })
            if on_url_processed:
                on_url_processed(screening_log)
            continue

        # Stage 2 — Fetch
        try:
            page_text = _fetch_page_text(url)
        except Exception as exc:
            screening_log.append({
                "URL": short_url,
                "Stage": "Fetch",
                "Result": "error",
                "Reason": str(exc)[:80],
                "Included": False,
            })
            if on_url_processed:
                on_url_processed(screening_log)
            continue

        # Stage 3 — Extract
        try:
            extracted = _extract_content(client, url, page_text, stats)
        except Exception as exc:
            screening_log.append({
                "URL": short_url,
                "Stage": "Extract",
                "Result": "error",
                "Reason": str(exc)[:80],
                "Included": False,
            })
            if on_url_processed:
                on_url_processed(screening_log)
            continue

        # Stage 4 — Classify
        classification = _classify_page(client, url, extracted, stats)

        if classification == "BRAND":
            results.append({"url": url, "content": extracted})
            screening_log.append({
                "URL": short_url,
                "Stage": "Classify",
                "Result": "BRAND ✓",
                "Reason": "passed all screens",
                "Included": True,
            })
        else:
            screening_log.append({
                "URL": short_url,
                "Stage": "Classify",
                "Result": "editorial",
                "Reason": "content classified as editorial",
                "Included": False,
            })

        if on_url_processed:
            on_url_processed(screening_log)

    return results, screening_log


def build_competitive_context(competitor_pages: list[dict]) -> str:
    """Assemble fetched competitor pages into the same format as the manual context doc."""
    parts = ["COMPETITIVE CONTEXT\n"]
    for i, page in enumerate(competitor_pages, 1):
        parts.append(f"--- Competitor {i}: {page['url']} ---")
        parts.append(page["content"])
        parts.append("")
    return "\n".join(parts)


# ── Step 2 ─────────────────────────────────────────────────────────────────────

def run_competitive_analysis(
    client: anthropic.Anthropic,
    competitive_context: str,
    stats: dict,
) -> str:
    user_msg = prompts.COMPETITIVE_ANALYSIS_USER.format(
        competitive_context=competitive_context
    )
    response = _call_claude(
        client,
        prompts.COMPETITIVE_ANALYSIS_SYSTEM,
        user_msg,
        max_tokens=1000,
        stats=stats,
    )
    return _text(response)


# ── Step 3 ─────────────────────────────────────────────────────────────────────

def _is_complete_html(html: str) -> bool:
    stripped = html.strip()
    starts_ok = stripped.lower().startswith("<!doctype") or stripped.lower().startswith("<html")
    ends_ok = stripped.lower().endswith("</html>")
    return starts_ok and ends_ok


def _doc_block(text: str, title: str, context: str) -> dict:
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": text},
        "title": title,
        "context": context,
    }


def run_html_generation(
    client: anthropic.Anthropic,
    brief_text: str,
    brand_name: str,
    competitive_summary: str,
    stats: dict,
    previous_html: str | None = None,
    latest_feedback: str | None = None,
) -> tuple[str, int]:
    """
    Generate or edit HTML.
    - First iteration: full generation from brief + competitive summary.
    - Subsequent iterations: targeted edit of previous_html guided by latest_feedback.
    Returns (html_string, output_token_count).
    """
    if previous_html and latest_feedback:
        # Edit mode — fix only the failures identified in the latest evaluation
        instruction = prompts.HTML_EDIT_INSTRUCTION.format(
            brand_name=brand_name,
            feedback_text=latest_feedback,
        )
        user_content = [
            _doc_block(
                previous_html,
                title="Current Landing Page HTML",
                context="The existing landing page that needs targeted edits based on the evaluation feedback below.",
            ),
            _doc_block(
                brief_text,
                title="Content Brief",
                context="The original content brief — use this to verify any edits remain aligned with brand, audience, and criteria.",
            ),
            {"type": "text", "text": instruction},
        ]
        system = prompts.HTML_EDIT_SYSTEM
    else:
        # Generation mode — build page from scratch
        instruction = prompts.HTML_GENERATION_INSTRUCTION.format(brand_name=brand_name)
        user_content = [
            _doc_block(
                brief_text,
                title="Content Brief",
                context="The content brief for this landing page — brand, audience, goals, criteria, and page sections.",
            ),
            _doc_block(
                competitive_summary,
                title="Competitive Analysis Summary",
                context="Patterns, gaps, and structural observations from competitor pages in this category.",
            ),
            {"type": "text", "text": instruction},
        ]
        system = prompts.HTML_GENERATION_SYSTEM

    response = _call_claude(
        client,
        system,
        "",
        max_tokens=8000,
        stats=stats,
        messages=[{"role": "user", "content": user_content}],
    )
    html = _text(response)
    token_count = response.usage.output_tokens

    # Continuation loop — up to 3 extra calls, each with full conversation context
    for _ in range(3):
        if _is_complete_html(html):
            break
        cont_response = _call_claude(
            client,
            system,
            "",
            max_tokens=8000,
            stats=stats,
            messages=[
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": html},
                {"role": "user", "content": prompts.HTML_CONTINUATION_USER},
            ],
        )
        html = html + _text(cont_response)
        token_count += cont_response.usage.output_tokens

    return html, token_count


# ── Step 4 ─────────────────────────────────────────────────────────────────────

_EVALUATOR_SPECS = [
    ("Conversion Rate Optimizer", prompts.EVALUATOR_1_SYSTEM),
    ("Target Reader",             prompts.EVALUATOR_2_SYSTEM),
    ("Brand Voice Auditor",       prompts.EVALUATOR_3_SYSTEM),
]


def run_evaluation(
    client: anthropic.Anthropic,
    html_content: str,
    criteria: list[str],
    stats: dict,
    on_progress=None,
    evaluator_indices: list[int] | None = None,
    competitive_context: str | None = None,
) -> dict:
    """
    Run selected evaluators then synthesize with minimum scores.
    evaluator_indices: which of the 3 evaluators to use (default all).

    Returns dict with keys:
        evaluations  — list of (name, raw_text) per selected evaluator
        synthesis    — synthesis text (markdown table + paragraph + FIX blocks)
        total_score  — int parsed from synthesis TOTAL line, or None
        max_score    — int (n_criteria * 3)
    """
    if evaluator_indices is None:
        evaluator_indices = list(range(len(_EVALUATOR_SPECS)))

    selected_specs = [_EVALUATOR_SPECS[i] for i in evaluator_indices]
    criteria_text = format_criteria_list(criteria)
    max_score = len(criteria) * 3
    user_msg = prompts.EVALUATOR_USER.format(criteria=criteria_text, html=html_content)
    if competitive_context:
        user_msg = (
            "COMPETITOR REFERENCE — use this to check any exclusions about copied language or patterns:\n"
            f"{competitive_context}\n\n"
            + user_msg
        )

    evaluations = []
    for name, system in selected_specs:
        if on_progress:
            on_progress(f"{name} evaluating...")
        response = _call_claude(client, system, user_msg, max_tokens=1500, stats=stats)
        evaluations.append((name, _text(response)))

    if len(evaluations) == 1:
        # Single evaluator — no synthesis call needed; use the evaluation directly
        synthesis_text = evaluations[0][1]
    else:
        if on_progress:
            on_progress("Synthesizing evaluations...")
        eval_blocks = "\n\n".join(
            f"EVALUATOR {i + 1} — {name}:\n{text}"
            for i, (name, text) in enumerate(evaluations)
        )
        synthesis_msg = (
            eval_blocks
            + "\n\nProduce the synthesis as specified in your instructions. "
            "Use minimum scores per criterion. Calculate the total from minimum scores only."
        )
        synthesis_response = _call_claude(
            client, prompts.EVALUATOR_SYNTHESIS_SYSTEM, synthesis_msg, max_tokens=2000, stats=stats
        )
        synthesis_text = _text(synthesis_response)

    parsed_score = _parse_score(synthesis_text)
    if parsed_score is not None:
        total_score = parsed_score
        parse_method = "text"
    else:
        total_score = _sum_min_scores(evaluations)
        parse_method = "fallback_sum" if total_score is not None else "failed"

    return {
        "evaluations": evaluations,
        "synthesis": synthesis_text,
        "total_score": total_score,
        "max_score": max_score,
        "score_parse_method": parse_method,
    }


def _parse_score(evaluation_text: str) -> Optional[int]:
    """
    Extract numeric total from synthesis text, trying multiple formats in order:
    1. TOTAL: X/Y
    2. Total score: X
    3. X out of Y
    4. X/Y on a line by itself
    Returns None if all patterns fail.
    """
    # Pattern 1: TOTAL: X/Y  (original)
    m = re.search(r"TOTAL:\s*(\d+)/\d+", evaluation_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Pattern 2: Total score: X
    m = re.search(r"total\s+score[:\s]+(\d+)", evaluation_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Pattern 3: X out of Y
    m = re.search(r"\b(\d+)\s+out\s+of\s+\d+", evaluation_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Pattern 4: bare X/Y on its own line
    m = re.search(r"^\s*(\d+)/(\d+)\s*$", evaluation_text, re.MULTILINE)
    if m:
        return int(m.group(1))
    return None


def _sum_min_scores(evaluations: list[tuple[str, str]]) -> Optional[int]:
    """
    Fallback scorer: for each criterion, take the minimum score across all
    evaluators and sum them. Reliable when the synthesis total cannot be parsed.
    """
    from collections import defaultdict
    criterion_scores: dict[int, list[int]] = defaultdict(list)
    for _, eval_text in evaluations:
        for row in parse_evaluation_rows(eval_text):
            cnum = int(row["Criterion"])
            score = int(row["Score"].split("/")[0])
            criterion_scores[cnum].append(score)
    if not criterion_scores:
        return None
    return sum(min(scores) for scores in criterion_scores.values())


def parse_evaluation_rows(evaluation_text: str) -> list[dict]:
    """
    Parse per-criterion lines into list of dicts with keys:
    criterion_num, score, reasoning.
    """
    rows = []
    pattern = re.compile(
        r"Criterion\s+(\d+):\s*(\d)/3\s*[—–-]+\s*(.+?)(?=\nCriterion|\nTOTAL|\nEVALUATOR|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern.finditer(evaluation_text):
        rows.append(
            {
                "Criterion": int(m.group(1)),
                "Score": f"{m.group(2)}/3",
                "Reasoning": m.group(3).strip(),
            }
        )
    return rows


def parse_evaluator_note(evaluation_text: str) -> str:
    match = re.search(r"EVALUATOR NOTE:\s*(.+)", evaluation_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# ── Step 6: Change log ─────────────────────────────────────────────────────────

def run_change_log(
    client: anthropic.Anthropic,
    v1_html: str,
    final_html: str,
    stats: dict,
) -> str:
    user_msg = prompts.CHANGE_LOG_USER.format(
        v1_html=v1_html,
        final_html=final_html,
    )
    response = _call_claude(
        client,
        prompts.CHANGE_LOG_SYSTEM,
        user_msg,
        max_tokens=1500,
        stats=stats,
    )
    return _text(response)


def parse_change_log_rows(change_log_text: str) -> list[dict]:
    """Parse change log into list of dicts: Section, What changed, Why (inferred)."""
    rows = []
    blocks = re.split(r"\n(?=Section:)", change_log_text.strip())
    for block in blocks:
        section_m = re.search(r"Section:\s*(.+)", block)
        what_m = re.search(r"What changed:\s*(.+?)(?=\nWhy|\Z)", block, re.DOTALL)
        why_m = re.search(r"Why \(inferred\):\s*(.+)", block, re.DOTALL)
        if section_m and what_m:
            rows.append(
                {
                    "Section": section_m.group(1).strip(),
                    "What changed": what_m.group(1).strip(),
                    "Why (inferred)": why_m.group(1).strip() if why_m else "",
                }
            )
    return rows
