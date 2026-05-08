"""
brief_parser.py — extracts structured fields from a plain-text content brief.

Supports two paste formats:
  1. ## Section Name markers (original template)
  2. Word table tab-separated rows: "Field *<TAB>Value" (alternate template)
"""

import re
from typing import Optional


MANDATORY_FIELDS = [
    "Brand",
    "Target reader",
    "Conversion goal",
    "Primary keyword",
    "Content criteria",
    "Explicit exclusions",
    "Brand voice",
]

ALL_FIELDS = MANDATORY_FIELDS + ["Competitive insights", "Competitive gaps", "Page sections"]


def _strip_hint(text: str) -> str:
    """
    When a Word table is pasted as plain text, the hint from the left column
    appears at the start of each section's content. Three cases:

    1. First line STARTS with a list marker (1/, 1., -, •):
       No hint present — keep everything as-is.

    2. First line CONTAINS a list marker mid-line:
       Word merged hint text (left cell) and first list item (right cell) onto
       one line. Slice from the marker onward and keep remaining lines.

    3. First line is pure prose hint with no list marker:
       Skip it entirely; the answer starts on line 2.

    Falls back to the original text if nothing survives.
    """
    lines = text.splitlines()
    if len(lines) <= 1:
        return text.strip()

    first = lines[0].strip()
    rest_lines = "\n".join(lines[1:]).strip()

    # Case 1 — first line is already the answer
    if re.match(r"^\d+[/\.\)]\s|^[-•*]\s", first):
        return text.strip()

    # Case 2 — hint and first list item merged onto one line
    mid = re.search(r"\d+[/\.\)]\s", first)
    if mid:
        answer_start = first[mid.start():]
        return (answer_start + ("\n" + rest_lines if rest_lines else "")).strip()

    # Case 3 — pure hint line, skip it
    return rest_lines if rest_lines else text.strip()


def _extract_section(text: str, section_name: str) -> Optional[str]:
    """Return student-answer text under a ## section heading, or None if not found.
    Accepts headings like '## Brand *' via [^\n]* after the name."""
    pattern = rf"##\s*{re.escape(section_name)}[^\n]*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return _strip_hint(match.group(1))


def _parse_list(text: str) -> list[str]:
    """Convert a numbered or bulleted text block into a list of strings."""
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading bullets, numbers, dashes (handles 1. 1) 1/ formats)
        cleaned = re.sub(r"^[\d]+[\.\)/]\s*", "", line)
        cleaned = re.sub(r"^[-•*]\s*", "", cleaned)
        if cleaned:
            items.append(cleaned)
    return items


def _sanitize(text: str) -> str:
    """
    Strip or normalize invisible/non-standard characters that Word inserts on copy-paste.
    Applied once at the top of parse_brief before any regex runs.
    """
    return (
        text
        .replace("¬", "")   # ¬  NOT SIGN — Word paragraph-mark artifact
        .replace("­", "")   # soft hyphen — invisible but breaks matching
        .replace("​", "")   # zero-width space
        .replace("﻿", "")   # BOM / zero-width no-break space
        .replace(" ", " ")  # non-breaking space → regular space
    )


# Maps lowercase tab-format field labels → canonical field names.
# Handles aliases like "Target *" → "Target reader".
_TAB_ALIASES: dict[str, Optional[str]] = {
    "brand":                    "Brand",
    "target":                   "Target reader",
    "target reader":            "Target reader",
    "conversion goal":          "Conversion goal",
    "primary keyword":          "Primary keyword",
    "content criteria":         "Content criteria",
    "explicit exclusions":      "Explicit exclusions",
    "brand voice":              "Brand voice",
    "competitive insights":     "Competitive insights",
    "competitive gaps":         "Competitive gaps",
    "page sections":            "Page sections",
    # rows to ignore
    "field":                    None,
    "your answer":              None,
    "customer journey moment":  None,
}


def _is_tab_format(text: str) -> bool:
    """Return True if the brief uses the tab-separated Word table format."""
    return bool(re.search(
        r"^(Brand|Target|Conversion\s+goal|Primary\s+keyword|"
        r"Content\s+criteria|Explicit\s+exclusions|Brand\s+voice)"
        r"\s*\*?\s*\t",
        text, re.IGNORECASE | re.MULTILINE,
    ))


def _parse_tab_brief(text: str) -> dict:
    """
    Parse a Word tab-separated table brief.
    Each row is "Field label *<TAB>value"; multi-line values continue on
    subsequent lines until the next tab-containing field row.
    """
    raw: dict[str, list[str]] = {f: [] for f in ALL_FIELDS}
    current: Optional[str] = None

    for line in text.splitlines():
        if "\t" in line:
            parts = line.split("\t", 1)
            label = parts[0].strip().rstrip("*").strip().lower()
            immediate = parts[1].strip()

            if not label:
                # Empty label = tab-prefixed continuation (\tContent).
                if current and immediate:
                    raw[current].append(immediate)
                continue

            # Resolve alias; try exact then partial match
            canonical = _TAB_ALIASES.get(label)
            recognized = label in _TAB_ALIASES
            if not recognized:
                for key, val in _TAB_ALIASES.items():
                    if key and key in label:
                        canonical = val
                        recognized = True
                        break

            if recognized:
                # Known field or explicitly ignored — update current
                current = canonical
                if current and immediate:
                    raw[current].append(immediate)
            else:
                # Unrecognized label: the line is student content with an
                # embedded tab (e.g. a criterion bullet with a tab stop).
                # Treat the whole line as content for the current field.
                if current:
                    raw[current].append(line.replace("\t", " ").strip())
        elif line.strip():
            # Check if this no-tab line is a bare field label (e.g. "Brand *")
            label_only = line.strip().rstrip("*").strip().lower()
            canonical = _TAB_ALIASES.get(label_only)
            recognized = label_only in _TAB_ALIASES
            if not recognized:
                for key, val in _TAB_ALIASES.items():
                    if key and key in label_only:
                        canonical = val
                        recognized = True
                        break
            if recognized:
                current = canonical
            elif current:
                raw[current].append(line.strip())

    result = {}
    for field in ALL_FIELDS:
        value = "\n".join(raw.get(field, [])).strip()
        if field in ("Content criteria", "Explicit exclusions"):
            result[field] = _parse_list(value) if value else []
        else:
            result[field] = value
    return result


def parse_brief(brief_text: str) -> dict:
    """
    Parse a structured brief and return a dict of field names to values.

    Accepts both the ## header format and the tab-separated Word table format.
    List fields (Content criteria, Explicit exclusions) are returned as list[str].
    All other fields are returned as str.

    Raises ValueError listing any mandatory fields that are missing or empty.
    """
    brief_text = _sanitize(brief_text)

    if _is_tab_format(brief_text):
        result = _parse_tab_brief(brief_text)
    else:
        result = {}
        for field in ALL_FIELDS:
            raw = _extract_section(brief_text, field)
            if field in ("Content criteria", "Explicit exclusions"):
                result[field] = _parse_list(raw) if raw else []
            else:
                result[field] = raw or ""

    missing = [f for f in MANDATORY_FIELDS if not result.get(f)]
    if missing:
        raise ValueError(f"Missing or empty mandatory fields: {', '.join(missing)}")

    return result


def get_primary_keyword(brief_text: str) -> str:
    """Extract just the primary keyword from a brief without full validation."""
    brief_text = _sanitize(brief_text)
    if _is_tab_format(brief_text):
        # Use the tab parser and pull the keyword field
        result = _parse_tab_brief(brief_text)
        return result.get("Primary keyword", "").splitlines()[0].strip()
    raw = _extract_section(brief_text, "Primary keyword")
    if not raw:
        return ""
    return raw.strip().splitlines()[0].strip()


def format_criteria_list(criteria: list[str]) -> str:
    """Return a numbered string suitable for inclusion in a prompt."""
    return "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))


def format_exclusions_list(exclusions: list[str]) -> str:
    """Return a bulleted string suitable for inclusion in a prompt."""
    return "\n".join(f"- {e}" for e in exclusions)


# Substrings that identify template hint/instruction text to strip from student briefs.
_HINT_PHRASES = [
    # Universal instruction markers
    "copy from",
    "one action only",
    "not a demographic",
    "each must be checkable",
    "the exact keyword from",
    "what action do you want",
    "yes/no checklist",
    "checklist for",
    "each item below",
    # Field-specific hint openers
    "3 adjectives describing",          # Brand voice
    "2-3 consistent patterns",           # Competitive insights
    "2-3 specific phrases",              # Explicit exclusions
    "list sections in order",            # Page sections
    "where is this reader",              # Customer journey moment
    "2-3 gaps",                          # Competitive gaps
]


def _is_hint(text: str) -> bool:
    lo = text.lower()
    return any(phrase in lo for phrase in _HINT_PHRASES)


def preprocess_brief(parsed: dict) -> tuple[dict, dict]:
    """
    Remove template hint/instruction text from every field in a parsed brief.

    Returns (cleaned, removed) where:
      cleaned  — copy of parsed with hint lines/items stripped
      removed  — {field: [dropped strings]} for any field that lost content
    """
    cleaned: dict = {}
    removed: dict = {}

    for field, value in parsed.items():
        if isinstance(value, list):
            kept = [item for item in value if not _is_hint(item)]
            dropped = [item for item in value if _is_hint(item)]
            cleaned[field] = kept
            if dropped:
                removed[field] = dropped
        else:
            kept_lines = [ln for ln in value.splitlines() if not _is_hint(ln)]
            dropped_lines = [ln for ln in value.splitlines() if _is_hint(ln)]
            result_str = "\n".join(kept_lines).strip()
            # Brand is a single name — take only the first non-empty line after cleaning.
            if field == "Brand":
                result_str = next((ln.strip() for ln in kept_lines if ln.strip()), "")
            cleaned[field] = result_str
            if dropped_lines:
                removed[field] = dropped_lines

    return cleaned, removed
