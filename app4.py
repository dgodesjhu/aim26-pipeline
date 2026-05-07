"""
app.py — AIM26 Agentic Content Pipeline
Johns Hopkins AI in Marketing, Spring 2026
"""

import base64
import time
from datetime import datetime

import anthropic
import streamlit as st

import pipeline
from brief_parser import parse_brief, preprocess_brief
from pipeline import (
    parse_change_log_rows,
    parse_evaluation_rows,
    parse_evaluator_note,
    run_change_log,
    run_competitive_analysis,
    run_evaluation,
    run_html_generation,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AIM26 Content Pipeline",
    page_icon="🧠",
    layout="wide",
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def render_html(html_string: str, height: int = 600) -> None:
    try:
        encoded = base64.b64encode(html_string.encode()).decode()
        src = f"data:text/html;base64,{encoded}"
        st.components.v1.iframe(src, height=height, scrolling=True)
    except Exception:
        st.warning("Could not render iframe. Download the file and open in your browser.")


def _safe_filename(brand: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", brand.lower()).strip("-")


def _elapsed(start: float) -> str:
    secs = int(time.time() - start)
    m, s = divmod(secs, 60)
    return f"{m}m {s}s" if m else f"{s}s"


def _append_log(log_placeholder, log_lines: list[str], new_line: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    log_lines.append(f"`{ts}` — {new_line}")
    log_placeholder.markdown("\n\n".join(log_lines))


def _update_sidebar_stats() -> None:
    ss = st.session_state
    sidebar_stats_placeholder = ss.get("sidebar_stats_placeholder")
    if sidebar_stats_placeholder is not None:
        sidebar_stats_placeholder.markdown(
            f"**Tokens in:** {ss.get('total_input_tokens', 0):,}  \n"
            f"**Tokens out:** {ss.get('total_output_tokens', 0):,}  \n"
            f"**API calls:** {ss.get('total_api_calls', 0)}"
        )


def _init_state() -> None:
    defaults = {
        "v1_html": None,
        "final_html": None,
        "all_evaluations": [],
        "all_scores": [],
        "all_html": [],
        "iteration_count": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_api_calls": 0,
        "pipeline_start_time": None,
        "parsed_brief": None,
    }
    for k, v in defaults.items():
        st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────────

import re

with st.sidebar:
    st.title("Pipeline settings")
    threshold_pct = st.slider(
        "Pass threshold (% of maximum score)",
        min_value=50,
        max_value=100,
        value=75,
        step=5,
        format="%d%%",
    )
    max_iterations = st.number_input(
        "Maximum iterations",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )
    st.caption(
        "These parameters are design choices — they determine how much "
        "you trust the agent's self-evaluation."
    )
    st.divider()
    st.subheader("Token usage")
    sidebar_stats_placeholder = st.empty()
    st.session_state["sidebar_stats_placeholder"] = sidebar_stats_placeholder
    _update_sidebar_stats()

# ── Main UI ────────────────────────────────────────────────────────────────────

st.title("AIM26 Agentic Content Pipeline")
st.caption("JHU AI in Marketing · Spring 2026")

col_brief, col_context = st.columns([1, 1])

with col_brief:
    st.markdown(
        "Paste your completed content brief below. "
        "Fill in all fields from `## Brand` through `## Page sections` using the brief template "
        "you received for this exercise."
    )
    brief_text = st.text_area(
        "Content brief",
        placeholder=(
            "Paste your completed brief here — all fields from ## Brand through ## Page sections"
        ),
        height=400,
        label_visibility="collapsed",
    )

with col_context:
    st.markdown(
        "Paste the competitive context document below. "
        "This document will be shared by your instructor before the exercise begins — "
        "do not create or modify it yourself."
    )
    competitive_context = st.text_area(
        "Competitive context document",
        placeholder=(
            "Paste the competitive context document here — "
            "competitor page extracts with annotations"
        ),
        height=400,
        label_visibility="collapsed",
    )

run_button = st.button("Run pipeline", type="primary", use_container_width=True)

# ── Pipeline execution ─────────────────────────────────────────────────────────

if run_button:
    if not brief_text.strip():
        st.error("Please paste a content brief before running the pipeline.")
        st.stop()
    if not competitive_context.strip():
        st.error("Please paste a competitive context document before running the pipeline.")
        st.stop()

    _init_state()
    ss = st.session_state
    ss["pipeline_start_time"] = time.time()

    stats = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_api_calls": 0,
    }

    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_placeholder = st.empty()
    log_lines: list[str] = []
    retry_warning = st.empty()

    def _on_api_retry(attempt: int) -> None:
        retry_warning.warning(
            f"API overloaded — retrying in 10 seconds (attempt {attempt} of 3)."
        )
        _append_log(log_placeholder, log_lines, f"API overloaded — retry {attempt}/3")

    pipeline.set_retry_callback(_on_api_retry)

    # ── Step 1: Parse brief ────────────────────────────────────────────────────

    status_text.info("Step 1 — Parsing brief...")
    try:
        parsed_raw = parse_brief(brief_text)
    except ValueError as exc:
        st.error(f"**Brief parsing failed:** {exc}")
        st.stop()

    parsed, removed_hints = preprocess_brief(parsed_raw)
    ss["parsed_brief"] = parsed

    n_criteria = len(parsed["Content criteria"])
    n_exclusions = len(parsed["Explicit exclusions"])
    n_voice = 1 if parsed.get("Brand voice", "").strip() else 0
    n_exclusions_criterion = 1 if n_exclusions else 0
    n_total_criteria = n_criteria + n_voice + n_exclusions_criterion
    max_score = n_total_criteria * 3
    threshold = round(threshold_pct / 100 * max_score)

    st.subheader("Step 1 — Brief parsed")
    table_rows = []
    for field, value in parsed.items():
        if isinstance(value, list):
            preview = "; ".join(value)[:120]
        else:
            preview = str(value)[:120]
        dropped = removed_hints.get(field, [])
        removed_preview = "; ".join(dropped)[:80] if dropped else ""
        table_rows.append({
            "Field": field,
            "Cleaned value": preview,
            "Hint text removed": removed_preview,
        })
    st.table(table_rows)

    if removed_hints:
        total_removed = sum(len(v) for v in removed_hints.values())
        st.caption(
            f"Preprocessing removed {total_removed} hint/instruction line(s) from "
            f"{len(removed_hints)} field(s) before scoring."
        )


    progress_bar.progress(10)
    status_text.success(
        f"Step 1 complete — {n_total_criteria} criteria total "
        f"({n_criteria} content, {n_voice} brand voice, {n_exclusions_criterion} exclusions); "
        f"pass threshold: {threshold}/{max_score} ({threshold_pct}%)"
    )
    _append_log(
        log_placeholder, log_lines,
        f"Brief parsed — {n_total_criteria} criteria, threshold {threshold}/{max_score}"
        + (f"; {sum(len(v) for v in removed_hints.values())} hint lines removed" if removed_hints else "")
    )

    # ── Step 2: Competitive analysis ───────────────────────────────────────────

    status_text.info("Step 2 — Analyzing competitive context...")
    try:
        competitive_summary = run_competitive_analysis(client, competitive_context, stats)
    except Exception as exc:
        st.error(f"**Step 2 failed:** {exc}")
        st.stop()

    ss["total_input_tokens"] = stats["total_input_tokens"]
    ss["total_output_tokens"] = stats["total_output_tokens"]
    ss["total_api_calls"] = stats["total_api_calls"]
    _update_sidebar_stats()

    st.subheader("Step 2 — Competitive analysis")
    with st.expander("Competitive context sent to Claude (raw input)", expanded=False):
        st.text(competitive_context)
    with st.expander("Competitive analysis output", expanded=False):
        st.text(competitive_summary)

    progress_bar.progress(25)
    status_text.success("Step 2 complete — Competitive analysis complete")
    _append_log(log_placeholder, log_lines, "Competitive analysis complete")

    # ── Steps 3–5: Generate → Evaluate → Decide loop ──────────────────────────

    iteration = 0
    all_evaluations: list[dict] = []
    all_scores: list[int | None] = []
    all_html: list[str] = []
    accumulated_feedback: list[str] = []
    best_html = None
    best_score: int | None = None

    while iteration < max_iterations:
        iteration += 1
        ss["iteration_count"] = iteration

        # ── Step 3: Generate HTML ──────────────────────────────────────────────

        status_text.info(f"Step 3 — Generating HTML (attempt {iteration} of {max_iterations})...")
        progress_pct = int(25 + (iteration / max_iterations) * 30)

        try:
            html_output, token_count = run_html_generation(
                client,
                brief_text,
                parsed["Brand"],
                competitive_summary,
                stats,
                previous_html=best_html if iteration > 1 else None,
                latest_feedback=accumulated_feedback[-1] if accumulated_feedback else None,
            )
        except Exception as exc:
            st.error(f"**Step 3 (attempt {iteration}) failed:** {exc}")
            st.stop()

        ss["total_input_tokens"] = stats["total_input_tokens"]
        ss["total_output_tokens"] = stats["total_output_tokens"]
        ss["total_api_calls"] = stats["total_api_calls"]
        _update_sidebar_stats()

        all_html.append(html_output)
        if iteration == 1:
            ss["v1_html"] = html_output

        if not html_output.strip().lower().endswith("</html>"):
            st.warning("HTML appears truncated even after multiple continuation attempts. Proceeding with partial output.")

        progress_bar.progress(progress_pct)
        status_text.success(f"Step 3 complete — HTML generated — {token_count} tokens (attempt {iteration})")

        st.subheader(f"Step 3 — Generated HTML (attempt {iteration})")
        v_label = f"v{iteration}_html"
        st.download_button(
            label=f"Download {v_label}.html",
            data=html_output,
            file_name=f"{v_label}.html",
            mime="text/html",
            key=f"dl_{v_label}",
        )
        with st.expander(f"Preview — attempt {iteration}", expanded=(iteration == 1)):
            render_html(html_output, height=500)

        _append_log(log_placeholder, log_lines, f"HTML generated — attempt {iteration}, {token_count} tokens")

        # ── Step 4: Evaluate (3 evaluators + synthesis) ───────────────────────

        st.subheader(f"Step 4 — Evaluation (attempt {iteration})")
        eval_status = st.empty()

        def _on_eval_progress(msg: str, _es=eval_status) -> None:
            _es.info(f"Step 4 — {msg}")

        # Build evaluation criteria: student's criteria + implicit brand voice and exclusions
        voice_spec = parsed.get("Brand voice", "").strip()
        exclusions = parsed.get("Explicit exclusions", [])
        eval_criteria = list(parsed["Content criteria"])
        if voice_spec:
            eval_criteria.append(
                f"Brand voice — all copy must match the tone, vocabulary, and style of these "
                f"brand voice examples: {voice_spec}"
            )
        if exclusions:
            eval_criteria.append(
                f"Explicit exclusions — the page must not contain any of the following, "
                f"even subtly or paraphrased: {'; '.join(exclusions)}"
            )

        with st.expander("Evaluation criteria sent to evaluators", expanded=False):
            for i, c in enumerate(eval_criteria, 1):
                st.markdown(f"**{i}.** {c}")

        try:
            eval_result = run_evaluation(
                client,
                html_output,
                eval_criteria,
                stats,
                on_progress=_on_eval_progress,
            )
            score = eval_result["total_score"]
        except Exception as exc:
            st.error(f"**Step 4 failed:** {exc}")
            eval_result = {"evaluations": [], "synthesis": "", "total_score": None, "max_score": max_score}
            score = None

        ss["total_input_tokens"] = stats["total_input_tokens"]
        ss["total_output_tokens"] = stats["total_output_tokens"]
        ss["total_api_calls"] = stats["total_api_calls"]
        _update_sidebar_stats()

        all_evaluations.append(eval_result)
        all_scores.append(score)

        score_display = f"{score}/{max_score}" if score is not None else f"?/{max_score}"
        progress_bar.progress(70)
        eval_status.success(f"Step 4 complete — score: {score_display}")

        # Evaluation result
        eval_text = eval_result["evaluations"][0][1] if eval_result.get("evaluations") else ""
        eval_rows = parse_evaluation_rows(eval_text)
        if eval_rows:
            st.table(eval_rows)
        else:
            st.text(eval_text)
        note = parse_evaluator_note(eval_text)
        if note:
            st.markdown(
                f'<div style="background:#f0f4f8;padding:12px 16px;border-radius:6px;'
                f'font-style:italic;color:#444;margin-top:8px;">'
                f'<strong>Evaluator note:</strong> {note}</div>',
                unsafe_allow_html=True,
            )
        with st.expander("Debug — raw evaluation text", expanded=False):
            st.text(eval_text)
        if score is None:
            st.warning(
                "Could not parse total score. "
                "Skipping threshold check and proceeding to output."
            )

        _append_log(log_placeholder, log_lines, f"Evaluation complete — score: {score_display}")

        # ── Step 5: Decide ─────────────────────────────────────────────────────

        prior_best = best_score
        if best_score is None or (score is not None and score > best_score):
            best_html = html_output
            best_score = score

        if score is None:
            break

        progress_bar.progress(75)

        if score >= threshold:
            status_text.success(
                f"Step 5 — Score {score}/{max_score} meets threshold {threshold}/{max_score}. Proceeding to output."
            )
            _append_log(log_placeholder, log_lines, f"Threshold met ({score}/{max_score} ≥ {threshold}/{max_score})")
            break
        elif iteration > 1 and prior_best is not None and score < prior_best:
            st.warning(
                f"Score {score}/{max_score} is lower than previous best {prior_best}/{max_score}. "
                f"Stopping — using best-scoring attempt."
            )
            _append_log(log_placeholder, log_lines, f"Score regressed ({score} < {prior_best}) — stopping early")
            break
        elif iteration >= max_iterations:
            st.warning(
                f"Maximum iterations reached. Using best-scoring attempt "
                f"(score: {best_score}/{max_score})."
            )
            _append_log(log_placeholder, log_lines, f"Max iterations reached — best score: {best_score}/{max_score}")
            break
        else:
            st.info(
                f"Score {score}/{max_score} is below threshold {threshold}/{max_score}. "
                f"Editing with targeted feedback..."
            )
            _append_log(log_placeholder, log_lines, f"Score {score}/{max_score} below threshold — editing")
            accumulated_feedback.append(
                f"ATTEMPT {iteration} FEEDBACK (score: {score}/{max_score}):\n{eval_text}"
            )

    # Save final state
    ss["final_html"] = best_html
    ss["all_evaluations"] = all_evaluations
    ss["all_scores"] = all_scores
    ss["all_html"] = all_html

    ss["total_input_tokens"] = stats["total_input_tokens"]
    ss["total_output_tokens"] = stats["total_output_tokens"]
    ss["total_api_calls"] = stats["total_api_calls"]
    _update_sidebar_stats()

    # ── Step 6: Output ─────────────────────────────────────────────────────────

    st.divider()
    st.header("Step 6 — Final output")

    tab_labels = ["Final page", "Evaluation scores", "Pipeline stats"]
    show_change_log = iteration > 1 and ss["v1_html"] and best_html != ss["v1_html"]
    if show_change_log:
        tab_labels.insert(2, "Agent change log")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # Tab 1 — Final page
    with tabs[tab_idx]:
        tab_idx += 1
        brand_slug = _safe_filename(parsed["Brand"])
        st.download_button(
            label=f"Download final HTML — {brand_slug}-pipeline-final.html",
            data=best_html,
            file_name=f"{brand_slug}-pipeline-final.html",
            mime="text/html",
        )
        if iteration > 1 and ss["v1_html"]:
            st.download_button(
                label="Download v1 HTML (first attempt)",
                data=ss["v1_html"],
                file_name=f"{brand_slug}-v1.html",
                mime="text/html",
            )
        render_html(best_html, height=650)

    # Tab 2 — Evaluation scores
    with tabs[tab_idx]:
        tab_idx += 1

        # Summary table across all attempts
        if iteration > 1:
            st.subheader("Iteration summary")
            iter_rows = []
            for i, s in enumerate(all_scores):
                score_str = f"{s}/{max_score}" if s is not None else f"?/{max_score}"
                passed = "Yes" if (s is not None and s >= threshold) else "No"
                iter_rows.append({"Attempt": i + 1, "Score": score_str, "Threshold passed": passed})
            st.table(iter_rows)

        # Per-attempt detail
        for i, (er, s) in enumerate(zip(all_evaluations, all_scores)):
            score_str = f"{s}/{max_score}" if s is not None else f"?/{max_score}"
            passed = "Yes" if (s is not None and s >= threshold) else "No"
            is_last = i == len(all_evaluations) - 1
            with st.expander(
                f"Attempt {i + 1} — Score: {score_str} — Threshold passed: {passed}",
                expanded=is_last,
            ):
                # Per-criterion scores
                tab_eval_text = er["evaluations"][0][1] if er.get("evaluations") else ""
                tab_rows = parse_evaluation_rows(tab_eval_text)
                if tab_rows:
                    st.table(tab_rows)
                else:
                    st.text(tab_eval_text)
                tab_note = parse_evaluator_note(tab_eval_text)
                if tab_note:
                    st.markdown(
                        f'<div style="background:#f0f4f8;padding:12px 16px;border-radius:6px;'
                        f'font-style:italic;color:#444;margin-top:8px;">'
                        f'<strong>Evaluator note:</strong> {tab_note}</div>',
                        unsafe_allow_html=True,
                    )

    # Tab 3 — Agent change log (conditional)
    if show_change_log:
        with tabs[tab_idx]:
            tab_idx += 1
            st.subheader("Agent change log")
            with st.spinner("Comparing v1 and final HTML..."):
                try:
                    change_log_text = run_change_log(
                        client,
                        ss["v1_html"],
                        best_html,
                        stats,
                    )
                    ss["total_input_tokens"] = stats["total_input_tokens"]
                    ss["total_output_tokens"] = stats["total_output_tokens"]
                    ss["total_api_calls"] = stats["total_api_calls"]
                    _update_sidebar_stats()

                    change_rows = parse_change_log_rows(change_log_text)
                    if change_rows:
                        st.table(change_rows)
                    else:
                        st.text(change_log_text)
                except Exception as exc:
                    st.error(f"Change log failed: {exc}")

    # Tab — Pipeline stats
    with tabs[tab_idx]:
        elapsed = _elapsed(ss["pipeline_start_time"])
        total_tokens = ss["total_input_tokens"] + ss["total_output_tokens"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total tokens", f"{total_tokens:,}")
            st.metric("Input tokens", f"{ss['total_input_tokens']:,}")
            st.metric("Output tokens", f"{ss['total_output_tokens']:,}")
        with col2:
            st.metric("API calls", ss["total_api_calls"])
            st.metric("Iterations", iteration)
            st.metric("Time elapsed", elapsed)

        st.divider()
        st.subheader("Design parameters")
        st.markdown(f"**Evaluation threshold:** {threshold}/{max_score} ({threshold_pct}%)")
        st.markdown(f"**Maximum iterations:** {max_iterations}")
        st.info(
            "These are design choices that determine where human judgment sits in this "
            "workflow — not technical parameters."
        )

    progress_bar.progress(100)
    status_text.success("Pipeline complete.")
    _append_log(log_placeholder, log_lines, f"Pipeline complete — {elapsed} elapsed")
