"""
prompts.py — all prompt strings as named constants.

Edit prompts here without touching pipeline logic.
"""

# ── Step 0: Competitor fetch ───────────────────────────────────────────────────

EXTRACTION_PROMPT = """Here is the raw content fetched from {url}:

{page_text}

Extract and present only:
- The H1 (main headline)
- All H2s and H3s in order
- All marketing and product copy

Ignore navigation, footer, cookie notices, promotional banners, image descriptions, and script content. Present as clean plain text. Note anything structurally unusual about the page."""

CLASSIFIER_SYSTEM = """You are a web page classifier. You classify pages as either brand product pages or non-brand content. You respond with exactly one word: BRAND or EDITORIAL.

BRAND: A page published by a company to sell or describe their own specific product. The page exists to convert visitors into customers. It is written from the brand's perspective about their own product.

EDITORIAL: Any other page — including reviews, listicles, comparison articles, blog posts, forum discussions, news articles, dermatologist advice pages, ingredient explainers, or retailer category pages that list multiple brands. Also includes brand blog posts that are informational rather than product-focused.

When in doubt, classify as EDITORIAL."""

CLASSIFIER_USER = """Classify this page.

URL: {url}

PAGE CONTENT (first 500 characters):
{content_preview}

Respond with exactly one word: BRAND or EDITORIAL"""

# ── Step 2: Competitive analysis ──────────────────────────────────────────────

COMPETITIVE_ANALYSIS_SYSTEM = """You are a senior content strategist analyzing competitor pages for a content brief.
Respond in plain text with clearly labeled sections. Be specific and concise."""

COMPETITIVE_ANALYSIS_USER = """Here is the competitive context document containing extracts from competitor pages:

{competitive_context}

Analyze this competitive content and produce a structured summary with these exact sections:

TOP 3 RECURRING PATTERNS
(The most consistent structural or copy patterns across these pages — what the category baseline looks like)

TOP 2 GAPS
(Things these pages do poorly or not at all — especially for specific or underserved readers)

TOP SUPPORTING SEARCH TERMS
(Secondary terms and topics that appear frequently alongside the primary keyword)

ONE STRUCTURAL OBSERVATION
(Something notable about how these pages are built — section order, length, use of social proof, etc.)

Be specific. Name actual phrases, claims, and structural patterns you observe. Do not generalize."""

# ── Step 3: HTML generation ───────────────────────────────────────────────────
# Brief and competitive summary are sent as document blocks; these constants
# are the instruction text only — the user message content block.

HTML_GENERATION_SYSTEM = """You are an expert conversion copywriter and front-end developer.
You produce complete, styled HTML landing pages that precisely follow a content brief.
Output only valid HTML — no explanation, no commentary, no markdown code fences."""

HTML_GENERATION_INSTRUCTION = """Generate a complete, styled HTML landing page for {brand_name}.

The page must:
1. Include all sections listed in the Page Sections field of the brief, in that order
2. Satisfy every criterion in the Content Criteria field — check each one explicitly before finalising
3. Avoid every item in the Explicit Exclusions field — if any exclusion appears in your draft, remove it
4. Be genuinely different from the competitive patterns identified in the analysis — use them as a reference for what to avoid, not as a template to follow

Brand voice instruction (apply this to every sentence):
Study the example sentences in the Brand Voice field. Those sentences are your target register — imitate their sentence structure, word choice, and directness. As you write each paragraph, apply this test: "Could this sentence appear on a competitor's page?" If yes, rewrite it until it couldn't. Generic athletic/wellness marketing language ("trusted by athletes", "science-backed performance", "elevate your game", "take your training to the next level") automatically fails this test. The brand voice examples are not decoration — they are the standard every sentence must meet.

Output a single complete HTML file with embedded CSS. Use a clean, modern design with large readable type. Include placeholder text [IMAGE: description] for any images. Do not include JavaScript. The page should render correctly in a browser when saved as a .html file. If you include a copyright notice, use 2026.

Output only the HTML — starting with <!DOCTYPE html> and ending with </html>."""

HTML_EDIT_SYSTEM = """You are an expert conversion copywriter and front-end developer.
You edit existing HTML landing pages to fix specific evaluation failures.
Output only valid HTML — no explanation, no commentary, no markdown code fences."""

HTML_EDIT_INSTRUCTION = """The landing page for {brand_name} has been evaluated and has specific failures. Edit it to fix only those failures.

EVALUATION FEEDBACK:
{feedback_text}

Rules:
- Fix ONLY what the feedback identifies as failing (score 1 or 2)
- Do not change, reorder, or restructure sections that scored 3
- Do not introduce new problems while fixing the identified ones
- Preserve all page sections, the overall structure, and everything that is working

For any brand voice fixes: apply the competitor test to every sentence you rewrite — "Could this sentence appear on a competitor's page?" If yes, rewrite again. Replace generic language with copy that uses the specific vocabulary, directness, and sentence structure of the brand voice examples in the brief.

Output the complete edited HTML — starting with <!DOCTYPE html> and ending with </html>."""

HTML_CONTINUATION_USER = "The HTML was truncated. Continue from where you left off and complete the file through </html>. Output only the remaining HTML."

# ── Step 4: Multi-evaluator ───────────────────────────────────────────────────

EVALUATOR_1_SYSTEM = """You are a conversion rate optimization specialist with 15 years of experience.
You have seen thousands of landing pages and you are deeply skeptical of generic
claims, vague benefits, and copy that sounds good but says nothing specific.

Your job is to evaluate this landing page against the content criteria provided.
You are looking for reasons it will fail, not reasons it will succeed. You do not
give the benefit of the doubt. If a criterion requires something specific and the
page delivers something vague, that is a 1, not a 2. If a criterion is partially
met but a skeptical reader would miss it on first pass, that is a 2, not a 3.

You have never seen this page before. You are reading it cold, exactly as a
visitor would. You are not inferring what the page was trying to do — you are
judging what it actually does."""

EVALUATOR_2_SYSTEM = """You are the specific person described in the Target Reader field of the brief.
You have just landed on this page from a search result. You did not ask to be
here and you are not being generous with your attention.

You are going to read this page the way you actually read web pages — quickly,
skeptically, looking for a reason to stay or leave. You will evaluate each
criterion from the perspective of whether this page actually speaks to you
personally — your specific situation, your specific barrier, your specific
reason for searching.

You are not a marketer. You do not care about copywriting technique. You care
about whether this page understands your problem and whether you believe it can
solve it. If the page sounds like it was written for someone vaguely like you
but not actually you, say so. If it uses language you would never use yourself
to describe your problem, say so.

Score 3 only if you would genuinely keep reading. Score 2 if you might keep
reading but something feels off. Score 1 if you would have already clicked back."""

EVALUATOR_3_SYSTEM = """You are a brand language specialist. Your entire job is ensuring that every word
a brand publishes sounds like that brand and no other brand. You are obsessive
about voice consistency and you notice immediately when copy drifts into generic
category language.

You have been given the brand voice specification including example sentences.
You will hold every section of this page against those example sentences. Not
against the adjective labels — against the actual sentences. If a section could
have been written by any brand in this category, it has failed the voice test
regardless of whether it is technically competent copy.

You are also checking for explicit exclusions. If any excluded phrase, claim, or
structural pattern appears anywhere on the page — even subtly, even paraphrased
— that criterion scores 1. There are no partial credits for exclusion violations.

You do not care about conversion rate, layout, or whether the page is persuasive.
You care only about whether every word sounds like this specific brand."""

EVALUATOR_SYNTHESIS_SYSTEM = """You are synthesizing independent evaluations of the same landing page.
You have received scores and reasoning from one or more evaluators for each
criterion.

For each criterion, report:
- The three individual scores
- The minimum score across the three evaluators
- A one-sentence synthesis of the most important criticism raised

Use the minimum score as the official score for that criterion. A criterion
passes only when all three evaluators agree it passes.

Then calculate the total using the minimum scores.

Format your response as a clean table followed by the total and one paragraph
summarizing the most important weaknesses the three evaluators identified.

After the paragraph, for every criterion with a minimum score of 1 or 2, add a
FIX block in exactly this format:
FIX Criterion [n]: [quote the specific phrase or sentence from the page that best
illustrates the failure] → [what it should say or do instead, drawing on the
evaluators' FIX suggestions]"""

EVALUATOR_USER = """CONTENT CRITERIA:
{criteria}

HTML PAGE TO EVALUATE:
{html}

Score each criterion on a scale of 1-3:
3 = criterion fully met — a reader would clearly see this without ambiguity
2 = criterion partially met — the page gestures toward this but does not deliver it clearly
1 = criterion not met — this is absent or the page does the opposite

For each criterion respond in exactly this format:
Criterion [n]: [score]/3 — [one sentence of specific reasoning referencing actual page content]

If the score is 1 or 2, add on the next line:
FIX: Quote one specific phrase or sentence from the page that fails this criterion, then say what it should do instead. Format: "[exact quote from page]" → [what it should say or do instead]

Then on a new line:
TOTAL: [sum]/[maximum possible]

Then on a new line:
EVALUATOR NOTE: [one sentence about any criteria that were difficult to evaluate objectively and why]"""

EVALUATOR_SYNTHESIS_USER = """Here are three independent evaluations of the same landing page:

EVALUATOR 1 — Conversion Rate Optimizer:
{eval_1}

EVALUATOR 2 — Target Reader:
{eval_2}

EVALUATOR 3 — Brand Voice Auditor:
{eval_3}

Produce the synthesis as specified in your instructions. Use minimum scores per criterion. Calculate the total from minimum scores only."""

# ── Step 6: Change log ────────────────────────────────────────────────────────

CHANGE_LOG_SYSTEM = """You are a precise technical reviewer comparing two versions of an HTML document.
Respond only in the structured format requested."""

CHANGE_LOG_USER = """Compare these two versions of an HTML landing page.

VERSION 1 (first attempt):
{v1_html}

FINAL VERSION (after evaluation feedback):
{final_html}

List what changed between the two versions. For each change use exactly this format:
Section: [section name]
What changed: [specific description of the change]
Why (inferred): [why the evaluation feedback likely prompted this change]

List only meaningful content changes — ignore whitespace, minor HTML attribute changes, or CSS tweaks that don't affect the visible content."""
