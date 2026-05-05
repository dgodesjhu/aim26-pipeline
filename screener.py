"""
screener.py — URL-level editorial screening for Step 0 competitor fetch.
"""

EDITORIAL_DOMAINS = [
    # Major editorial/review sites
    "healthline.com",
    "byrdie.com",
    "allure.com",
    "wirecutter.com",
    "nytimes.com",
    "goodhousekeeping.com",
    "cosmopolitan.com",
    "vogue.com",
    "elle.com",
    "harpersbazaar.com",
    "instyle.com",
    "refinery29.com",
    "popsugar.com",
    "womenshealthmag.com",
    "menshealth.com",
    "self.com",
    "shape.com",
    "verywellhealth.com",
    "medicalnewstoday.com",
    "everydayhealth.com",
    "prevention.com",
    "realsimple.com",
    "bhg.com",
    # Aggregators and marketplaces
    "amazon.com",
    "sephora.com",
    "ulta.com",
    "target.com",
    "walmart.com",
    "cvs.com",
    "walgreens.com",
    "dermstore.com",
    "skinstore.com",
    # Community and forum
    "reddit.com",
    "quora.com",
    "mumsnet.com",
    "makeupalley.com",
    # General editorial
    "buzzfeed.com",
    "huffpost.com",
    "forbes.com",
    "businessinsider.com",
    "thecut.com",
    "nymag.com",
    "glamour.com",
    # Running/athlete specific editorial
    "runnersworld.com",
    "bicycling.com",
    "triathlete.com",
    "outsideonline.com",
    "active.com",
    "livestrong.com",
]

EDITORIAL_URL_PATTERNS = [
    "/best-",
    "/top-",
    "/review",
    "/reviews/",
    "/vs-",
    "/versus/",
    "/compare",
    "/comparison",
    "/ranked",
    "/ranking",
    "/roundup",
    "/buying-guide",
    "/what-is-",
    "/how-to-",
    "/guide-to-",
    "/everything-you-need",
    "/explained",
]


def is_editorial_by_url(url: str) -> tuple[bool, str | None]:
    from urllib.parse import urlparse
    parsed = urlparse(url.lower())
    domain = parsed.netloc.replace("www.", "")
    if any(domain == ed or domain.endswith("." + ed) for ed in EDITORIAL_DOMAINS):
        return True, "known editorial domain"
    path = parsed.path.lower()
    for pattern in EDITORIAL_URL_PATTERNS:
        if pattern in path:
            return True, f"URL pattern '{pattern}'"
    return False, None
