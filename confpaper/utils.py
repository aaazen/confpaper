import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from confpaper.models import Paper

_VENUE_ALIASES: dict[str, str] = {}
_VENUE_CVF_CODES: dict[str, str] = {}


def _load_venues() -> dict:
    venues_path = Path(__file__).parent.parent / "data" / "venues.yaml"
    if venues_path.exists():
        with open(venues_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _init_venue_maps():
    global _VENUE_ALIASES, _VENUE_CVF_CODES
    if _VENUE_ALIASES:
        return
    data = _load_venues()
    venues = data.get("venues", {})
    for canonical, info in venues.items():
        canonical_upper = canonical.upper()
        _VENUE_ALIASES[canonical.lower()] = canonical
        for alias in info.get("aliases", []):
            _VENUE_ALIASES[alias.lower().strip()] = canonical
        cvf_code = info.get("cvf_code")
        if cvf_code:
            _VENUE_CVF_CODES[canonical] = cvf_code


def canonicalize_venue(venue: str | None) -> str | None:
    if not venue:
        return None
    _init_venue_maps()
    return _VENUE_ALIASES.get(venue.lower().strip(), venue)


def get_cvf_code(venue: str) -> str | None:
    _init_venue_maps()
    return _VENUE_CVF_CODES.get(venue)


def normalize_title(title: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def clean_title(title: str, max_len: int = 80) -> str:
    """Sanitize title for use as a filename."""
    if not title:
        return "untitled"
    cleaned = title.strip()
    cleaned = re.sub(r'[\\/:*?"<>|]', "", cleaned)
    cleaned = cleaned.replace(" ", "_")
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    cleaned = cleaned[:max_len]
    cleaned = cleaned.strip("_")
    return cleaned or "untitled"


def build_paper_id(paper: "Paper") -> str:
    """Build a stable paper ID: arxiv: > doi: > hash."""
    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id}"
    if paper.doi:
        return f"doi:{paper.doi}"
    normalized = normalize_title(paper.title)
    hash_hex = hashlib.sha1(normalized.encode()).hexdigest()[:8]
    venue = paper.venue or "unknown"
    year = str(paper.year) if paper.year else "unknown"
    source = paper.source or "unknown"
    return f"{source}:{venue}:{year}:{hash_hex}"


def is_cvf_venue(venue: str) -> bool:
    canonical = canonicalize_venue(venue)
    if not canonical:
        return False
    return get_cvf_code(canonical) is not None


def is_valid_venue_year(venue: str, year: int) -> bool:
    """Return True if the venue holds a conference in the given year.

    ICCV: odd years (2023, 2025, …)
    ECCV: even years (2022, 2024, …)
    All other venues: every year.
    """
    canonical = canonicalize_venue(venue)
    if canonical == "ICCV":
        return year % 2 == 1
    if canonical == "ECCV":
        return year % 2 == 0
    return True


def is_openreview_venue(venue: str) -> bool:
    _init_venue_maps()
    data = _load_venues()
    venues = data.get("venues", {})
    canonical = canonicalize_venue(venue)
    if not canonical:
        return False
    info = venues.get(canonical, {})
    return info.get("openreview_id") is not None
