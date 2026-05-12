from confpaper.sources.base import PaperSource
from confpaper.sources.cvf import CVFSource
from confpaper.sources.arxiv_source import ArxivSource
from confpaper.sources.openreview_source import OpenReviewSource
from confpaper.sources.aaai_source import AAAISource
from confpaper.utils import canonicalize_venue

CVF_VENUES = {"CVPR", "ICCV", "WACV", "ECCV"}
OPENREVIEW_VENUES = {"ICLR", "NeurIPS", "ICML"}
AAAI_VENUES = {"AAAI"}


def route_search(
    venue: str | None = None,
    year: int | None = None,
    source: str = "auto",
) -> list[PaperSource]:
    if source == "cvf":
        return [CVFSource()]
    if source == "openreview":
        return [OpenReviewSource()]
    if source == "arxiv":
        return [ArxivSource()]
    # Public version: general is arXiv-only (no Semantic Scholar).
    if source == "general":
        return [ArxivSource()]

    if source == "aaai":
        return [AAAISource()]

    if source == "auto":
        if venue and year:
            venue_canonical = canonicalize_venue(venue)
            if venue_canonical and venue_canonical in CVF_VENUES:
                return [CVFSource()]
            if venue_canonical and venue_canonical in AAAI_VENUES:
                return [AAAISource()]
            if venue_canonical and venue_canonical in OPENREVIEW_VENUES:
                return [OpenReviewSource()]
            return [ArxivSource()]
        return [ArxivSource()]

    if source in ("semantic_scholar", "semantic", "ss"):
        raise ValueError(
            "Semantic Scholar is not included in the public version. "
            "Use --source arxiv instead."
        )

    raise ValueError(f"Unknown source: {source}")
