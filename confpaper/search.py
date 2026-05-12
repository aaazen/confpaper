import re
from dataclasses import dataclass, field

from confpaper.matcher import score_paper_query, author_match_score, near_miss_score, is_and_query
from confpaper.models import Paper
from confpaper.router import route_search
from confpaper.utils import normalize_title


@dataclass
class SearchResult:
    papers: list[Paper]
    near_misses: list[Paper] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


async def search_papers(
    query: str,
    venue: str | None = None,
    year: int | None = None,
    source: str = "auto",
    max_results: int = 20,
    match_mode: str = "normal",
    match_threshold: float | None = None,
    author: str | None = None,
) -> list[Paper]:
    sources = route_search(venue=venue, year=year, source=source)
    all_papers: list[Paper] = []

    for src in sources:
        try:
            papers = await src.search(
                query=query,
                venue=venue,
                year=year,
                max_results=max_results,
            )
            all_papers.extend(papers)
        except Exception as e:
            import logging
            msg = str(e).strip()
            # Strip "Details:" lines that are empty or just whitespace
            msg = re.sub(r"\nDetails:\s*(\n|$)", r"\1", msg)
            # Strip trailing "Details:" with no content
            msg = re.sub(r"\nDetails:\s*$", "", msg)
            if msg:
                logger = logging.getLogger(__name__)
                logger.warning("%s: %s", src.name, msg)

    # Author search: score by author match only
    if author:
        for p in all_papers:
            s = author_match_score(author, p)
            p.match_score = s
            p.match_reason = "authors" if s >= 0.5 else "none"
            p.matched_query = author
        matched = [p for p in all_papers if p.match_score and p.match_score >= 0.5]
        merged = merge_papers(matched)
        merged.sort(key=_sort_key)
        return SearchResult(papers=merged[:max_results])

    # Normal relevance scoring via AND-aware multi-group matcher
    thresh = match_threshold or {"strict": 0.75, "normal": 0.58, "loose": 0.42}[match_mode]
    matched: list[Paper] = []
    near_misses: list[Paper] = []

    for p in all_papers:
        r = score_paper_query(query, p, mode=match_mode, threshold=match_threshold)
        if r.matched and r.score >= thresh:
            p.match_score = r.score
            p.match_reason = r.reason
            p.matched_query = query
            matched.append(p)
        elif is_and_query(query) and r.missing_groups:
            # Near miss: paper matched some groups but not all
            nm = near_miss_score(r)
            if nm > 0:
                p.match_score = nm
                p.match_reason = r.reason
                p.matched_query = query
                p.match_group_scores = r.group_scores
                p.match_group_reasons = r.group_reasons
                p.match_missing_groups = r.missing_groups
                near_misses.append(p)

    # Dedup and sort matched
    merged = merge_papers(matched)
    merged.sort(key=_sort_key)

    # Dedup and sort near misses
    near_misses = merge_papers(near_misses)
    near_misses.sort(key=_sort_key)

    return SearchResult(papers=merged[:max_results], near_misses=near_misses)


def _dedup_keys(paper: Paper) -> list[str]:
    keys = []
    if paper.arxiv_id:
        keys.append(f"arxiv:{paper.arxiv_id}")
    if paper.doi:
        keys.append(f"doi:{paper.doi}")
    keys.append(f"title:{normalize_title(paper.title)}")
    return keys


def merge_papers(papers: list[Paper]) -> list[Paper]:
    """Merge papers from multiple sources, deduplicating and enriching."""
    groups: dict[str, Paper] = {}
    key_order: list[str] = []

    for paper in papers:
        matched_key = None
        for key in _dedup_keys(paper):
            if key in groups:
                matched_key = key
                break

        if matched_key:
            _enrich(groups[matched_key], paper)
            # Keep the higher match score
            if (paper.match_score or 0) > (groups[matched_key].match_score or 0):
                groups[matched_key].match_score = paper.match_score
                groups[matched_key].match_reason = paper.match_reason
        else:
            for key in _dedup_keys(paper):
                if key not in groups:
                    groups[key] = paper
                    key_order.append(key)
                    break

    return [groups[k] for k in key_order]


def _enrich(existing: Paper, new: Paper):
    """Enrich existing paper with fields from new source."""
    if new.pdf_url and not existing.pdf_url:
        existing.pdf_url = new.pdf_url
        existing.source = new.source

    if new.arxiv_id and not existing.arxiv_id:
        existing.arxiv_id = new.arxiv_id
    if new.doi and not existing.doi:
        existing.doi = new.doi
    if not existing.authors and new.authors:
        existing.authors = new.authors


def _sort_key(paper: Paper):
    """Sort key: score desc → year desc → title asc.

    Omitting venue from the tie-breaker lets papers from different venues
    interleave naturally when they share the same score and year.
    """
    return (
        -(paper.match_score or 0.0),
        -(paper.year or 0),
        paper.title or "",
    )


def _quality_score(paper: Paper) -> float:
    score = 0.0
    if paper.venue and paper.venue_source != "unknown":
        score += 100.0
    score += min(paper.citation_count or 0, 10000) * 0.001
    return score
