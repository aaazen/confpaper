import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urlencode

import httpx

from confpaper.models import Paper
from confpaper.utils import build_paper_id, canonicalize_venue

logger = logging.getLogger(__name__)
ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "confpaper/0.2 (+https://github.com/confpaper)"

# Global rate limiter for arXiv API (1 req / 5s to be safe)
_last_arxiv_request = 0.0


async def _arxiv_rate_limit():
    global _last_arxiv_request
    now = asyncio.get_event_loop().time()
    wait = _last_arxiv_request + 5.0 - now
    if wait > 0:
        await asyncio.sleep(wait)
    _last_arxiv_request = asyncio.get_event_loop().time()


def _extract_venue_from_text(text: str) -> tuple[str | None, int | None]:
    """Extract venue and year from arXiv comment/journal_ref."""
    if not text:
        return None, None
    # Phase 1: regex for "Accepted at XXX YYYY" and "XXX YYYY" patterns
    patterns = [
        r"Accepted\s+(?:at|to|in|by)\s+(CVPR|ICCV|ECCV|WACV|NeurIPS|NIPS|ICLR|ICML|AAAI|ACL|EMNLP|ACM\s*MM)\s*'?(\d{2,4})?",
        r"(CVPR|ICCV|ECCV|WACV|NeurIPS|NIPS|ICLR|ICML|AAAI)\s*'?(\d{2,4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            venue = canonicalize_venue(m.group(1))
            year = None
            if m.lastindex and m.lastindex >= 2 and m.group(2):
                y = int(m.group(2))
                if y < 100:
                    y += 2000
                year = y
            return venue, year
    return _extract_venue_by_alias(text)


def _extract_venue_by_alias(text: str) -> tuple[str | None, int | None]:
    """Fallback: match known venue aliases against journal_ref text."""
    from confpaper.utils import _load_venues, _init_venue_maps
    _init_venue_maps()
    data = _load_venues()
    venues = data.get("venues", {})

    # Check each venue's aliases against the text
    for canonical, info in venues.items():
        aliases = info.get("aliases", [])
        for alias in aliases:
            if len(alias) < 3:
                continue
            idx = text.lower().find(alias.lower())
            if idx >= 0:
                # Try to find a year near the alias
                nearby = text[idx:idx + len(alias) + 50]
                year_match = re.search(r"(\d{4})", nearby)
                year = int(year_match.group(1)) if year_match else None
                return canonical, year

    return None, None


def _parse_arxiv_feed(xml_text: str) -> list[dict]:
    """Parse arXiv Atom feed and return list of paper dicts."""
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        title = " ".join(title_el.text.split()) if title_el is not None and title_el.text else ""

        authors = []
        for author_el in entry.findall("atom:author", ns):
            name_el = author_el.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        summary_el = entry.find("atom:summary", ns)
        abstract = summary_el.text.strip() if summary_el is not None and summary_el.text else ""

        published_el = entry.find("atom:published", ns)
        pub_year = None
        if published_el is not None and published_el.text:
            try:
                pub_year = datetime.fromisoformat(published_el.text.replace("Z", "+00:00")).year
            except (ValueError, TypeError):
                pass

        # Extract IDs
        arxiv_id = None
        id_el = entry.find("atom:id", ns)
        if id_el is not None and id_el.text:
            # http://arxiv.org/abs/2401.17270v3 → 2401.17270v3
            arxiv_id = id_el.text.strip().split("/")[-1]

        doi = None
        for link_el in entry.findall("atom:link", ns):
            href = link_el.get("href", "")
            title_attr = link_el.get("title", "")
            if "doi" in title_attr.lower() or "doi.org" in href:
                doi = href.strip()

        # PDF URL
        pdf_url = None
        for link_el in entry.findall("atom:link", ns):
            title_attr = link_el.get("title", "")
            if title_attr == "pdf":
                pdf_url = link_el.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pure_id = arxiv_id.split("v")[0]
            pdf_url = f"https://arxiv.org/pdf/{pure_id}.pdf"

        # Comment and journal_ref for venue detection
        comment_el = entry.find("arxiv:comment", ns)
        comment = " ".join(comment_el.text.split()) if comment_el is not None and comment_el.text else ""

        journal_el = entry.find("arxiv:journal_ref", ns)
        journal_ref = " ".join(journal_el.text.split()) if journal_el is not None and journal_el.text else ""

        papers.append({
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "year": pub_year,
            "arxiv_id": arxiv_id,
            "doi": doi,
            "pdf_url": pdf_url,
            "comment": comment,
            "journal_ref": journal_ref,
        })
    return papers


class ArxivSource:
    name = "arxiv"

    async def search(
        self,
        query: str,
        venue: str | None = None,
        year: int | None = None,
        max_results: int = 20,
    ) -> list[Paper]:
        # Convert confpaper AND syntax ("+") to arXiv-compatible query.
        # Do NOT inject venue or year as search keywords — those are handled
        # by source routing and post-filtering respectively.
        arxiv_query = " AND ".join(
            part.strip() for part in query.split("+") if part.strip()
        )

        # Fetch a bit more to allow for post-filtering
        fetch_count = min(max_results * 3, 100)

        papers: list[Paper] = []
        for attempt in range(3):
            try:
                await _arxiv_rate_limit()
                papers = await self._do_search(arxiv_query, fetch_count, venue, year)
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 2:
                    wait = (2 ** attempt) * 30  # 30s, 60s backoff
                    logger.warning("arXiv rate limited, waiting %ds...", wait)
                    await asyncio.sleep(wait)
                    continue
                raise
            except (httpx.RequestError, ET.ParseError) as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        return papers[:max_results]

    async def _do_search(
        self, query: str, count: int, venue: str | None, year: int | None
    ) -> list[Paper]:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": count,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API}?{urlencode(params)}"

        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=30,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            raw_papers = await asyncio.to_thread(_parse_arxiv_feed, response.text)

        papers = []
        for r in raw_papers:
            # Extract venue from comment/journal_ref
            combined = f"{r.get('journal_ref', '')} {r.get('comment', '')}"
            extracted_venue, extracted_year = _extract_venue_from_text(combined)

            # paper.year comes from data only: extracted (comment) > published date
            paper_year = extracted_year or r.get("year")

            paper = Paper(
                title=r["title"],
                authors=r["authors"],
                year=paper_year,
                venue=extracted_venue,
                venue_source="arxiv_comment" if extracted_venue else None,
                abstract=r["abstract"],
                source="arxiv",
                pdf_url=r["pdf_url"],
                arxiv_id=r["arxiv_id"],
                doi=r["doi"],
            )
            paper.id = build_paper_id(paper)
            papers.append(paper)

        # Post-filter by year: paper must have a year within ±1 of target
        if year:
            papers = [p for p in papers if p.year and abs(p.year - year) <= 1]

        # Post-filter by venue: only keep papers where venue was confirmed
        if venue:
            venue_canon = canonicalize_venue(venue)
            if venue_canon:
                papers = [p for p in papers if p.venue == venue_canon]

        return papers
