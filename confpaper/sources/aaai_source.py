import logging
import re
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from confpaper.models import Paper
from confpaper.utils import build_paper_id, canonicalize_venue

logger = logging.getLogger(__name__)
USER_AGENT = "Mozilla/5.0 (compatible; confpaper/0.2)"
OJS_BASE = "https://ojs.aaai.org/index.php/AAAI/"

# AAAI proceedings span multiple OJS issues per year (one per technical track).
# Issue ranges verified from https://ojs.aaai.org/index.php/AAAI/issue/archive
AAAI_ISSUE_RANGES = {
    2025: range(624, 637),  # Vol 39, Tracks 1-13
    2024: range(576, 583),  # Vol 38, Tracks 1-7
    2023: range(548, 561),  # Vol 37, Tracks 1-13
}


class AAAISource:
    name = "aaai"

    async def search(
        self,
        query: str,
        venue: str | None = None,
        year: int | None = None,
        max_results: int = 20,
    ) -> list[Paper]:
        venue_canonical = canonicalize_venue(venue) if venue else None
        if not venue_canonical or venue_canonical != "AAAI":
            raise ValueError("AAAI source requires --venue AAAI")
        if not year:
            raise ValueError("AAAI source requires --year")

        issue_ids = await self._find_issues(year)
        if not issue_ids:
            raise ValueError(
                f"No AAAI OJS issues found for year {year}. "
                f"Supported years: {sorted(AAAI_ISSUE_RANGES.keys())}"
            )

        all_papers: list[Paper] = []
        for iid in issue_ids:
            url = f"{OJS_BASE}issue/view/{iid}"
            try:
                html = await self._fetch(url)
                papers = self._parse_ojs(html, venue_canonical, year, url)
                all_papers.extend(papers)
            except RuntimeError:
                continue

        if not all_papers:
            raise RuntimeError(f"No papers parsed from AAAI {year} OJS pages.")

        return all_papers  # scoring done centrally in search.py

    async def _find_issues(self, year: int) -> list[int]:
        """Return all OJS issue IDs for the given AAAI year."""
        rng = AAAI_ISSUE_RANGES.get(year)
        return list(rng) if rng else []

    async def _fetch(self, url: str) -> str:
        for trust_env in (True, False):
            try:
                async with httpx.AsyncClient(
                    headers={"User-Agent": USER_AGENT}, timeout=30,
                    follow_redirects=True, trust_env=trust_env,
                ) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    return response.text
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
                if not trust_env:
                    raise
                continue
            except httpx.HTTPError as e:
                raise RuntimeError(f"Cannot reach {url}\nDetails: {e}")
        raise RuntimeError(f"Cannot reach {url}")

    def _parse_ojs(
        self, html: str, venue: str, year: int, base_url: str
    ) -> list[Paper]:
        soup = BeautifulSoup(html, "html.parser")
        papers: list[Paper] = []
        seen_titles: set[str] = set()

        for item in soup.select(".obj_article_summary"):
            title_el = item.select_one("h2, h3, .tocTitle a, a.tocTitle")
            if not title_el:
                title_el = item.find("a", href=re.compile(r"/article/view/"))
            title = title_el.get_text(strip=True) if title_el else ""
            if not title or len(title) < 5:
                continue
            norm_title = title.lower().strip()
            if norm_title in seen_titles:
                continue
            seen_titles.add(norm_title)

            authors: list[str] = []
            author_el = item.select_one(".tocAuthors, .authors, .meta-authors")
            if author_el:
                author_text = author_el.get_text(strip=True)
                authors = [a.strip() for a in author_text.split(",") if a.strip()]

            pdf_url = None
            for a in item.find_all("a", href=re.compile(r"/article/view/\d+/\d+")):
                href = a.get("href", "")
                if "/article/view/" in href and href.count("/") >= 5:
                    pdf_url = urljoin(base_url, href)
                    break

            paper = Paper(
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                venue_source="ojs",
                source="aaai",
                pdf_url=pdf_url,
            )
            paper.id = build_paper_id(paper)
            papers.append(paper)

        return papers

