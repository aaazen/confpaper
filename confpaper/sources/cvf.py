import logging
import os
import re
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from confpaper.models import Paper
from confpaper.utils import build_paper_id, canonicalize_venue, get_cvf_code

logger = logging.getLogger(__name__)
USER_AGENT = "confpaper/0.1 (+https://github.com/confpaper)"
CVF_BASE = "https://openaccess.thecvf.com/"
ECCV_BASE = "https://www.ecva.net/papers.php"
ECCV_YEAR_PATTERN = re.compile(r"/eccv_(\d{4})/")


def _debug() -> bool:
    return os.environ.get("CONFPAPER_DEBUG", "").strip() in ("1", "true", "yes")


def _get_cvf_url_candidates(cvf_code: str, year: int) -> list[str]:
    """Return URL candidates for a CVF venue/year, ordered by likelihood.

    Different CVF conferences and years use different URL patterns.
    We try each candidate until one returns HTTP 200 with actual papers.
    """
    code_upper = cvf_code.upper()
    candidates = [
        f"{CVF_BASE}{code_upper}{year}?day=all",
        f"{CVF_BASE}{code_upper}{year}",
        f"{CVF_BASE}{code_upper}{year}.py",
    ]
    # Some ICCV/WACV years only respond to the .py variant
    if code_upper in ("ICCV", "WACV"):
        candidates.append(f"{CVF_BASE}{code_upper}/{year}")
    return candidates


class CVFSource:
    name = "cvf"

    async def search(
        self,
        query: str,
        venue: str | None = None,
        year: int | None = None,
        max_results: int = 20,
    ) -> list[Paper]:
        venue_canonical = canonicalize_venue(venue) if venue else None
        if not venue_canonical or not year:
            raise ValueError("CVF source requires --venue and --year")

        # ECCV is on ecva.net (same dt/dd structure, different base URL)
        if venue_canonical == "ECCV":
            return await self._search_eccv(query, venue_canonical, year, max_results)

        cvf_code = get_cvf_code(venue_canonical)
        if not cvf_code:
            raise ValueError(
                f"Venue '{venue_canonical}' is not available via CVF/ECVA. "
                f"Supported: CVPR, ICCV, WACV, ECCV."
            )

        candidates = _get_cvf_url_candidates(cvf_code, year)
        errors: list[str] = []

        for url in candidates:
            try:
                html = await self._fetch(url)
            except RuntimeError as e:
                errors.append(f"{url}: {e}")
                continue

            papers = self._parse_papers(html, venue_canonical, year, url)
            if papers:
                if _debug():
                    logger.info(
                        "cvf: %s %d succeeded via %s, parsed %d papers",
                        venue_canonical, year, url, len(papers),
                    )
                return papers

            # HTTP 200 but no papers → try next candidate
            errors.append(f"{url}: HTTP 200 but parsed 0 papers")

        # All candidates failed
        raise RuntimeError(
            f"{venue_canonical} {year} failed after trying all URL candidates:\n  "
            + "\n  ".join(errors)
        )

    async def _search_eccv(
        self, query: str, venue: str, year: int, max_results: int
    ) -> list[Paper]:
        """ECCV uses ecva.net/papers.php with all years on one page."""
        try:
            html = await self._fetch(ECCV_BASE)
        except RuntimeError as e:
            raise RuntimeError(f"ECCV {year}: Cannot reach {ECCV_BASE}\n  {e}")

        papers = self._parse_papers(html, venue, year, ECCV_BASE)

        # ecva.net has ALL years — filter by year from URL path
        papers = [p for p in papers if p.year == year]

        if not papers:
            raise RuntimeError(
                f"ECCV {year}: No papers parsed from ecva.net. "
                "The page layout may have changed."
            )

        return papers  # scoring done centrally in search.py

    async def _fetch(self, url: str) -> str:
        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        except httpx.HTTPError as e:
            detail = str(e).strip()
            if detail:
                raise RuntimeError(f"Cannot reach {url}\n  {detail}")
            raise RuntimeError(f"Cannot reach {url}")

    def _parse_papers(
        self, html: str, venue: str, year: int, base_url: str
    ) -> list[Paper]:
        soup = BeautifulSoup(html, "html.parser")
        papers: list[Paper] = []

        title_links = soup.find_all("a", href=re.compile(r"/html/"))
        if not title_links:
            title_links = self._fallback_title_links(soup)

        for link in title_links:
            title = link.get_text(strip=True)
            if not title or len(title) < 3:
                continue

            html_url = urljoin(base_url, link.get("href", ""))
            dds = self._find_dds(link)
            authors, pdf_url, supp_url = self._extract_from_dds(dds, base_url)

            # For ECCV (ecva.net), extract year from URL path
            paper_year = year
            if html_url:
                ym = ECCV_YEAR_PATTERN.search(html_url)
                if ym:
                    paper_year = int(ym.group(1))

            paper = Paper(
                title=title,
                authors=authors,
                year=paper_year,
                venue=venue,
                venue_source="cvf",
                source="cvf",
                pdf_url=pdf_url,
                html_url=html_url,
                supp_url=supp_url,
            )
            paper.id = build_paper_id(paper)
            papers.append(paper)

        return papers

    def _fallback_title_links(self, soup: BeautifulSoup) -> list:
        links = []
        for dt in soup.find_all("dt"):
            a = dt.find("a")
            if a and a.get_text(strip=True):
                links.append(a)
        return links

    def _find_dds(self, title_link) -> list:
        """Return all <dd> elements after the title's <dt> (CVPR 2025 has two:
        first for authors, second for pdf/bibtex)."""
        parent = title_link.parent
        if parent and parent.name == "dt":
            dds = []
            dd = parent.find_next("dd")
            if dd:
                dds.append(dd)
                dd2 = dd.find_next("dd")
                if dd2:
                    dds.append(dd2)
            return dds
        return []

    def _extract_from_dds(
        self, dds: list, base_url: str
    ) -> tuple[list[str], str | None, str | None]:
        authors: list[str] = []
        pdf_url: str | None = None
        supp_url: str | None = None

        for dd in dds:
            if dd is None:
                continue

            for form in dd.find_all("form", class_="authsearch"):
                inp = form.find("input", attrs={"name": "query_author"})
                if inp and inp.get("value"):
                    authors.append(inp["value"].strip())

            if not authors:
                full_text = dd.get_text(separator=" ", strip=True)
                for noise in ["pdf", "supp", "supplemental", "arXiv", "DOI", "Back"]:
                    full_text = re.sub(
                        r"\b" + re.escape(noise) + r"\b", "", full_text, flags=re.IGNORECASE
                    )
                full_text = re.sub(r"\s+", " ", full_text).strip().strip(",").strip()
                if full_text:
                    authors = [a.strip() for a in full_text.split(",") if a.strip()]

            for a in dd.find_all("a"):
                href = a.get("href", "")
                h = href.lower()
                link_text = a.get_text(strip=True).lower()
                if not pdf_url and ("pdf" in h or link_text == "pdf"):
                    pdf_url = urljoin(base_url, href)
                if not supp_url and ("supp" in h or "supp" in link_text):
                    supp_url = urljoin(base_url, href)

        return authors, pdf_url, supp_url

