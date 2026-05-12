import asyncio
import logging
import re

import httpx

from confpaper.models import Paper
from confpaper.utils import build_paper_id, canonicalize_venue

logger = logging.getLogger(__name__)
OR_API = "https://api2.openreview.net/notes"
OR_PDF_BASE = "https://openreview.net"
USER_AGENT = "confpaper/0.2 (+https://github.com/confpaper)"
PAGE_SIZE = 100

OR_VENUE_MAP = {
    "ICLR": "ICLR.cc",
    "NeurIPS": "NeurIPS.cc",
    "ICML": "ICML.cc",
}


class OpenReviewSource:
    name = "openreview"

    async def search(
        self,
        query: str,
        venue: str | None = None,
        year: int | None = None,
        max_results: int = 20,
    ) -> list[Paper]:
        venue_canonical = canonicalize_venue(venue) if venue else None
        if not venue_canonical or not year:
            raise ValueError("OpenReview source requires --venue and --year")

        or_id = OR_VENUE_MAP.get(venue_canonical)
        if not or_id:
            raise ValueError(
                f"Venue '{venue_canonical}' is not on OpenReview. "
                f"OpenReview supports: ICLR, NeurIPS, ICML."
            )

        venue_id = f"{or_id}/{year}/Conference"
        papers = await self._fetch_accepted(venue_id, venue_canonical, year, max_results)
        return papers

    async def _fetch_accepted(
        self, venue_id: str, venue: str, year: int, max_results: int
    ) -> list[Paper]:
        papers: list[Paper] = []
        offset = 0
        seen_ids: set[str] = set()
        page = 0

        while True:
            async with httpx.AsyncClient(
                headers={"User-Agent": USER_AGENT},
                timeout=30,
                follow_redirects=True,
            ) as client:
                for retry in range(3):
                    response = await client.get(OR_API, params={
                        "content.venueid": venue_id,
                        "limit": PAGE_SIZE,
                        "offset": offset,
                    })
                    if response.status_code == 429 and retry < 2:
                        await asyncio.sleep((retry + 1) * 10)
                        continue
                    response.raise_for_status()
                    break
                data = response.json()

            notes = data.get("notes", [])
            if not notes:
                break

            for note in notes:
                nid = note.get("id", "")
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)

                content = note.get("content", {})
                title = _val(content, "title")
                if not title:
                    continue

                authors = _val(content, "authors", [])
                abstract = _val(content, "abstract")
                keywords = _val_keywords(content)
                pdf_path = _val(content, "pdf")
                pdf_url = f"{OR_PDF_BASE}{pdf_path}" if pdf_path else None

                paper = Paper(
                    title=title,
                    authors=authors if isinstance(authors, list) else [authors],
                    year=year,
                    venue=venue,
                    venue_source="openreview",
                    source="openreview",
                    abstract=abstract,
                    keywords=keywords,
                    pdf_url=pdf_url,
                )
                paper.id = build_paper_id(paper)
                papers.append(paper)

            offset += PAGE_SIZE
            page += 1
            if page % 5 == 0:
                logger.info("%s %d: fetched %d papers...", venue_id, year, len(papers))
            await asyncio.sleep(0.3)

        logger.info("%s %d: total %d papers fetched", venue_id, year, len(papers))
        return papers


def _val_keywords(content: dict) -> list[str]:
    for field_name in ("keywords", "Keywords", "subject_areas"):
        val = content.get(field_name)
        if val is None:
            continue
        if isinstance(val, dict):
            v = val.get("value", "")
        else:
            v = val
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [k.strip() for k in re.split(r"[,;]", v) if k.strip()]
    return []


def _val(content: dict, key: str, default=""):
    field = content.get(key)
    if isinstance(field, dict):
        v = field.get("value", default)
        return v
    if isinstance(field, (str, list)):
        return field
    return default
