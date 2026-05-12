from typing import Protocol

from confpaper.models import Paper


class PaperSource(Protocol):
    """Protocol for paper search sources. Download is handled by downloader.py."""

    name: str

    async def search(
        self,
        query: str,
        venue: str | None = None,
        year: int | None = None,
        max_results: int = 20,
    ) -> list[Paper]:
        ...
