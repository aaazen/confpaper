import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from confpaper.models import Paper
from confpaper.utils import clean_title

USER_AGENT = "confpaper/0.2 (+https://github.com/confpaper)"


@dataclass
class DownloadResult:
    status: str  # "downloaded" | "skipped" | "failed"
    path: Path | None = None
    message: str | None = None


async def download_paper(
    paper: Paper,
    output_dir: Path,
    client: httpx.AsyncClient | None = None,
) -> DownloadResult:
    if not paper.pdf_url:
        return DownloadResult(status="failed", message="No PDF URL")

    venue_dir = paper.venue or "unknown"
    year_dir = str(paper.year) if paper.year else "unknown"
    filename = f"{clean_title(paper.title)}.pdf"
    target_dir = output_dir / venue_dir / year_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    if target_path.exists() and target_path.stat().st_size > 0:
        return DownloadResult(status="skipped", path=target_path)

    tmp_path = target_path.with_suffix(".tmp")

    # If shared client provided, use it; otherwise try proxy→direct
    if client is not None:
        return await _do_download(client, paper.pdf_url, tmp_path, target_path, paper)

    for trust_env in (True, False):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(60, connect=15),
            follow_redirects=True,
            trust_env=trust_env,
        ) as c:
            try:
                return await _do_download(c, paper.pdf_url, tmp_path, target_path, paper)
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError):
                if not trust_env:
                    raise
                continue

    _cleanup(tmp_path)
    return DownloadResult(status="failed", message="All connection attempts failed")


async def _do_download(
    client: httpx.AsyncClient,
    url: str,
    tmp_path: Path,
    target_path: Path,
    paper: Paper,
) -> DownloadResult:
    for attempt in range(3):
        try:
            response = await client.get(url)
            response.raise_for_status()
            tmp_path.write_bytes(response.content)
            tmp_path.rename(target_path)
            paper.local_path = str(target_path)
            paper.downloaded_at = datetime.now(timezone.utc).isoformat()
            return DownloadResult(status="downloaded", path=target_path)
        except httpx.HTTPStatusError as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return DownloadResult(status="failed", message=f"HTTP {e.response.status_code}")
        except (httpx.RequestError, OSError) as e:
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return DownloadResult(status="failed", message=str(e)[:120])

    return DownloadResult(status="failed", message="Max retries exceeded")


def _cleanup(path: Path):
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
