import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from confpaper.database import find_existing_download, init_db, record_download
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
    db_path: Path | None = None,
) -> DownloadResult:
    if not paper.pdf_url:
        return DownloadResult(status="failed", message="No PDF URL")

    venue_dir = paper.venue or "unknown"
    year_dir = str(paper.year) if paper.year else "unknown"
    filename = f"{clean_title(paper.title)}.pdf"
    target_dir = output_dir / venue_dir / year_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    # 0-byte files should be re-downloaded
    if target_path.exists() and target_path.stat().st_size == 0:
        target_path.unlink()

    # --- Check database for existing record ---
    conn = None
    paper_id = None
    existing = None
    if db_path is not None:
        from confpaper.database import _paper_id
        conn = init_db(db_path)
        paper_id = _paper_id(paper)
        existing = find_existing_download(conn, paper, paper_id)

        if existing:
            recorded_path = existing.get("local_path")
            if recorded_path:
                recorded_file = Path(recorded_path)
                if recorded_file.exists() and recorded_file.stat().st_size > 0:
                    conn.close()
                    return DownloadResult(
                        status="skipped",
                        path=recorded_file,
                        message="Already downloaded (tracked)",
                    )
            # Record exists but file is missing or 0-byte — re-download.
            # Use the existing record's id so the upsert updates the right row.
            paper_id = existing["id"]

    # If no DB record but file is already on disk, record it and skip.
    if existing is None and target_path.exists() and target_path.stat().st_size > 0:
        if conn is not None and paper_id is not None:
            record_download(
                conn, paper, paper_id,
                local_path=str(target_path),
                file_size=target_path.stat().st_size,
            )
            conn.close()
        return DownloadResult(status="skipped", path=target_path)

    tmp_path = target_path.with_suffix(".tmp")

    # Build download context for DB recording on success
    ctx = {
        "conn": conn,
        "paper_id": paper_id,
        "target_path": target_path,
        "existing": existing if db_path is not None else None,
    }

    # If shared client provided, use it; otherwise try proxy→direct
    if client is not None:
        return await _do_download(client, paper.pdf_url, tmp_path, target_path, paper, ctx)

    for trust_env in (True, False):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(60, connect=15),
            follow_redirects=True,
            trust_env=trust_env,
        ) as c:
            try:
                return await _do_download(c, paper.pdf_url, tmp_path, target_path, paper, ctx)
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
    ctx: dict | None = None,
) -> DownloadResult:
    for attempt in range(3):
        try:
            response = await client.get(url)
            response.raise_for_status()
            tmp_path.write_bytes(response.content)
            tmp_path.rename(target_path)
            paper.local_path = str(target_path)
            paper.downloaded_at = datetime.now(timezone.utc).isoformat()

            # Record in database on success
            _record_on_success(ctx, paper, target_path)

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


def _record_on_success(ctx: dict | None, paper: Paper, target_path: Path) -> None:
    """Write or update the download record after a successful download."""
    if ctx is None:
        return
    conn = ctx.get("conn")
    paper_id = ctx.get("paper_id")
    if conn is None or paper_id is None:
        return

    try:
        file_size = target_path.stat().st_size
    except OSError:
        file_size = 0

    sha256 = None
    try:
        content = target_path.read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
    except OSError:
        pass

    record_download(
        conn, paper, paper_id,
        local_path=str(target_path),
        file_size=file_size,
        sha256=sha256,
    )
    conn.close()


def _cleanup(path: Path):
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
