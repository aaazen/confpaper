"""SQLite-based download tracking for confpaper.

Provides deduplication across sources (arXiv, CVF, OpenReview, etc.)
and handles re-download when local files are lost or corrupted.
"""

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from confpaper.models import Paper

_DEFAULT_DB = Path("data/papers.sqlite")

DDL = """
CREATE TABLE IF NOT EXISTS downloads (
    id              TEXT PRIMARY KEY,
    title           TEXT    NOT NULL,
    normalized_title TEXT   NOT NULL,
    authors         TEXT,
    year            INTEGER,
    venue           TEXT,
    source          TEXT,
    pdf_url         TEXT,
    arxiv_id        TEXT,
    doi             TEXT,
    local_path      TEXT,
    file_size       INTEGER,
    sha256          TEXT,
    downloaded_at   TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_downloads_arxiv_id
    ON downloads(arxiv_id);

CREATE INDEX IF NOT EXISTS idx_downloads_doi
    ON downloads(doi);

CREATE INDEX IF NOT EXISTS idx_downloads_norm_title_year
    ON downloads(normalized_title, year);

CREATE INDEX IF NOT EXISTS idx_downloads_local_path
    ON downloads(local_path);
"""


def init_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open (or create) the downloads database and ensure the schema exists."""
    path = Path(db_path) if db_path else _DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(DDL)
    conn.commit()
    return conn


def _paper_id(paper: "Paper") -> str:
    """Build a stable dedup key for *this process*, without the full utils import
    chain.  Uses the same rules as utils.build_paper_id()."""
    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id}"
    if paper.doi:
        return f"doi:{paper.doi}"
    from confpaper.utils import normalize_title
    norm = normalize_title(paper.title)
    h = hashlib.sha1(norm.encode()).hexdigest()[:8]
    v = paper.venue or "unknown"
    y = str(paper.year) if paper.year else "unknown"
    s = paper.source or "unknown"
    return f"{s}:{v}:{y}:{h}"


def find_existing_download(
    conn: sqlite3.Connection, paper: "Paper", paper_id: str,
) -> dict | None:
    """Return the existing download row (as a dict) or None.

    Lookup order: arxiv_id → doi → paper_id → normalized_title + year.
    """
    from confpaper.utils import normalize_title
    norm_title = normalize_title(paper.title)

    # 1. arxiv_id
    if paper.arxiv_id:
        row = conn.execute(
            "SELECT * FROM downloads WHERE arxiv_id = ?", (paper.arxiv_id,)
        ).fetchone()
        if row:
            return dict(row)

    # 2. doi
    if paper.doi:
        row = conn.execute(
            "SELECT * FROM downloads WHERE doi = ?", (paper.doi,)
        ).fetchone()
        if row:
            return dict(row)

    # 3. paper id
    row = conn.execute(
        "SELECT * FROM downloads WHERE id = ?", (paper_id,)
    ).fetchone()
    if row:
        return dict(row)

    # 4. normalized_title + year (only if year is known)
    if paper.year is not None:
        row = conn.execute(
            "SELECT * FROM downloads WHERE normalized_title = ? AND year = ?",
            (norm_title, paper.year),
        ).fetchone()
        if row:
            return dict(row)

    return None


def record_download(
    conn: sqlite3.Connection,
    paper: "Paper",
    paper_id: str,
    local_path: str,
    file_size: int,
    sha256: str | None = None,
) -> None:
    """Insert or update a download record.

    ``downloaded_at`` is set only on first insert; ``updated_at`` is always
    refreshed on update.
    """
    from confpaper.utils import normalize_title
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO downloads (id, title, normalized_title, authors,
                               year, venue, source, pdf_url,
                               arxiv_id, doi, local_path,
                               file_size, sha256, downloaded_at, updated_at)
        VALUES (:id, :title, :normalized_title, :authors,
                :year, :venue, :source, :pdf_url,
                :arxiv_id, :doi, :local_path,
                :file_size, :sha256, :downloaded_at, :updated_at)
        ON CONFLICT(id) DO UPDATE SET
            title           = excluded.title,
            normalized_title = excluded.normalized_title,
            authors         = excluded.authors,
            year            = excluded.year,
            venue           = excluded.venue,
            source          = excluded.source,
            pdf_url         = excluded.pdf_url,
            arxiv_id        = excluded.arxiv_id,
            doi             = excluded.doi,
            local_path      = excluded.local_path,
            file_size       = excluded.file_size,
            sha256          = excluded.sha256,
            updated_at      = excluded.updated_at
        """,
        {
            "id": paper_id,
            "title": paper.title,
            "normalized_title": normalize_title(paper.title),
            "authors": json.dumps(paper.authors, ensure_ascii=False) if paper.authors else "[]",
            "year": paper.year,
            "venue": paper.venue,
            "source": paper.source,
            "pdf_url": paper.pdf_url,
            "arxiv_id": paper.arxiv_id,
            "doi": paper.doi,
            "local_path": local_path,
            "file_size": file_size,
            "sha256": sha256,
            "downloaded_at": now,
            "updated_at": now,
        },
    )
    conn.commit()


def export_downloads_to_csv(db_path: Path | str, csv_path: Path | str) -> int:
    """Export the downloads table to a UTF-8 CSV file.

    Returns the number of rows written.
    """
    path = Path(db_path) if db_path else _DEFAULT_DB
    if not path.exists():
        return 0

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM downloads ORDER BY downloaded_at DESC").fetchall()
    conn.close()

    if not rows:
        return 0

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(rows[0].keys())
        for row in rows:
            writer.writerow(row)

    return len(rows)
