import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from confpaper.downloader import DownloadResult, download_paper
from confpaper.search import search_papers, SearchResult

app = typer.Typer(
    name="confpaper",
    help="AI/CV conference paper search and download tool",
)
console = Console()


def _clean_title(title: str) -> str:
    """Sanitize title for terminal display: replace common LaTeX Unicode that renders poorly."""
    # Replace mathematical script/bold/double-struck chars with ASCII equivalents
    replacements = {
        '\N{MATHEMATICAL BOLD CAPITAL A}': 'A', '\N{MATHEMATICAL BOLD CAPITAL B}': 'B',
        '\N{MATHEMATICAL BOLD CAPITAL C}': 'C', '\N{MATHEMATICAL BOLD CAPITAL D}': 'D',
        '\N{MATHEMATICAL BOLD CAPITAL X}': 'X', '\N{MATHEMATICAL BOLD CAPITAL Y}': 'Y',
        '\N{MATHEMATICAL BOLD CAPITAL Z}': 'Z',
    }
    for fancy, plain in replacements.items():
        title = title.replace(fancy, plain)
    # Replace other common problematic Unicode (strip combining chars, zero-width spaces)
    import unicodedata
    title = ''.join(c for c in title if unicodedata.category(c) != 'Mn')  # remove combining marks
    title = title.replace('​', '').replace('‌', '').replace('‍', '')
    title = title.replace('﻿', '')
    return title


def _format_authors(authors: list[str], max_width: int = 35) -> str:
    if not authors:
        return ""
    if len(authors) == 1:
        return authors[0][:max_width]
    text = f"{authors[0]} et al."
    return text[:max_width]


def _display_near_misses(query: str, near_misses: list, limit: int = 5):
    """Show near misses for failed AND queries."""
    from confpaper.matcher import is_strong_group_match
    near = near_misses[:limit]
    if not near:
        console.print("[yellow]No papers found.[/yellow]")
        return
    console.print(f'\n[yellow]No exact matches found for "{query}". Near misses:[/yellow]\n')
    table = Table()
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", style="bold cyan", width=6)
    table.add_column("Title", style="cyan", no_wrap=False, overflow="fold", width=55)
    table.add_column("Matched", style="green", no_wrap=False, overflow="fold", width=16)
    table.add_column("Missing", style="red", no_wrap=False, overflow="fold", width=16)
    table.add_column("Venue", style="yellow", width=7)
    table.add_column("Year", style="yellow", width=5)
    for i, p in enumerate(near, 1):
        scores = getattr(p, "match_group_scores", {}) or {}
        reasons = getattr(p, "match_group_reasons", {}) or {}
        missing_raw = getattr(p, "match_missing_groups", []) or []
        matched_parts = []
        missing_parts = []
        for g in scores:
            s = scores.get(g, 0.0)
            r = reasons.get(g, "none")
            if is_strong_group_match(s, r):
                matched_parts.append(f"{g}={r}:{s:.2f}")
            elif s > 0:
                missing_parts.append(f"{g}(weak:{r}={s:.2f})")
            else:
                missing_parts.append(g)
        table.add_row(
            str(i), f"{p.match_score:.2f}" if p.match_score else "-",
            _clean_title(p.title),
            " ".join(matched_parts) or "-",
            " ".join(missing_parts) or "-",
            p.venue or "-", str(p.year) if p.year else "-",
        )
    console.print(table)


def parse_years(year_str: str | None) -> list[int] | None:
    """Parse year string: 2025, 2023-2025, 2021,2023-2025 → sorted list."""
    if year_str is None or not year_str.strip():
        return None
    years: set[int] = set()
    for part in year_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if start > end:
                raise ValueError(
                    f"Invalid --year range: {part}. "
                    f"Use e.g. 2023-2025, 2025, or 2021,2023-2025."
                )
            for y in range(start, end + 1):
                years.add(y)
        else:
            years.add(int(part))
    return sorted(years) if years else None


def _parse_venues(venue_str: str | None) -> list[str]:
    """Split comma-separated venue string into a list."""
    if not venue_str:
        return []
    return [v.strip() for v in venue_str.split(",") if v.strip()]


async def _search_multi_year(
    query: str, venues: list[str], years: list[int], source: str,
    max_results: int, match_mode: str, match_threshold: float | None,
    author: str | None,
) -> SearchResult:
    """Search across multiple years, merge and dedup."""
    from confpaper.search import merge_papers
    all_papers: list = []
    all_near: list = []
    for y in years:
        try:
            sr = await _search_multi_venue(
                query=query, venues=venues, year=y, source=source,
                max_results=max_results, match_mode=match_mode,
                match_threshold=match_threshold, author=author,
            )
            all_papers.extend(sr.papers)
            all_near.extend(sr.near_misses)
        except Exception as e:
            msg = str(e).strip()
            if msg:
                console.print(f"[yellow]{y}: {msg}[/yellow]")
            continue
    from confpaper.search import _sort_key
    merged = merge_papers(all_papers)
    merged.sort(key=_sort_key)
    near_merged = merge_papers(all_near)
    near_merged.sort(key=_sort_key)
    return SearchResult(papers=merged[:max_results], near_misses=near_merged)


async def _search_multi_venue(
    query: str,
    venues: list[str],
    year: int | None,
    source: str,
    max_results: int,
    match_mode: str = "normal",
    match_threshold: float | None = None,
    author: str | None = None,
) -> SearchResult:
    """Search across multiple venues and merge results."""
    from confpaper.search import merge_papers

    if not venues:
        return await search_papers(
            query=query, venue=None, year=year, source=source, max_results=max_results,
            match_mode=match_mode, match_threshold=match_threshold, author=author,
        )

    all_papers: list = []
    all_near: list = []
    # Disable per-venue truncation when searching multiple venues so
    # a single venue/year can't monopolise results.  The final --max
    # truncation happens after the merge below.
    per_venue_limit = max_results if len(venues) <= 1 else 99999
    for v in venues:
        from confpaper.utils import is_valid_venue_year
        if year is not None and not is_valid_venue_year(v, year):
            continue
        try:
            sr = await search_papers(
                query=query, venue=v, year=year, source=source, max_results=per_venue_limit,
                match_mode=match_mode, match_threshold=match_threshold, author=author,
            )
            all_papers.extend(sr.papers)
            all_near.extend(sr.near_misses)
        except Exception as e:
            msg = str(e).strip()
            if msg:
                console.print(f"[yellow]{v}: {msg}[/yellow]")
            continue

    from confpaper.search import _sort_key
    merged = merge_papers(all_papers)
    merged.sort(key=_sort_key)
    near_merged = merge_papers(all_near)
    near_merged.sort(key=_sort_key)
    return SearchResult(papers=merged[:max_results], near_misses=near_merged)


async def _search_async(
    query: str,
    venue: str | None,
    year: str | None,
    max_results: int,
    download: bool,
    output_dir: Path,
    source: str,
    match_mode: str = "normal",
    match_threshold: float | None = None,
    author: str | None = None,
    near_misses_limit: int = 5,
    db_path: Path | None = None,
):
    """Core async search logic."""
    venues = _parse_venues(venue)
    years = parse_years(year)

    # Detect source aliases passed via --venue (e.g. -v general, -v journal).
    # These should be converted to source routing, not treated as venue names.
    SOURCE_ALIASES = {
        "general": "general", "journal": "general", "journals": "general",
        "arxiv": "arxiv", "semantic": "semantic_scholar",
        "semantic_scholar": "semantic_scholar", "ss": "semantic_scholar",
    }
    alias_venues = [v for v in venues if v.lower() in SOURCE_ALIASES]
    real_venues = [v for v in venues if v.lower() not in SOURCE_ALIASES]

    if alias_venues and real_venues:
        console.print(
            "[red]Error:[/red] Cannot mix conference venues and source aliases "
            "in --venue. Use --source general instead."
        )
        raise typer.Exit(code=1)

    if alias_venues and source == "auto":
        # Convert alias to source routing (first alias wins if multiple)
        source = SOURCE_ALIASES[alias_venues[0].lower()]
        venues = real_venues  # should be empty, but use whatever is left

    # If no --venue and no explicit --source, default to core AI/CV conferences.
    if not venues and source == "auto":
        venues = [
            "CVPR", "ICCV", "ECCV", "WACV",
            "AAAI", "ICLR", "ICML", "NeurIPS",
        ]
        console.print("[dim]No venue specified. Using default venues;[/dim]")

    try:
        if years and len(years) > 1:
            result = await _search_multi_year(
                query=query, venues=venues, years=years, source=source,
                max_results=max_results, match_mode=match_mode,
                match_threshold=match_threshold, author=author,
            )
        else:
            y = years[0] if years else None
            result = await _search_multi_venue(
                query=query, venues=venues, year=y, source=source,
                max_results=max_results, match_mode=match_mode,
                match_threshold=match_threshold, author=author,
            )
        papers = result.papers
        near_misses = result.near_misses

        # Post-filter: enforce --year range on final results.
        # Sources (especially arXiv / Semantic Scholar) may return papers
        # outside the requested year range.
        if years:
            papers = [p for p in papers if p.year is not None and p.year in years]
            near_misses = [p for p in near_misses if p.year is not None and p.year in years]
    except NotImplementedError as e:
        console.print(f"[yellow]{e}[/yellow]")
        if source in ("general", "arxiv", "semantic_scholar"):
            console.print(
                "[dim]General search is planned for Phase 2. "
                "Current working source: CVF. "
                'Try: confpaper search "..." --venue CVPR --year 2024[/dim]'
            )
        elif source == "openreview":
            console.print(
                "[dim]OpenReview search is planned for Phase 3. "
                "Current working source: CVF. "
                'Try: confpaper search "..." --venue CVPR --year 2024[/dim]'
            )
        else:
            console.print(
                '[dim]Try: confpaper search "..." --venue CVPR --year 2024[/dim]'
            )
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        # Network errors, etc. — user-friendly message, no traceback
        msg = str(e).strip()
        if not msg:
            msg = type(e).__name__
        console.print(f"[red]Network/connection error:[/red] {msg}")
        console.print(
            "[dim]The server may be unreachable. Check your network or proxy settings.[/dim]"
        )
        raise typer.Exit(code=1)

    if not papers:
        if near_misses and near_misses_limit > 0:
            _display_near_misses(query, near_misses[:near_misses_limit])
        else:
            console.print("[yellow]No papers found.[/yellow]")
        raise typer.Exit()

    # --- Display results table ---
    title_str = f'Results for author: "{author}"' if author else f'Results for "{query}"'
    table = Table(title=title_str)
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", style="bold cyan", width=6)
    table.add_column("Match", style="dim", width=7)
    table.add_column("Title", style="cyan", no_wrap=False, overflow="fold", width=60)
    table.add_column("Authors", style="green", no_wrap=False, overflow="fold", width=16)
    table.add_column("Venue", style="yellow", width=7)
    table.add_column("Year", style="yellow", width=5)
    table.add_column("Source", style="magenta", width=6)
    table.add_column("PDF", style="blue", width=4)

    for i, paper in enumerate(papers, 1):
        pdf_str = "✓" if paper.pdf_url else "-"
        score_str = f"{paper.match_score:.2f}" if paper.match_score else "-"
        reason_str = paper.match_reason or "-"
        authors_str = _format_authors(paper.authors)
        if author:
            authors_str = f"{authors_str} [{author}]"
        table.add_row(
            str(i),
            score_str,
            reason_str,
            _clean_title(paper.title),
            authors_str,
            paper.venue or "-",
            str(paper.year) if paper.year else "-",
            paper.source[:5],
            pdf_str,
        )

    console.print(table)

    # --- Download if requested ---
    if download:
        await _download_papers(papers, query, output_dir, db_path=db_path)


async def _download_papers(
    papers: list,
    query: str,
    output_dir: Path,
    db_path: Path | None = None,
):
    """Download PDFs for a list of papers."""
    console.print(f"\n[bold]Downloading {len(papers)} papers...[/bold]")
    download_results: dict[str, DownloadResult] = {}

    import httpx
    for i, paper in enumerate(papers, 1):
        result = await download_paper(paper, output_dir, client=None, db_path=db_path)
        key = paper.id or str(i)
        download_results[key] = result

        if result.status == "downloaded":
            console.print(f"  [{i}] [green]OK[/green] {result.path}")
        elif result.status == "skipped":
            console.print(f"  [{i}] [dim]SKIP[/dim] {result.path}")
        else:
            console.print(f"  [{i}] [red]FAIL[/red] {result.message}")

    # Re-display table with download status
    table_dl = Table(title=f'Download Results for "{query}"')
    table_dl.add_column("#", style="dim", width=4)
    table_dl.add_column("Title", style="cyan", no_wrap=False, overflow="fold", width=55)
    table_dl.add_column("Venue", style="yellow", width=10)
    table_dl.add_column("Year", style="yellow", width=6)
    table_dl.add_column("Status", style="green", no_wrap=False, overflow="fold")

    for i, paper in enumerate(papers, 1):
        key = paper.id or str(i)
        result = download_results.get(key)
        if result and result.status == "downloaded":
            status = f"[green]downloaded[/green] {result.path}"
        elif result and result.status == "skipped":
            status = f"[dim]skipped[/dim] {result.path}"
        elif result and result.status == "failed":
            status = f"[red]failed[/red] {result.message}"
        else:
            status = "[yellow]unknown[/yellow]"

        table_dl.add_row(
            str(i),
            _clean_title(paper.title),
            paper.venue or "unknown",
            str(paper.year) if paper.year else "?",
            status,
        )

    console.print()
    console.print(table_dl)


@app.command()
def search(
    query: str = typer.Argument("", help="Search query (title or keywords). Use --author for author-only search."),
    venue: Optional[str] = typer.Option(
        None, "--venue", "-v", help="Target conference(s), comma-separated (e.g. CVPR,ICCV,NeurIPS)"
    ),
    year: Optional[str] = typer.Option(
        None, "--year", "-y", help="Year or range: 2025, 2023-2025, 2021,2023-2025"
    ),
    max_results: int = typer.Option(
        20, "--max", "-m", help="Maximum results to show"
    ),
    download: bool = typer.Option(
        False, "--download", "-d", help="Download PDFs for matching papers"
    ),
    output_dir: Path = typer.Option(
        Path("downloads"),
        "--output-dir", "-o",
        help="Directory for downloaded PDFs",
    ),
    source: str = typer.Option(
        "auto",
        "--source", "-s",
        help="Search source: auto, cvf, openreview, general, arxiv, semantic_scholar",
    ),
    match_mode: str = typer.Option(
        "normal",
        "--match", help="Match strictness: strict, normal, loose",
    ),
    match_threshold: Optional[float] = typer.Option(
        None, "--threshold", help="Override match threshold (0.0-1.0)",
    ),
    author: Optional[str] = typer.Option(
        None, "--author", help="Filter by author name (e.g. \"Kaiming He\")",
    ),
    near_misses_limit: int = typer.Option(
        5, "--near-misses", help="Show up to N near misses for AND queries with no exact matches (0=off)",
    ),
    db: Path = typer.Option(
        Path("data/papers.sqlite"),
        "--db", help="Path to download tracking database",
    ),
):
    """Search for AI/CV conference papers."""
    if not query and not author:
        console.print("[red]Error:[/red] Provide a search query or --author.")
        raise typer.Exit(code=1)
    try:
        parse_years(year)  # validate early
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e
    asyncio.run(_search_async(
        query=query,
        venue=venue,
        year=year,
        max_results=max_results,
        download=download,
        output_dir=output_dir,
        source=source,
        match_mode=match_mode,
        match_threshold=match_threshold,
        author=author,
        near_misses_limit=near_misses_limit,
        db_path=db if download else None,
    ))


@app.command()
def download(
    identifier: str = typer.Argument(..., help="arXiv ID, DOI, or paper URL"),
):
    """Download a paper by arXiv ID, DOI, or URL."""
    console.print(
        "[yellow]The 'download' command is planned for a later phase.[/yellow]"
    )
    console.print(
        "[dim]Use 'confpaper search' with --download instead.[/dim]"
    )


@app.command()
def scrape(
    venue: str = typer.Option(..., "--venue", "-v", help="Conference to scrape (e.g. ICLR, NeurIPS, ICML)"),
    year: int = typer.Option(..., "--year", "-y", help="Conference year"),
    filter: Optional[str] = typer.Option(
        None, "--filter", "-f", help="Keyword filter on titles/abstracts"
    ),
    max_results: int = typer.Option(
        50, "--max", "-m", help="Maximum papers to fetch"
    ),
    download: bool = typer.Option(
        False, "--download", "-d", help="Download PDFs"
    ),
    output_dir: Path = typer.Option(
        Path("downloads"), "--output-dir", "-o"
    ),
):
    """Scrape all accepted papers from an OpenReview conference (ICLR, NeurIPS, ICML)."""
    query = filter or ""
    asyncio.run(_search_async(
        query=query,
        venue=venue,
        year=year,
        max_results=max_results,
        download=download,
        output_dir=output_dir,
        source="auto",
    ))


@app.command()
def bib(
    query: str = typer.Argument(..., help="Search query for BibTeX export"),
    venue: Optional[str] = typer.Option(
        None, "--venue", "-v", help="Target conference"
    ),
    year: Optional[str] = typer.Option(
        None, "--year", "-y", help="Year or range: 2025, 2023-2025, 2021,2023-2025"
    ),
    output: Path = typer.Option(
        Path("references.bib"), "--output", "-o", help="Output .bib file"
    ),
    max_results: int = typer.Option(
        20, "--max", "-m", help="Maximum references"
    ),
):
    """Search papers and export BibTeX references."""
    console.print(
        "[yellow]The 'bib' command is planned for Phase 4.[/yellow]"
    )
    console.print(
        "[dim]For now, use 'confpaper search' to find and download papers.[/dim]"
    )


@app.command()
def export_downloads(
    output: Path = typer.Option(
        Path("downloads.csv"),
        "--output", "-o",
        help="Output CSV file path",
    ),
    db: Path = typer.Option(
        Path("data/papers.sqlite"),
        "--db", help="Path to download tracking database",
    ),
):
    """Export download history to a CSV file."""
    from confpaper.database import export_downloads_to_csv

    if not db.exists():
        console.print("[yellow]No download database found.[/yellow]")
        raise typer.Exit()

    count = export_downloads_to_csv(db, output)
    if count == 0:
        console.print("[yellow]No download records to export.[/yellow]")
    else:
        console.print(f"[green]Exported {count} records to {output}[/green]")


def main():
    app()


if __name__ == "__main__":
    main()
