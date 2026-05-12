# confpaper

A CLI tool for searching and downloading AI/CV conference papers from official
sources and arXiv.

> This public version does not use Semantic Scholar and does not require any
> private API key.

## Features

- Search AI/CV conference papers from official sources
- Default venues: CVPR, ICCV, ECCV, WACV, AAAI, ICLR, ICML, NeurIPS
- Search by keyword, title, author
- Year and year-range filtering (`-y 2023-2025`)
- AND query with `+` (e.g. `yolo + object detection`)
- arXiv search via `--source arxiv` or `--source general`
- PDF download with `--download`
- Automatic download deduplication via SQLite
- Local file organization by venue and year
- Export download history to CSV

## Supported sources

| Venue / Source | Backend |
|---|---|
| CVPR, ICCV, WACV | CVF Open Access |
| ECCV | ecva.net |
| AAAI | AAAI OJS |
| ICLR, NeurIPS, ICML | OpenReview |
| arXiv | arXiv API |

## Installation

```bash
git clone https://github.com/aaazen/confpaper.git
cd confpaper
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

```bash
# Default AI/CV conference search
confpaper search "object detection" -y 2023-2025 -m 20

# AND query: must match both concepts
confpaper search "yolo + object detection" -y 2023-2025 -m 50

# Filter by specific venues
confpaper search "real-time + object detection" \
  -v CVPR,ICCV,WACV,AAAI,ICLR,ICML,NeurIPS -y 2023-2025 -m 50

# Author search
confpaper search --author "Kaiming He" -v CVPR,ICCV,NeurIPS -y 2020-2025 -m 20

# arXiv search
confpaper search "state space model object detection" --source arxiv -y 2023-2025 -m 30

# Download PDFs
confpaper search "yolo + object detection" -y 2023-2025 --download
```

## Query syntax

- **Space-separated query** — normal keyword phrase matching.
- **`+` operator** — strict AND between concepts. Every group must match.
- **Model/acronym search** — `YOLO`, `DETR`, `SAM`, `ViT`, `CLIP`, `Mamba`, etc.

```
object detection          # broad topic search
yolo + object detection   # must match both YOLO and object detection
feature selection         # fixed phrase search
```

## Options

| Option | Description |
|---|---|
| `-v`, `--venue` | Target conference(s), comma-separated |
| `-y`, `--year` | Year or range: `2025`, `2023-2025` |
| `-m`, `--max` | Maximum results to show (default: 20) |
| `--match` | Match strictness: `strict`, `normal`, `loose` |
| `--threshold` | Override match threshold (0.0–1.0) |
| `--author` | Filter by author name |
| `--near-misses` | Show near misses for AND queries (default: 5) |
| `--download`, `-d` | Download PDFs for matching papers |
| `--output-dir`, `-o` | Directory for downloaded PDFs |
| `--db` | Path to download tracking database (default: `data/papers.sqlite`) |
| `--source`, `-s` | Search source: `auto`, `cvf`, `openreview`, `aaai`, `arxiv`, `general` |

## Venue filtering

ICCV (odd years) and ECCV (even years) are automatically skipped in non-event
years — no warnings, no HTTP requests.

```bash
confpaper search "object detection" -v ICCV -y 2023-2025  # odd years only
confpaper search "object detection" -v ECCV -y 2023-2025  # even years only
```

## Download tracking

confpaper uses a SQLite database (`data/papers.sqlite`) to track downloaded
papers and avoid duplicate downloads. Deduplication uses the first available of:
arxiv ID, DOI, paper ID, or normalized title + year.

```bash
# Download PDFs (automatically records and deduplicates)
confpaper search "yolo + object detection" -y 2023-2025 --download

# Use a custom database path
confpaper search "yolo + object detection" -y 2023-2025 --download --db my_tracking.sqlite
```

**Behavior:**

- If a paper was already downloaded and the file still exists → **skipped**.
- If the database has a record but the PDF is missing or 0-byte → **re-downloaded**
  and the record is updated.
- If a file exists on disk but isn't tracked yet → recorded and skipped.

### Export download history

```bash
confpaper export-downloads -o downloads.csv
confpaper export-downloads -o downloads.csv --db data/papers.sqlite
```

PDFs are saved to `downloads/{venue}/{year}/{clean_title}.pdf`.

## arXiv search

```bash
confpaper search "state space model object detection" --source arxiv -y 2023-2025 -m 30
```

`--source general` is treated as arXiv-only in the public version.

## Limitations

- Search quality depends on official pages and available metadata.
- Some official pages may change layout.
- arXiv results may include preprints and non-conference papers.
- This public version does not include Semantic Scholar metadata enrichment.
- Rate limits may apply for arXiv and OpenReview APIs.

## Development

```
confpaper/
  cli.py           CLI interface (Typer)
  router.py        Source routing logic
  search.py        Search orchestration, scoring, dedup
  matcher.py       Query classification and matching
  models.py        Paper data model
  downloader.py    PDF downloader
  database.py      Download tracking (SQLite)
  utils.py         Venue normalization, helpers
  sources/
    cvf.py         CVPR, ICCV, WACV, ECCV
    aaai_source.py AAAI
    openreview_source.py  ICLR, NeurIPS, ICML
    arxiv_source.py       arXiv API
```

## License

MIT
