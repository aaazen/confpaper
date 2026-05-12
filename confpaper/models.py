from pydantic import BaseModel


class Paper(BaseModel):
    id: str | None = None
    title: str
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    venue_source: str | None = None
    abstract: str | None = None
    keywords: list[str] = []
    source: str
    pdf_url: str | None = None
    html_url: str | None = None
    supp_url: str | None = None
    arxiv_id: str | None = None
    doi: str | None = None
    citation_count: int | None = None
    # Relevance scoring (set by matcher after search)
    match_score: float | None = None
    match_reason: str | None = None
    matched_query: str | None = None
    # Near-miss fields (AND queries): per-group scores/reasons/missing
    match_group_scores: dict | None = None
    match_group_reasons: dict | None = None
    match_missing_groups: list | None = None
    local_path: str | None = None
    downloaded_at: str | None = None
