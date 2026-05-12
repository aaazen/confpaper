# Phase 4 TODO:
# - SQLite database for tracking downloaded papers
# - Table: papers(id, title, authors, year, venue, source, pdf_url,
#           arxiv_id, doi, normalized_title, local_path, downloaded_at)
# - Dedup strategy: arxiv_id > doi > normalized_title + year
# - Functions: init_db(), insert_paper(), is_downloaded(), get_all_papers()
