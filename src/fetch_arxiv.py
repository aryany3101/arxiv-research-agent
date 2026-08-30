import arxiv
import json
import time
import requests
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
METADATA_PATH = DATA_DIR / "metadata.json"

def fetch_papers(max_results):
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:cs.AI",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    results = list(client.results(search))
    print(f"Found {len(results)} papers")

    all_metadata = []
    for r in results:
        arxiv_id = r.get_short_id()
        pdf_path = RAW_PDF_DIR / f"{arxiv_id}.pdf"

        print(f"Downloading: {r.title}")
        response = requests.get(r.pdf_url)
        response.raise_for_status()
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        time.sleep(1)

        all_metadata.append({
            "arxiv_id": arxiv_id,
            "title": r.title,
            "abstract": r.summary,
            "url": r.entry_id,
        })

    with open(METADATA_PATH, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Saved {len(all_metadata)} papers to {METADATA_PATH}")

if __name__ == "__main__":
    fetch_papers(max_results=15)
