import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_PATH = DATA_DIR / "metadata.json"

with open(METADATA_PATH) as f:
    papers = json.load(f)

for i, p in enumerate(papers, 1):
    print(f"\n{i}. [{p['arxiv_id']}] {p['title']}")
    print(f"   {p['abstract'][:200]}...")