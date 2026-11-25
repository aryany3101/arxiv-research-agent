"""Build an isolated retrieval benchmark from versioned arXiv PDFs."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from extract_text import chunk_text, extract_text

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "rag_benchmark.json"


def prepare(output_dir):
    papers = json.loads(MANIFEST.read_text())["papers"]
    chunk_path = output_dir / "processed" / "chunks.jsonl"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=chunk_path.parent) as tmp:
        scratch = Path(tmp)
        pending = scratch / "chunks.jsonl"
        with pending.open("w", encoding="utf-8") as out:
            for arxiv_id, title in papers.items():
                print(f"Downloading {arxiv_id}: {title}", flush=True)
                pdf_path = scratch / f"{arxiv_id}.pdf"
                with urlopen(f"https://arxiv.org/pdf/{arxiv_id}", timeout=30) as response:
                    pdf_path.write_bytes(response.read())
                for index, text in enumerate(chunk_text(extract_text(pdf_path))):
                    record = {
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "url": f"https://arxiv.org/abs/{arxiv_id}",
                        "chunk_index": index,
                        "text": text,
                    }
                    out.write(json.dumps(record) + "\n")
        pending.replace(chunk_path)
    print(f"Wrote chunks from {len(papers)} papers to {chunk_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    prepare(parser.parse_args().output_dir)
