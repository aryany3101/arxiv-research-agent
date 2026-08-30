import fitz
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
METADATA_PATH = DATA_DIR / "metadata.json"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.jsonl"

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_all_papers():
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METADATA_PATH) as f:
        papers = json.load(f)

    total_chunks = 0
    with open(CHUNKS_PATH, "w") as out_f:
        for paper in papers:
            pdf_path = RAW_PDF_DIR / f"{paper['arxiv_id']}.pdf"
            if not pdf_path.exists():
                print(f"Skipping missing file: {pdf_path}")
                continue

            print(f"Processing: {paper['title']}")
            text = extract_text(pdf_path)
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                record = {
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "url": paper["url"],
                    "chunk_index": i,
                    "text": chunk,
                }
                out_f.write(json.dumps(record) + "\n")
                total_chunks += 1

    print(f"\nDone. Wrote {total_chunks} chunks from {len(papers)} papers.")
    print(f"Saved to: {CHUNKS_PATH}")

if __name__ == "__main__":
    process_all_papers()