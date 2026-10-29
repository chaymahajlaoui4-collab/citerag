"""Day 1-2: load documents -> chunk -> embed -> store in Chroma."""
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

import config


def load_pages(path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                yield i, page.get_text()
    elif suffix in {".md", ".txt"}:
        yield 1, path.read_text(encoding="utf-8", errors="ignore")


def chunk_words(text, size, overlap):
    words = text.split()
    step = size - overlap
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) >= 30:  # skip tiny fragments
            yield " ".join(piece)


def main():
    files = [p for p in config.DATA_DIR.rglob("*") if p.is_file()]
    if not files:
        raise SystemExit(f"No files found in {config.DATA_DIR}/")

    model = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=config.DB_DIR)
    collection = client.get_or_create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    total = 0
    for path in files:
        ids, docs, metas = [], [], []
        for page_no, text in load_pages(path):
            for n, chunk in enumerate(
                chunk_words(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP)
            ):
                ids.append(f"{path.name}-p{page_no}-c{n}")
                docs.append(chunk)
                metas.append({"source": path.name, "page": page_no})
        if not docs:
            continue
        # e5 models expect the "passage: " prefix for documents
        embeddings = model.encode(
            [f"passage: {d}" for d in docs], normalize_embeddings=True
        ).tolist()
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        total += len(docs)
        print(f"{path.name}: {len(docs)} chunks")
    print(f"Done. {total} chunks indexed.")


if __name__ == "__main__":
    main()
