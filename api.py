"""Day 3: FastAPI wrapper around the RAG pipeline."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ask import answer

app = FastAPI(title="CiteRAG")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Question):
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is empty")
    text, hits = answer(q)
    return {
        "answer": text,
        "sources": [
            {"source": m["source"], "page": m["page"], "distance": round(dist, 3), "text": d}
            for d, m, dist in hits
        ],
    }
