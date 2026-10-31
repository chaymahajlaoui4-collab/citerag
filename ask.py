"""Day 2-3: retrieve the best chunks and answer with Groq, citing sources."""
import os
import sys

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

import config

load_dotenv()

SYSTEM_PROMPT = (
    "You are a technical documentation assistant. Answer ONLY using facts that are "
    "explicitly stated in the provided context. Do NOT use outside knowledge. "
    "A link or a reference to another page is NOT an answer: if the context only "
    "points to a page without explaining the content, say you don't know. "
    "Cite every claim as [source, p.X]. If the context does not contain the answer, "
    "reply exactly: \"Je ne sais pas : l'information n'est pas dans les documents.\" "
    "(in the language of the question). Reply in the same language as the question."

)

_model = SentenceTransformer(config.EMBED_MODEL)
_collection = chromadb.PersistentClient(path=config.DB_DIR).get_collection(config.COLLECTION)
_llm = Groq(api_key=os.environ["GROQ_API_KEY"])


def retrieve(question, k=config.TOP_K):
    # e5 models expect the "query: " prefix for questions
    emb = _model.encode([f"query: {question}"], normalize_embeddings=True).tolist()
    res = _collection.query(query_embeddings=emb, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def answer(question):
    hits = retrieve(question)
    context = "\n\n".join(
        f"[{m['source']}, p.{m['page']}]\n{d}" for d, m, _ in hits
    )
    completion = _llm.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return completion.choices[0].message.content, hits


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or input("Question: ")
    text, hits = answer(q)
    print("\n" + text)
    print("\n--- Retrieved chunks (check these by hand) ---")
    for d, m, dist in hits:
        print(f"{m['source']} p.{m['page']} (distance {dist:.3f}): {d[:100]}...")

