from pathlib import Path

DATA_DIR = Path("data")          # put your PDF / MD / TXT files here
DB_DIR = "chroma_db"
COLLECTION = "docs"

# Multilingual (FR/EN), light model. Upgrade later: "BAAI/bge-m3" (heavier, better)
EMBED_MODEL = "intfloat/multilingual-e5-small"

# Chunking: parameters to experiment with on day 4 (e.g. 150 / 300 / 500 words)
CHUNK_WORDS = 300
CHUNK_OVERLAP = 50

TOP_K = 5
# Check the current model names in the Groq console if this one is unavailable
LLM_MODEL = "openai/gpt-oss-120b"
