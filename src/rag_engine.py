"""
rag_engine.py
-------------

Desc: Handles document loading, chunking, embedding, SQLite storage, and retrieval for the mini-RAG pipeline.

"""



import os
import json
import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"     
CHUNK_SIZE   = 400                   
CHUNK_OVERLAP = 80                   
TOP_K        = 3                     
DB_PATH      = "rag_store.db"


# ---------------------------------------------------------------------------
# Embedding 
# ---------------------------------------------------------------------------

def get_embedder():
    '''This function initialises embedding model'''
    
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    from langchain_huggingface import HuggingFaceEmbeddings

    embedder = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"device": "cpu"},        
                encode_kwargs={"normalize_embeddings": True}, 
                )
    
    return embedder


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

#to create sqlite connetion obj
def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    '''This function returns a database connection object'''

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    print(conn)
    return conn


#to initialise chunk & query_cache tables in sqlite db
def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist."""

    conn = get_connection(db_path)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id  INTEGER  PRIMARY KEY  AUTOINCREMENT,
            doc_name  TEXT  NOT NULL,
            chunk_index  INTEGER  NOT NULL,
            content  TEXT  NOT NULL,
            content_hash  TEXT  NOT NULL UNIQUE,
            embedding  BLOB  NOT NULL
        );

        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash  TEXT  PRIMARY KEY,
            query_text  TEXT  NOT NULL,
            result_json TEXT  NOT NULL,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sessions (
            user_id     TEXT PRIMARY KEY,
            history     TEXT NOT NULL DEFAULT '[]'
        );               
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialised.")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_documents(docs_dir: str = "Knowledge Base", db_path: str = DB_PATH) -> int:
    """
    Read all .md files in docs_dir, chunk them, embed, and store.
    Skips chunks that are already in the DB.
    Returns the number of new chunks inserted.
    """

    init_db(db_path)
    conn = get_connection(db_path)
    model = get_embedder()
    inserted = 0

    for fpath in sorted(Path(docs_dir).glob("**/*")): 
        if fpath.suffix not in {".md", ".txt"}:
            continue
        doc_name = fpath.name
        text = fpath.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            content_hash = hashlib.sha256(chunk.encode()).hexdigest()

            #check if chunk already stored
            exists = conn.execute(
                f"SELECT 1 FROM chunks WHERE content_hash = ?", (content_hash,)
            ).fetchone()

            if exists:
                logger.info("Chunk is already embedded")
                continue

            vec = np.array(model.embed_documents([chunk])[0], dtype=np.float32)
            
            conn.execute(
                """INSERT INTO chunks (doc_name, chunk_index, content, content_hash, embedding)
                   VALUES (?, ?, ?, ?, ?)""",
                (doc_name, idx, chunk, content_hash, vec.tobytes())
            )
            inserted += 1
            logger.info(f"Indexed chunk {idx} from {doc_name}")

        if chunks:
            logger.info(f"Processed {doc_name}: {len(chunks)} chunks")

    conn.commit()
    conn.close()
    logger.info(f"Indexing complete. New chunks inserted: {inserted}")
    return inserted


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # both are already normalised

def retrieve(query: str, top_k: int = TOP_K, db_path: str = DB_PATH) -> list[dict]:
    """
    Returns top_k most relevant chunks for the query.
    Uses a query cache to avoid re-embedding identical queries.
    Steps: Normalise the query -> check cache -> if not cached, embed the query and score every chunk by similarity -> return top 3 -> save result to cache.
    
    """
    query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()
    conn = get_connection(db_path)

    #cache
    cached = conn.execute(
        "SELECT result_json FROM query_cache WHERE query_hash = ?", (query_hash,)
    ).fetchone()
    if cached:
        logger.info("Cache available for query.")
        conn.close()
        return json.loads(cached["result_json"])

    #embed the query
    logger.info("Cache not available for query, craeting new embeddings")
    model = get_embedder()
    q_vec = np.array(model.embed_query(query), dtype=np.float32)

    #score all chunks
    rows = conn.execute("SELECT id, doc_name, content, embedding FROM chunks").fetchall()
    scored = []
    for row in rows:
        chunk_vec = np.frombuffer(row["embedding"], dtype=np.float32)
        score = cosine_similarity(q_vec, chunk_vec)
        scored.append({
            "doc_name": row["doc_name"],
            "content":  row["content"],
            "score":    round(score, 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]
    logger.info(f'Retrieved Context: {results}')

    #store in cache
    conn.execute(
        "INSERT INTO query_cache (query_hash, query_text, result_json) VALUES (?, ?, ?)",
        (query_hash, query, json.dumps(results))
    )
    conn.commit()
    conn.close()

    return results


# ---------------------------------------------------------------------------
# Context builder (used by the LLM prompt)
# ---------------------------------------------------------------------------
def build_context(chunks: list[dict]) -> tuple[str, list[str]]:
    """
    Returns (context_text, source_list).
    context_text is retrieved snippets.
    source_list is the list of source doc names.
    """
    lines, sources = [], []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] (from {chunk['doc_name']}, relevance {chunk['score']})\n{chunk['content']}")
        if chunk["doc_name"] not in sources:
            sources.append(chunk["doc_name"])
    return "\n\n".join(lines), sources




#----------------------------------END---------------------------------------
