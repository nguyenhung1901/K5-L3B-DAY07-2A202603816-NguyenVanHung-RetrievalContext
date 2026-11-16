"""Day 07 read-only bridge to the existing Day 06 pgvector database.

Usage from the Day 07 repository root (PowerShell):
    python starter/bridge_day6.py --env-file "D:\\AI_Action\\...DAY06...\\starter\\.env" check
    python starter/bridge_day6.py --env-file "D:\\AI_Action\\...DAY06...\\starter\\.env" export --output starter/data/day6_chunks_B.json
    python starter/bridge_day6.py --env-file "D:\\AI_Action\\...DAY06...\\starter\\.env" search --question "What does a firewall do?"

Read only: never creates tables, inserts rows, or modifies the Day 06 database.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values

TABLE = "public.day06_chunks"
EXPECTED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384


def db_url(env_file: Path) -> str:
    if not env_file.is_file():
        raise FileNotFoundError(f"Day 06 .env not found: {env_file}")
    settings = dotenv_values(env_file)
    url = settings.get("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL is missing from Day 06 .env")
    model = settings.get("EMBEDDING_MODEL")
    if model and model != EXPECTED_MODEL:
        raise ValueError(
            f"Configured Day 06 embedding model is {model!r}, but this bridge "
            f"expects {EXPECTED_MODEL!r}. Confirm the actual indexing model first."
        )
    return url


def inspect(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT strategy, COUNT(*) AS total, COUNT(DISTINCT chunk_id) AS unique_chunks "
            f"FROM {TABLE} GROUP BY strategy ORDER BY strategy"
        )
        rows = cur.fetchall()
        print("Day 06 stored chunks:")
        for row in rows:
            print(f"  {row['strategy']}: total={row['total']}, unique_ids={row['unique_chunks']}")
        if not any(row["strategy"] == "B" for row in rows):
            raise RuntimeError("No strategy B data found. Do not reindex automatically.")


def export_chunks(conn: psycopg.Connection, out_file: Path) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT chunk_id, document_id, source, page, section, text, metadata "
            f"FROM {TABLE} WHERE strategy = %s ORDER BY chunk_id",
            ("B",),
        )
        chunks = [dict(row) for row in cur.fetchall()]
    if not chunks:
        raise RuntimeError("Strategy B has no chunks; refusing to export an empty corpus")
    if len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise RuntimeError("Duplicate chunk IDs in strategy B")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(chunks)} strategy B chunks to {out_file}")
    print("This JSON contains source text; keep it local unless redistribution is permitted.")


def dense_search(conn: psycopg.Connection, question: str, top_k: int, section: str | None) -> None:
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EXPECTED_MODEL, device="cpu")
    vector = model.encode(question, normalize_embeddings=True, convert_to_numpy=True).tolist()
    if len(vector) != DIMENSIONS or not all(math.isfinite(v) for v in vector):
        raise RuntimeError(f"Expected a valid {DIMENSIONS}-dimensional query vector")
    literal = "[" + ",".join(format(float(v), ".10g") for v in vector) + "]"
    where = "strategy = %s"
    args = [literal, "B"]
    if section:
        where += " AND section = %s"
        args.append(section)
    args.extend([literal, top_k])
    sql = (
        f"SELECT chunk_id, document_id, page, section, source, "
        f"1.0 - (embedding <=> %s::vector) AS score, "
        f"LEFT(text, 140) AS preview FROM {TABLE} WHERE {where} "
        f"ORDER BY embedding <=> %s::vector ASC, chunk_id ASC LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, args)
        results = cur.fetchall()
    if not results:
        print("No matches for the requested section.")
    for rank, row in enumerate(results, 1):
        preview = " ".join(row["preview"].split())
        print(f"{rank}. {row['chunk_id']} | p{row['page']} | {row['section']} | "
              f"cosine similarity={float(row['score']):.4f}")
        print(f"   {preview}")


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only bridge from Day 07 to existing Day 06 pgvector")
    p.add_argument("--env-file", required=True, type=Path, help="Path to Day 06 starter/.env")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Check Day 06 row counts")
    export = commands.add_parser("export", help="Export only strategy B chunk text and metadata")
    export.add_argument("--output", type=Path, default=Path("starter/data/day6_chunks_B.json"))
    search = commands.add_parser("search", help="Smoke test dense retrieval, without modifying index")
    search.add_argument("--question", required=True)
    search.add_argument("--top-k", type=int, default=3)
    search.add_argument("--section", help="Optional trusted section, e.g. module_02")
    a = p.parse_args()
    with psycopg.connect(db_url(a.env_file), connect_timeout=10, row_factory=dict_row) as conn:
        if a.command == "check":
            inspect(conn)
        elif a.command == "export":
            export_chunks(conn, a.output)
        else:
            dense_search(conn, a.question, a.top_k, a.section)


if __name__ == "__main__":
    main()
