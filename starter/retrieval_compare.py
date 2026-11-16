"""Day 07: read-only comparison of dense / TF-IDF / hybrid RRF on Day 06 B chunks.

Copy this file to Day07/starter/retrieval_compare.py. From Day07 root:
    python starter/retrieval_compare.py --env-file <Day06>/starter/.env \
      --chunks starter/data/converted/chunks.json \
      --questions starter/data/converted/questions.json \
      --split dev --output starter/results/dev-retrieval-compare

Requires baseline.py (the Day06-compatible Day07 version), bridge_day6.py,
psycopg[binary], python-dotenv and sentence-transformers. No DB writes.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from baseline import TfidfIndex, _gold_present_across_chunks, hit, pack_context, validate


def _choose_section(filters: dict[str, Any] | None) -> str | None:
    if not filters:
        return None
    if set(filters) != {'section'} or not isinstance(filters['section'], str):
        raise ValueError("This experiment supports only explicit {'section': 'module_XX'} filters")
    return filters['section']


class Day6RetrievalMethods:
    def __init__(self, chunks: list[dict[str, Any]], env_file: Path | None = None, *,
                 include_dense: bool = True):
        self.chunks = chunks
        self.by_id = {c['id']: c for c in chunks}
        if len(self.by_id) != len(chunks):
            raise ValueError('Duplicate chunk IDs')
        self.lexical = TfidfIndex(chunks)
        self.conn = None
        self.model = None
        if include_dense:
            if env_file is None:
                raise ValueError('--env-file is required when using dense or hybrid')
            import psycopg
            from psycopg.rows import dict_row
            from bridge_day6 import EXPECTED_MODEL, TABLE, db_url
            self.table = TABLE
            self.expected_model = EXPECTED_MODEL
            self.conn = psycopg.connect(db_url(env_file), connect_timeout=10, row_factory=dict_row)
            try:
                self._verify_same_index()
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(EXPECTED_MODEL, device='cpu')
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _verify_same_index(self) -> None:
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT chunk_id FROM {self.table} WHERE strategy = %s", ('B',))
            db_ids = {r['chunk_id'] for r in cur.fetchall()}
        json_ids = set(self.by_id)
        if db_ids != json_ids:
            raise RuntimeError(
                f"Day06 strategy B IDs differ from local JSON: "
                f"in_DB_not_JSON={len(db_ids-json_ids)}, "
                f"in_JSON_not_DB={len(json_ids-db_ids)}. "
                "Re-export and reconvert B chunks before comparing methods."
            )

    def lexical_search(self, query: str, *, top_k: int, section: str | None = None) -> list[dict[str, Any]]:
        # For filtered lexical retrieval, apply filtering BEFORE truncating to top_k.
        all_ranked = self.lexical.search(query, top_k=len(self.chunks) if section else top_k)
        if section:
            all_ranked = [c for c in all_ranked if c.get('metadata', {}).get('section') == section
                          or c.get('section') == section]
        return all_ranked[:top_k]

    def dense_search(self, query: str, *, top_k: int, section: str | None = None) -> list[dict[str, Any]]:
        if self.model is None or self.conn is None:
            raise RuntimeError('Dense model / PostgreSQL connection not initialized')
        vector = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True).tolist()
        if len(vector) != 384 or not all(math.isfinite(float(x)) for x in vector):
            raise RuntimeError('Invalid or incompatible query embedding dimensions')
        literal = '[' + ','.join(format(float(x), '.10g') for x in vector) + ']'
        where = 'strategy = %s'
        params: list[Any] = [literal, 'B']
        if section:
            where += ' AND section = %s'
            params.append(section)
        params.extend([literal, top_k])
        sql = (
            f'SELECT chunk_id, document_id, source, page, section, text, metadata, '
            f'1.0 - (embedding <=> %s::vector) AS score '
            f'FROM {self.table} WHERE {where} '
            f'ORDER BY embedding <=> %s::vector ASC, chunk_id ASC LIMIT %s'
        )
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out = []
        for r in rows:
            local = self.by_id.get(r['chunk_id'])
            if local is None:
                raise RuntimeError('Dense query returned a chunk absent from the JSON corpus')
            if r['document_id'] != local['source_id'] or r['page'] != local['page'] or r['text'] != local['text']:
                raise RuntimeError(f"DB/JSON contents mismatch for chunk {r['chunk_id']}")
            out.append({**local, 'score': float(r['score']), 'strategy': 'dense'})
        return out

    def search(self, query: str, *, method: str, top_k: int = 3, candidate_k: int = 12,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1 or candidate_k < top_k:
            raise ValueError('Nonempty query and candidate_k >= top_k >= 1 required')
        section = _choose_section(filters)
        if method == 'tfidf':
            return self.lexical_search(query, top_k=top_k, section=section)
        if method == 'dense':
            return self.dense_search(query, top_k=top_k, section=section)
        if method != 'hybrid':
            raise ValueError(f'Unknown method {method!r}')
        dense = self.dense_search(query, top_k=candidate_k, section=section)
        lexical = self.lexical_search(query, top_k=candidate_k, section=section)
        # Reciprocal Rank Fusion: 1/(k + rank) for each ranked list. Do not add
        # raw cosine and TF-IDF scores; they are not calibrated to each other.
        rrf_k = 60
        scores: dict[str, float] = {}
        records: dict[str, dict[str, Any]] = {}
        rank_details: dict[str, dict[str, int]] = {}
        for method_name, candidates in [('dense', dense), ('tfidf', lexical)]:
            for rank, c in enumerate(candidates, 1):
                cid = c['id']
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
                records[cid] = self.by_id[cid]
                rank_details.setdefault(cid, {})[method_name] = rank
        ordering = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top_k]
        return [{**records[cid], 'score': scores[cid], 'strategy': 'hybrid_rrf',
                 'ranks': rank_details[cid]} for cid in ordering]


def evaluate_one(retriever: Day6RetrievalMethods, questions: list[dict[str, Any]], *,
                 method: str, budget: int, top_k: int, candidate_k: int,
                 use_query_filter: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for q in questions:
        filters = q.get('filter') if use_query_filter else None
        start = time.perf_counter()
        result = retriever.search(q['question'], method=method, top_k=top_k,
                                  candidate_k=candidate_k, filters=filters)
        context, ids = pack_context(result, budget=budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**r, 'hit': hit(r, q['gold'])} for r in result]
        packed = [r for r in result if r['id'] in ids]
        traces.append({
            'query_id': q['id'], 'split': q['split'], 'question': q['question'],
            'strategy': method, 'top_k': top_k, 'candidate_k': candidate_k,
            'budget_chars': budget, 'filters': filters, 'gold': q['gold'],
            'results': marked, 'context': context, 'context_ids': ids,
            'context_chars': len(context), 'elapsed_ms': elapsed_ms,
            'hit1': bool(marked and marked[0]['hit']),
            'hit3': any(c['hit'] for c in marked),
            'context_hit': any(_gold_present_across_chunks(packed, g) for g in q['gold']),
        })
    summary = {
        'strategy': method, 'split': questions[0]['split'] if questions else None,
        'n': len(traces), 'hit1': sum(t['hit1'] for t in traces),
        'hit3': sum(t['hit3'] for t in traces),
        'context_hit': sum(t['context_hit'] for t in traces),
        'avg_context_chars': statistics.mean(t['context_chars'] for t in traces) if traces else 0,
        'median_ms': statistics.median(t['elapsed_ms'] for t in traces) if traces else 0,
        'top_k': top_k, 'candidate_k': candidate_k,
        'budget_chars': budget, 'use_query_filter': use_query_filter,
        'service_cost_usd': 0,
    }
    return traces, summary


def main():
    parser = argparse.ArgumentParser(description='Day07 compare Dense / TF-IDF / Hybrid on one fixed Day06 B corpus')
    parser.add_argument('--env-file', type=Path, help='Day06 starter/.env (for dense/hybrid)')
    parser.add_argument('--chunks', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--enable-test', action='store_true', help='Must explicitly opt in to test evaluation')
    parser.add_argument('--methods', choices=['all', 'dense', 'tfidf', 'hybrid'], default='all')
    parser.add_argument('--budget', type=int, default=800, help='Context size in characters')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--candidate-k', type=int, default=12, help='Candidates per retriever for RRF')
    parser.add_argument('--use-query-filter', action='store_true', help='Use explicit benchmark query filters for ALL methods')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.split == 'test' and not args.enable_test:
        parser.error('Test split requires --enable-test, and should only be used after dev selection is frozen')
    if args.budget < 1 or args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error('budget>0, top-k>0, candidate-k>=top-k are required')
    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    selected = [q for q in questions if q.get('split') == args.split]
    validate(chunks, selected)  # Do not validate the other split.
    methods = ['dense', 'tfidf', 'hybrid'] if args.methods == 'all' else [args.methods]
    started = time.perf_counter()
    with Day6RetrievalMethods(chunks, args.env_file,
                              include_dense=any(m != 'tfidf' for m in methods)) as retriever:
        startup_seconds = time.perf_counter() - started
        summaries = []
        for method in methods:
            traces, summary = evaluate_one(
                retriever, selected, method=method, budget=args.budget,
                top_k=args.top_k, candidate_k=args.candidate_k,
                use_query_filter=args.use_query_filter
            )
            summary['initialization_seconds_shared'] = startup_seconds
            subdir = args.output / method
            subdir.mkdir(parents=True, exist_ok=True)
            (subdir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            (subdir / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            summaries.append(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for s in summaries:
        print(f"{s['strategy']:6} | Hit@1 {s['hit1']}/{s['n']} | "
              f"Hit@3 {s['hit3']}/{s['n']} | Context {s['context_hit']}/{s['n']} | "
              f"median {s['median_ms']:.2f} ms")
    print(f'Outputs: {args.output}')
    print('Note: automated evidence and context checks must be manually audited; no LLM agent was run.')


if __name__ == '__main__':
    main()
