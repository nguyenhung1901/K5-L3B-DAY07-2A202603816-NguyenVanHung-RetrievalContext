"""TF-IDF retrieval baseline for Day 7. Python 3.11+, standard library only."""
import argparse
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from itertools import permutations
from pathlib import Path


def tokenize(text):
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def _normalize_evidence(text):
    """Normalize only Unicode and whitespace; never alter substantive evidence."""
    return ' '.join(unicodedata.normalize('NFC', text).split())


def hit(chunk, gold):
    """Single-chunk evidence hit: keep Hit@1/Hit@3 strict and comparable."""
    content = _normalize_evidence(chunk['text'])
    return any(
        chunk['source_id'] == g['source_id']
        and chunk['page'] == g['page']
        and _normalize_evidence(g['evidence']) in content
        for g in gold
    )


def _gold_present_across_chunks(chunks, gold):
    """Check a full quote on the same source page, including a chunk boundary.

    This is for dataset validation and CONTEXT evidence only; not for Hit@1/3.
    Chunk IDs exported from Day06 contain a hash instead of source order, so
    consider orders of same-page chunks when verifying the contiguous quote.
    Return False when no exact normalized quote can be reconstructed.
    """
    if any(hit(c, [gold]) for c in chunks):
        return True
    page_chunks = [c for c in chunks
                   if c['source_id'] == gold['source_id'] and c['page'] == gold['page']]
    if len(page_chunks) < 2:
        return False
    target = _normalize_evidence(gold['evidence'])
    # In the Day06 B corpus a page contains few chunks. Avoid factorial blowups
    # for unexpected inputs instead of incorrectly declaring a hit.
    if len(page_chunks) > 7:
        return False
    for ordered in permutations(page_chunks):
        together = ' '.join(_normalize_evidence(c['text']) for c in ordered)
        if target in together:
            return True
    return False


def validate(chunks, questions):
    if not isinstance(chunks, list) or not chunks:
        raise ValueError('chunks phải là JSON array không rỗng')
    if not isinstance(questions, list) or not questions:
        raise ValueError('questions phải là JSON array không rỗng')
    ids = set()
    for c in chunks:
        for key in ('id', 'text', 'source_id', 'source', 'page', 'metadata'):
            if key not in c or c[key] is None:
                raise ValueError(f'chunk thiếu {key}')
        if not all(isinstance(c[k], str) and c[k].strip() for k in ('id', 'text', 'source_id', 'source')):
            raise ValueError('chunk id/text/source_id/source phải là chuỗi không rỗng')
        if not isinstance(c['metadata'], dict) or c['id'] in ids:
            raise ValueError('metadata phải là object và ID chunk phải duy nhất')
        ids.add(c['id'])
    ids.clear()
    for q in questions:
        if not isinstance(q.get('id'), str) or not q['id'].strip() or q['id'] in ids:
            raise ValueError('question id phải là chuỗi không rỗng và duy nhất')
        ids.add(q['id'])
        if q.get('split') not in ('dev', 'test') or not isinstance(q.get('question'), str) or not q['question'].strip():
            raise ValueError(f"question {q['id']}: split hoặc question không hợp lệ")
        if not isinstance(q.get('expected_answer'), str) or not q['expected_answer'].strip():
            raise ValueError(f"question {q['id']}: expected_answer phải là chuỗi không rỗng")
        if not isinstance(q.get('gold'), list) or not q['gold']:
            raise ValueError(f"question {q['id']}: gold phải là danh sách không rỗng")
        for g in q['gold']:
            if not all(key in g and g[key] is not None for key in ('source_id', 'page', 'evidence')) or not isinstance(g['evidence'], str) or not g['evidence']:
                raise ValueError(f"question {q['id']}: gold không hợp lệ")
            if not _gold_present_across_chunks(chunks, g):
                raise ValueError(f"question {q['id']}: gold không xác minh được trên chunk cùng nguồn/trang")


class TfidfIndex:
    def __init__(self, chunks):
        self.chunks = chunks
        counts = [Counter(tokenize(c['text'])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {term: math.log((1 + len(chunks)) / (1 + count)) + 1 for term, count in df.items()}
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts):
        vector = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1
        return {term: value / norm for term, value in vector.items()}

    def search(self, question, top_k=3):
        query = self._vector(Counter(tokenize(question)))
        results = []
        for chunk, vector in zip(self.chunks, self.vectors):
            score = sum(value * vector.get(term, 0) for term, value in query.items())
            if score > 0:
                results.append({**chunk, 'score': score, 'strategy': 'tfidf'})
        return sorted(results, key=lambda c: (-c['score'], c['id']))[:top_k]


def pack_context(results, budget=800):
    blocks, ids, seen = [], [], set()
    for c in results:
        if c['id'] in seen:
            continue
        seen.add(c['id'])
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(c['id'])
    return '\n\n'.join(blocks), ids


def evaluate(index, questions, budget=800):
    traces = []
    for q in questions:
        start = time.perf_counter()
        results = index.search(q['question'], top_k=3)
        context, ids = pack_context(results, budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        traces.append({'query_id': q['id'], 'split': q['split'], 'question': q['question'],
                       'strategy': 'tfidf', 'top_k': 3, 'budget_chars': budget,
                       'gold': q['gold'], 'results': marked, 'context': context, 'context_ids': ids,
                       'context_chars': len(context), 'elapsed_ms': elapsed_ms,
                       'hit1': bool(marked and marked[0]['hit']),
                       'hit3': any(c['hit'] for c in marked),
                       'context_hit': any(_gold_present_across_chunks([c for c in results if c['id'] in ids], g) for g in q['gold'])})
    return traces


def main():
    parser = argparse.ArgumentParser(description='Chạy baseline TF-IDF trên một split rõ ràng')
    parser.add_argument('--chunks', required=True, type=Path)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--output', required=True, type=Path, help='Thư mục output; file cùng tên sẽ bị ghi đè')
    args = parser.parse_args()
    chunks = json.loads(args.chunks.read_text(encoding='utf-8'))
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    # Do not inspect held-out test evidence during dev work.
    selected = [q for q in questions if q.get('split') == args.split]
    validate(chunks, selected)
    started = time.perf_counter()
    index = TfidfIndex(chunks)
    index_seconds = time.perf_counter() - started
    traces = evaluate(index, selected)
    summary = {'strategy': 'tfidf', 'split': args.split, 'n': len(traces),
               'hit1': sum(t['hit1'] for t in traces), 'hit3': sum(t['hit3'] for t in traces),
               'context_hit': sum(t['context_hit'] for t in traces),
               'avg_context_chars': statistics.mean(t['context_chars'] for t in traces),
               'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
               'index_seconds': index_seconds, 'service_cost_usd': 0}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, obj in (('traces.json', traces), ('summary.json', summary)):
        (args.output / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
