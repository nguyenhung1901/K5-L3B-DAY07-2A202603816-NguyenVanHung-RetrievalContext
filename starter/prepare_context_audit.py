"""Create an EMPTY manual context grading table for Day07 test answerable cases.

Rubric: grade 0 = no evidence; 1 = partial evidence or missing conditions/source;
2 = full sufficient evidence with valid source. This script never invents grades.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def prepare(cases: list[dict], traces: list[dict], split: str) -> list[dict]:
    answerable = {q['id']: q for q in cases if q.get('split') == split and q.get('answerable') is True}
    trace_by_id = {t['query_id']: t for t in traces}
    if len(trace_by_id) != len(traces):
        raise ValueError('duplicate query_id in trace')
    missing = answerable.keys() - trace_by_id.keys()
    if missing:
        raise ValueError(f'missing answerable traces: {sorted(missing)}')
    output = []
    for qid in sorted(answerable):
        t = trace_by_id[qid]
        output.append({
            'query_id': qid,
            'split': split,
            'method': t.get('method'),
            'agent_status': t.get('status'),
            'context_chars': len(t.get('final_context') or ''),
            'context_grade_0_1_2': '',
            'review_reason': '',
            'supporting_chunk_ids': '',
        })
    return output


def main():
    ap = argparse.ArgumentParser(description='Create blank manual context audit CSV; does not grade')
    ap.add_argument('--questions', type=Path, default=Path('benchmark/questions.json'))
    ap.add_argument('--traces', type=Path, required=True)
    ap.add_argument('--split', choices=['dev', 'test'], default='test')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        ap.error('Refusing to overwrite an existing manual audit file')
    cases = json.loads(a.questions.read_text(encoding='utf-8'))
    traces = json.loads(a.traces.read_text(encoding='utf-8'))
    result = prepare(cases, traces, a.split)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]) if result else [])
        writer.writeheader()
        writer.writerows(result)
    print(f'Wrote {len(result)} EMPTY context review rows to {a.output}; fill manually.')


if __name__ == '__main__':
    main()
