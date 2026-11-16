"""Validate 24-case benchmark against exported strategy B chunks (read-only).

Strictly separates 20 answerable cases from four unanswerable cases.
Absence cannot be proved by keyword scanning: inspect corpus manually for each absence_reason.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from baseline import _gold_present_across_chunks, validate


def check(chunks, cases):
    if len(cases) != 24:
        raise ValueError(f'Expected exactly 24 cases, got {len(cases)}')
    ids=[q['id'] for q in cases]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate question ID')
    counts=Counter((q.get('split'),q.get('answerable')) for q in cases)
    expected={('dev',True):15,('dev',False):2,('test',True):5,('test',False):2}
    if counts != expected:
        raise ValueError(f'Split/answerability counts differ: {counts}')
    for split in ('dev','test'):
        answerable=[q for q in cases if q['split']==split and q['answerable']]
        validate(chunks,answerable)
    for q in cases:
        if not q['answerable']:
            if q.get('gold') != [] or not q.get('absence_reason') or not q.get('absence_audit'):
                raise ValueError(f"{q['id']}: unanswerable needs empty gold and documented absence review")
        elif not any(_gold_present_across_chunks(chunks,g) for g in q['gold']):
            raise ValueError(f"{q['id']}: answerable evidence is not supported by the chunks")
    return counts


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--chunks',type=Path,default=Path('starter/data/converted/chunks.json'))
    ap.add_argument('--questions',type=Path,default=Path('benchmark/questions.json'))
    args=ap.parse_args()
    chunks=json.loads(args.chunks.read_text('utf-8'))
    questions=json.loads(args.questions.read_text('utf-8'))
    counts=check(chunks,questions)
    print('PASS: 84 B chunks expected:',len(chunks))
    for split in ('dev','test'):
        print(f'{split}: answerable={counts[(split,True)]}; unanswerable={counts[(split,False)]}')
    print('All 20 answerable gold evidence items validated against matching source/page/chunks.')
    print('IMPORTANT: the 4 unanswerable items still require manual absence review over all 40 pages.')
    print('No retrieval test split was run.')

if __name__=='__main__':main()
