"""Create BLANK human-review labels from real agent traces; never invent grades.

The starter agent_metrics.py intentionally rejects incomplete reviews. Fill null values
only after reading the real answer, supporting contexts, and the source PDF.
"""
import argparse
import json
from pathlib import Path


def make_review(cases,traces):
    case_by_id={q['id']:q for q in cases}
    output=[]
    for t in traces:
        q=case_by_id[t['query_id']]
        if t.get('status')=='error':continue
        obj={
            'query_id':t['query_id'],
            'citation_support':[None]*len(t.get('citations',[])),
            'reason':'PENDING MANUAL REVIEW: check answer against actual cited chunks and PDF.',
        }
        if q['answerable']:
            # The rubric explicitly grades all false abstentions as 0.
            obj['answer_grade']=0 if t.get('status')=='abstain' else None
        else:
            obj['abstention_correct']=None
        output.append(obj)
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--questions',type=Path,default=Path('benchmark/questions.json'))
    p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output exists; choose another file to avoid overwriting manual reviews')
    cases=json.loads(args.questions.read_text('utf-8'))
    traces=json.loads(args.traces.read_text('utf-8'))
    caseids={q['id'] for q in cases}
    if any(t['query_id'] not in caseids for t in traces):p.error('Trace contains unknown query IDs')
    review=make_review(cases,traces)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(review)} INCOMPLETE review templates. Fill nulls manually before agent_metrics.py.')

if __name__=='__main__':main()
