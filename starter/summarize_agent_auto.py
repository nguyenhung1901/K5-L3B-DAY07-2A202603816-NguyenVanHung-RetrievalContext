"""Auto-count observable agent outcomes WITHOUT pretending to grade grounding/citations."""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def summarize(traces, cases):
    indexed={q['id']:q for q in cases}
    selected=[t for t in traces if t['query_id'] in indexed]
    if not selected:raise ValueError('No matching traces')
    successful=[t for t in selected if t.get('status')!='error']
    costs=[t.get('service_cost_usd') for t in selected]
    known_cost=all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in costs)
    return {
        'n_cases_seen':len(selected),
        'answerable_count':sum(bool(indexed[t['query_id']]['answerable']) for t in selected),
        'unanswerable_count':sum(not indexed[t['query_id']]['answerable'] for t in selected),
        'status_counts':dict(Counter(t.get('status') for t in selected)),
        'successful_queries':len(successful),
        'queries_with_retrieval_tool':sum(t.get('search_count',0)>0 for t in successful),
        'search_call_distribution':dict(Counter(str(t.get('search_count',0)) for t in successful)),
        'mean_search_calls':statistics.mean(t.get('search_count',0) for t in successful) if successful else None,
        'median_end_to_end_ms':statistics.median(t['elapsed_ms'] for t in successful) if successful else None,
        'prompt_tokens_total':sum(t.get('prompt_tokens_total',0) for t in selected),
        'completion_tokens_total':sum(t.get('completion_tokens_total',0) for t in selected),
        'service_cost_usd_recorded':sum(costs) if known_cost else None,
        'service_cost_note':('Recorded API cost summed.' if known_cost else
                            'Gemini billing cost not estimated; use token totals and current tier pricing.'),
        'not_graded': ['grounded_answer', 'citation_validity', 'abstention_correct', 'false_abstention_correct']
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--questions',type=Path,default=Path('benchmark/questions.json'))
    p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--split',choices=('dev','test'),default='dev')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.traces.read_text('utf-8'))
    cases=[q for q in json.loads(args.questions.read_text('utf-8')) if q['split']==args.split]
    result=summarize(data,cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
