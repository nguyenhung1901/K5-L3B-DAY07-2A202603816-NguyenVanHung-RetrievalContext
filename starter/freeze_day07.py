"""Record a selection chosen on dev BEFORE evaluating test. No test running."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--questions',type=Path,default=Path('benchmark/questions.json'))
    p.add_argument('--method',choices=('tfidf','dense','hybrid'),default='tfidf')
    p.add_argument('--provider',choices=('gemini','ollama'),required=True)
    p.add_argument('--model',required=True,help='Exact model ID tested on dev')
    p.add_argument('--baseline-method',choices=('tfidf','dense','hybrid'),default='dense')
    p.add_argument('--baseline-budget',type=int,default=800)
    p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--candidate-k',type=int,default=12)
    p.add_argument('--budget',type=int,default=1600)
    p.add_argument('--max-searches',type=int,default=2)
    p.add_argument('--use-query-filter',action='store_true')
    p.add_argument('--output',type=Path,default=Path('artifacts/frozen_selection.json'))
    p.add_argument('--note',default='Selected from Day07 dev results; test must not be used for tuning.')
    args=p.parse_args()
    if args.output.exists():p.error('Freeze file already exists; do not silently change frozen test configuration')
    output={k:getattr(args,k) for k in ('method','provider','model','top_k','candidate_k','budget','max_searches','use_query_filter')}
    output['baseline_method']=args.baseline_method
    output['baseline_budget']=args.baseline_budget
    output['benchmark_sha256']=hashlib.sha256(args.questions.read_bytes()).hexdigest()
    output['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
    output['selection_note']=args.note
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'FROZEN: {args.output}; benchmark_sha256={output["benchmark_sha256"][:12]}')

if __name__=='__main__':main()
