"""Day07 retrieval comparison wrapper: excludes unanswerable cases from retrieval denominator.

Requires the previously working starter/retrieval_compare.py and baseline.py.
This script has an explicit test unlock; it writes no scores for the four unanswerables.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from baseline import validate
from retrieval_compare import Day6RetrievalMethods, evaluate_one


def main():
    p=argparse.ArgumentParser(description='Compare 3 retrieval methods on answerable Day07 cases only')
    p.add_argument('--env-file',type=Path)
    p.add_argument('--chunks',type=Path,default=Path('starter/data/converted/chunks.json'))
    p.add_argument('--questions',type=Path,default=Path('benchmark/questions.json'))
    p.add_argument('--split',choices=('dev','test'),default='dev')
    p.add_argument('--enable-test',action='store_true')
    p.add_argument('--freeze-file',type=Path,default=Path('artifacts/frozen_selection.json'))
    p.add_argument('--methods',choices=('all','dense','tfidf','hybrid'),default='all')
    p.add_argument('--budget',type=int,default=1600)
    p.add_argument('--top-k',type=int,default=3)
    p.add_argument('--candidate-k',type=int,default=12)
    p.add_argument('--use-query-filter',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.split=='test' and (not args.enable_test or not args.freeze_file.is_file()):
        p.error('Do not evaluate test until configuration is frozen: --enable-test and freeze file required')
    if args.split=='test':
        frozen=json.loads(args.freeze_file.read_text('utf-8'))
        if frozen.get('benchmark_sha256')!=hashlib.sha256(args.questions.read_bytes()).hexdigest():
            p.error('Benchmark differs from frozen version')
        if args.methods=='all':
            p.error('Run baseline and selected test separately with their frozen budgets')
        if args.methods==frozen.get('method'):
            expected_budget=frozen.get('budget')
        elif args.methods==frozen.get('baseline_method'):
            expected_budget=frozen.get('baseline_budget')
        else:
            p.error('Test method is neither frozen baseline nor frozen selected')
        if (args.budget!=expected_budget or args.top_k!=frozen.get('top_k')
                or args.candidate_k!=frozen.get('candidate_k')
                or args.use_query_filter!=frozen.get('use_query_filter')):
            p.error('Test parameters differ from the frozen selection')
    if args.budget<1 or args.top_k<1 or args.candidate_k<args.top_k:
        p.error('Invalid budget/top-k/candidate-k')
    chunks=json.loads(args.chunks.read_text('utf-8'))
    all_cases=json.loads(args.questions.read_text('utf-8'))
    raw=[q for q in all_cases if q['split']==args.split]
    selected=[q for q in raw if q.get('answerable',True)]
    if not selected:p.error('No answerable questions in selected split')
    validate(chunks,selected)
    methods=['dense','tfidf','hybrid'] if args.methods=='all' else [args.methods]
    start=time.perf_counter()
    with Day6RetrievalMethods(chunks,args.env_file,include_dense=any(m!='tfidf' for m in methods)) as engine:
        startup=time.perf_counter()-start
        summaries=[]
        for method in methods:
            traces,summary=evaluate_one(engine,selected,method=method,budget=args.budget,
                                        top_k=args.top_k,candidate_k=args.candidate_k,
                                        use_query_filter=args.use_query_filter)
            summary['excluded_unanswerable']=len(raw)-len(selected)
            summary['initialization_seconds_shared']=startup
            sub=args.output/method
            sub.mkdir(parents=True,exist_ok=True)
            (sub/'traces.json').write_text(json.dumps(traces,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            (sub/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            summaries.append(summary)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for s in summaries:
        print(f"{s['strategy']}: Hit@1={s['hit1']}/{s['n']} Hit@3={s['hit3']}/{s['n']} Context={s['context_hit']}/{s['n']} (unanswerable excluded: {s['excluded_unanswerable']})")
    print('No agent answer/citation scores were computed.')

if __name__=='__main__':main()
