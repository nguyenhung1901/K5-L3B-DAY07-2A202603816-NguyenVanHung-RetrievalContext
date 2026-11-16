"""Remove source texts and prompts from local traces before optional public sharing.

This produces limited evidence metadata, not a substitute for full local audit.
"""
import argparse
import json
from pathlib import Path


def sanitize(traces):
    result=[]
    for t in traces:
        out={k:t.get(k) for k in ('query_id','split','question','method','model',
                                  'status','answer','citations','search_count','elapsed_ms',
                                  'prompt_tokens_total','completion_tokens_total','service_cost_usd')}
        events=[]
        for event in t.get('events',[]):
            e={k:v for k,v in event.items() if k not in ('results','context')}
            if event.get('event')=='search':
                e['results']=[{k:r.get(k) for k in ('id','source_id','source','page','metadata','score','strategy')}
                              for r in event.get('results',[])]
            events.append(e)
        out['events']=events
        result.append(out)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--traces',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Output exists; choose new path')
    raw=json.loads(args.traces.read_text('utf-8'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(sanitize(raw),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Wrote {len(raw)} sanitized traces. Review question/answer text and other metadata before sharing.')

if __name__=='__main__':main()
