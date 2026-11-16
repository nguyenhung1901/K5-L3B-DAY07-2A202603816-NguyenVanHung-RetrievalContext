"""Offline tests: validate adapter logic; never claim this is a real LLM benchmark."""
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'starter'))
from run_day07_agent import OllamaController
from agentic import run_agent
from prepare_day07_reviews import make_review
from sanitize_day07_traces import sanitize


class FakeResponse:
    def __init__(self, data): self.data=json.dumps(data).encode()
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self): return self.data


class FakeRetriever:
    def search(self, query, *, filters, top_k):
        return [{'id':'chunk-abc','text':'A LAN covers a small local area.',
                 'source_id':'Basics_Networking','source':'Basics_Networking.pdf','page':7,
                 'metadata':{'section':'module_01'},'score':0.9}][:top_k]


class AgentAdapterTests(unittest.TestCase):
    def test_24_question_split_without_leaking_previous_test(self):
        cases=json.loads((ROOT/'benchmark/questions.json').read_text('utf-8'))
        self.assertEqual(len(cases),24)
        self.assertEqual(sum(c['split']=='dev' and c['answerable'] for c in cases),15)
        self.assertEqual(sum(c['split']=='dev' and not c['answerable'] for c in cases),2)
        self.assertEqual(sum(c['split']=='test' and c['answerable'] for c in cases),5)
        self.assertEqual(sum(c['split']=='test' and not c['answerable'] for c in cases),2)
        self.assertTrue(all(c['id'] not in {'q16','q17','q18','q19','q20'} for c in cases))

    def test_native_ollama_tool_then_cited_answer(self):
        incoming=[]
        responses=[
            {'message':{'role':'assistant','content':'','tool_calls':[{'function':{
                'name':'search_day06','arguments':{'query':'LAN network definition'}}}]},
             'prompt_eval_count':50,'eval_count':30},
            {'message':{'role':'assistant','content':json.dumps({
                'action':'answer','answer':'LAN is local [chunk-abc]', 'citations':['chunk-abc']})},
             'prompt_eval_count':70,'eval_count':50},
        ]
        def mock_urlopen(request,timeout):
            req=json.loads(request.data)
            incoming.append(req)
            return FakeResponse(responses[len(incoming)-1])
        with patch('urllib.request.urlopen',side_effect=mock_urlopen):
            controller=OllamaController('unit-test-model','http://127.0.0.1:11434')
            got=run_agent('What is LAN?',controller,FakeRetriever(),budget=1500)
        self.assertEqual(got['status'],'answered')
        self.assertEqual(got['citations'],['chunk-abc'])
        self.assertEqual(len([e for e in got['events'] if e['event']=='search']),1)
        self.assertEqual(incoming[1]['messages'][-1]['role'],'tool')
        self.assertIn('A LAN covers',incoming[1]['messages'][-1]['content'])
        self.assertNotIn('expected_answer',json.dumps(incoming))
        self.assertNotIn('gold_page',json.dumps(incoming))
        self.assertNotIn('gold_source_id',json.dumps(incoming))

    def test_uncited_answer_abstains_without_fake_citation(self):
        responses=[
            {'message':{'tool_calls':[{'function':{'name':'search_day06',
                                                 'arguments':{'query':'network'}}}]}} ,
            {'message':{'content':json.dumps({'action':'answer','answer':'Not enough context','citations':[]})}},
        ]
        with patch('urllib.request.urlopen',side_effect=[FakeResponse(x) for x in responses]):
            ctrl=OllamaController('mock','http://127.0.0.1:11434')
            out=run_agent('Unknown parameter?',ctrl,FakeRetriever())
        self.assertEqual(out['status'],'abstain')
        self.assertEqual(out['citations'],[])

    def test_review_never_fabricates_grade(self):
        cases=[{'id':'one','answerable':True},{'id':'two','answerable':False}]
        traces=[{'query_id':'one','status':'answered','citations':['chunk-abc']},
                {'query_id':'two','status':'abstain','citations':[]}]
        reviews=make_review(cases,traces)
        self.assertIsNone(reviews[0]['answer_grade'])
        self.assertEqual(reviews[0]['citation_support'],[None])
        self.assertIsNone(reviews[1]['abstention_correct'])

    def test_public_trace_sanitization(self):
        traces=[{'query_id':'q1','events':[{'event':'search','context':'PRIVATE FULL TEXT',
               'results':[{'id':'c','text':'PRIVATE FULL TEXT','source_id':'s'}]}],
                'final_context':'PRIVATE FULL TEXT','answer':'a','citations':[]}]
        out=json.dumps(sanitize(traces))
        self.assertNotIn('PRIVATE FULL TEXT',out)

if __name__=='__main__':unittest.main()
