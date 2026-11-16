"""Offline tests for Gemini REST adapter. Not an LLM benchmark."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from run_day07_agent import GeminiController
from agentic import run_agent
from summarize_agent_auto import summarize


class FakeResponse:
    def __init__(self, value):
        self.payload = json.dumps(value).encode('utf-8')
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return self.payload


class Retriever:
    def search(self, query, *, filters, top_k):
        return [{
            'id': 'network-p7-abc', 'text': 'A LAN covers a small area such as a home, office, or school.',
            'source_id': 'Basics_Networking', 'source': 'Basics_Networking.pdf',
            'page': 7, 'metadata': {'section': 'module_01'}, 'score': 0.91,
        }][:top_k]


class GeminiControllerTests(unittest.TestCase):
    def test_native_tool_response_keeps_thought_signature_and_id(self):
        responses = [
            {'candidates': [{'content': {'role': 'model', 'parts': [
                {'functionCall': {'id': 'call-one', 'name': 'search_day06',
                                  'args': {'query': 'LAN definition'}},
                 'thoughtSignature': 'opaque-thought-marker'}]}}],
             'usageMetadata': {'promptTokenCount': 40, 'candidatesTokenCount': 17, 'totalTokenCount': 57}},
            {'candidates': [{'content': {'role': 'model', 'parts': [
                {'text': json.dumps({'action': 'answer',
                    'answer': 'A LAN covers a small local area [network-p7-abc].',
                    'citations': ['network-p7-abc']})}]}}],
             'usageMetadata': {'promptTokenCount': 72, 'candidatesTokenCount': 31, 'totalTokenCount': 103}}
        ]
        sent = []
        def fake_open(req, timeout):
            sent.append((req.full_url, dict(req.headers), json.loads(req.data)))
            return FakeResponse(responses[len(sent) - 1])
        with patch('urllib.request.urlopen', side_effect=fake_open):
            ctrl = GeminiController('gemini-test', 'do-not-share-unit-key')
            result = run_agent('Define LAN', ctrl, Retriever(), max_searches=2, budget=1600)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['citations'], ['network-p7-abc'])
        self.assertEqual(len([x for x in result['events'] if x['event'] == 'search']), 1)
        self.assertIn('x-goog-api-key', {k.lower() for k in sent[0][1]})
        self.assertNotIn('do-not-share-unit-key', sent[0][0])
        self.assertEqual(sent[0][2]['toolConfig']['functionCallingConfig']['mode'], 'ANY')
        self.assertEqual(sent[1][2]['toolConfig']['functionCallingConfig']['mode'], 'AUTO')
        parts = sent[1][2]['contents']
        self.assertEqual(parts[1]['parts'][0]['thoughtSignature'], 'opaque-thought-marker')
        self.assertEqual(parts[-1]['parts'][0]['functionResponse']['id'], 'call-one')
        self.assertIn('A LAN covers a small area', parts[-1]['parts'][0]['functionResponse']['response']['context'])
        self.assertEqual(sum((x['prompt_tokens'] or 0) for x in ctrl.calls), 112)
        self.assertNotIn('expected_answer', json.dumps(sent))
        self.assertNotIn('\"gold\":', json.dumps(sent))

    def test_no_evidence_abstains_without_fake_citation(self):
        rows = [
            {'candidates': [{'content': {'parts': [{'functionCall': {
                'name': 'search_day06', 'args': {'query': 'network standards'}}}]}}]},
            {'candidates': [{'content': {'parts': [{'text': json.dumps({
                'action': 'answer', 'answer': 'Insufficient evidence in the retrieved document.',
                'citations': []})}]}}]},
        ]
        with patch('urllib.request.urlopen', side_effect=[FakeResponse(x) for x in rows]):
            ctrl = GeminiController('gemini-test', 'unit-key')
            out = run_agent('Is this exact fact in the corpus?', ctrl, Retriever(), max_searches=2)
        self.assertEqual(out['status'], 'abstain')
        self.assertEqual(out['citations'], [])

    def test_no_api_key_rejected(self):
        with self.assertRaisesRegex(ValueError, 'GEMINI_API_KEY'):
            GeminiController('gemini-test', '')

    def test_gemini_cost_never_assumed_free(self):
        traces = [{'query_id': 'q1', 'status': 'answered', 'elapsed_ms': 100,
                   'search_count': 1, 'prompt_tokens_total': 100, 'completion_tokens_total': 20,
                   'service_cost_usd': None}]
        result = summarize(traces, [{'id': 'q1', 'answerable': True}])
        self.assertIsNone(result['service_cost_usd_recorded'])
        self.assertEqual(result['prompt_tokens_total'], 100)

if __name__ == '__main__':
    unittest.main()
