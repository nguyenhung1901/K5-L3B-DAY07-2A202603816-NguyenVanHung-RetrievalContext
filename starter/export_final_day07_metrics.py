"""Export public aggregate Day07 metrics from frozen LOCAL experiment outputs.

No Gemini API, DB connection, source text, trace, answer, citation or secret is exported.
Expected to be run from the project root after the local experiment and audit.
"""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required local artifact: {path}")
    return json.loads(path.read_text(encoding='utf-8-sig'))


def check_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path,
                        default=Path('artifacts/day07_final_metrics.json'))
    args = parser.parse_args()
    root = args.root.resolve()
    frozen = load(root / 'artifacts/frozen_selection.json')
    questions = root / 'benchmark/questions.json'
    if not questions.is_file():
        raise FileNotFoundError(f"Missing benchmark: {questions}")
    current_sha = hashlib.sha256(questions.read_bytes()).hexdigest()
    check_equal(current_sha, frozen.get('benchmark_sha256'), 'Frozen benchmark SHA256')
    check_equal(frozen.get('method'), 'tfidf', 'Frozen selected method')
    check_equal(frozen.get('baseline_method'), 'dense', 'Frozen baseline method')

    def retrieval(method, budget):
        folder = f'retrieval-test-{method}{budget}'
        path = root / 'starter/results' / folder / method / 'summary.json'
        s = load(path)
        for field, expected in [('strategy', method), ('split', 'test'),
                                ('n', 5), ('excluded_unanswerable', 2),
                                ('top_k', frozen.get('top_k')),
                                ('candidate_k', frozen.get('candidate_k')),
                                ('budget_chars', budget),
                                ('use_query_filter', frozen.get('use_query_filter'))]:
            check_equal(s.get(field), expected, f'{path}: {field}')
        return {
            'method': method, 'budget_chars': budget,
            'answerable_n': s['n'],
            'excluded_unanswerable': s['excluded_unanswerable'],
            'hit_at_1': s['hit1'], 'evidence_hit_at_3': s['hit3'],
            'context_evidence': s['context_hit'],
            'hit_at_1_rate': s['hit1'] / s['n'],
            'evidence_hit_at_3_rate': s['hit3'] / s['n'],
            'context_evidence_rate': s['context_hit'] / s['n'],
            'median_retrieval_ms': s.get('median_ms'),
            'avg_context_chars': s.get('avg_context_chars'),
            'source_summary_path': path.relative_to(root).as_posix(),
        }

    def agent(method, budget, split):
        folder = f'gemini-{split}-{method}{budget}'
        path = root / 'starter/results' / folder / 'agent_metrics.json'
        a = load(path)
        check_equal(a.get('split'), split, f'{path}: split')
        check_equal(a.get('cases'), 17 if split == 'dev' else 7,
                    f'{path}: number of cases')
        return {
            'cases': a['cases'], 'tool_use': a.get('tool_use'),
            'grounded_answer': a.get('grounded_answer'),
            'citation_validity': a.get('citation_validity'),
            'answer_citation_coverage': a.get('answer_citation_coverage'),
            'abstention': a.get('abstention'),
            'false_abstention': a.get('false_abstention'),
            'provider_errors': a.get('provider_errors'),
            'source_metrics_path': path.relative_to(root).as_posix(),
        }

    baseline_method = frozen['baseline_method']
    baseline_budget = frozen['baseline_budget']
    selected_method = frozen['method']
    selected_budget = frozen['budget']
    result = {
        'experiment': 'Day07 retrieval/context agent baseline vs frozen selection',
        'review_status': 'AI-assisted review labels; student sign-off required',
        'benchmark_sha256': current_sha,
        'frozen_selection': {
            k: frozen.get(k) for k in
            ('model', 'provider', 'method', 'baseline_method', 'budget',
             'baseline_budget', 'top_k', 'candidate_k', 'max_searches',
             'use_query_filter')
        },
        'retrieval_test': {
            'baseline': retrieval(baseline_method, baseline_budget),
            'selected': retrieval(selected_method, selected_budget),
            'metric_scope': 'original benchmark queries; 5 answerable cases',
        },
        'agent': {
            'dev': {
                'baseline': agent(baseline_method, baseline_budget, 'dev'),
                'selected': agent(selected_method, selected_budget, 'dev'),
            },
            'test': {
                'baseline': agent(baseline_method, baseline_budget, 'test'),
                'selected': agent(selected_method, selected_budget, 'test'),
            },
        },
        'limitations': [
            'Selected method and context budget both differ from baseline.',
            'TEST contains only 5 answerable and 2 unanswerable questions.',
            'Agent context metrics and original-query retrieval metrics differ.',
            'Human confirmation of AI-assisted labels is not machine-verifiable.',
            'LLM service cost is unknown until tier-specific billing is checked.',
        ],
    }
    output = args.output
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: frozen selection and 2 retrieval TEST summaries verified')
    print(f'PASS: agent DEV/TEST metrics from both configurations loaded')
    print(f'Wrote: {output}')
    print('NOTE: review labels still need student sign-off; no raw traces exported.')


if __name__ == '__main__':
    main()
