import json
import tempfile
import unittest
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from hospital_assistant.assessment.comparison import compare, shared_context, summarize
from hospital_assistant.rag.chain import RagAssistant

DOC = {'protocol_id': 'PR-001', 'text': 'Use um identificador existente.'}

class FakeModel:
    identity = {'model_id': 'fake'}
    def __init__(self, config=None):
        self.model = self
        self.enabled = True
        self.calls = []
        self.generation = self
    def to_dict(self):
        return {'do_sample': False}
    def fits(self, prompt):
        return True
    @contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True
    def generate(self, prompt):
        self.calls.append(self.enabled)
        return {'answer': '[PR-001] Use um identificador existente.', 'truncated': False}

class ComparisonTests(unittest.TestCase):
    def test_automatic_response_does_not_call_model(self):
        from hospital_assistant.assessment.comparison import FrozenRetriever
        model = FakeModel()
        response = RagAssistant(FrozenRetriever([]), model).ask('fora da base')
        self.assertFalse(response['generation_performed'])
        self.assertEqual(model.calls, [])
        metrics = summarize([{'status': 'completed', 'response': response, 'seconds': 1}])
        self.assertEqual(metrics['generation'], 0)
        self.assertIsNone(metrics['citation_problem_rate'])

    def test_shared_context_respects_both_prompt_budgets(self):
        model = FakeModel()
        model.fits = lambda prompt: 'LONG' not in prompt.to_string()
        self.assertEqual(shared_context('q', [DOC, {**DOC, 'text': 'LONG'}], model), [DOC])

    def test_comparison_reuses_context_and_restores_adapter(self):
        model = FakeModel()
        class Retriever:
            calls = []
            def __init__(self, config): pass
            def search(self, question):
                self.calls.append(question)
                return [DOC] if question == 'covered' else []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/'config.json'
            config.write_text(json.dumps({'index': str(root/'index'), 'run': str(root/'run')}))
            cases = root/'cases.json'
            cases.write_text(json.dumps([{'id': str(i), 'question': q, 'split': 'validation'}
                                         for i, q in enumerate(['covered', 'uncovered'])]))
            with patch('hospital_assistant.rag.index.Retriever', Retriever), patch(
                    'hospital_assistant.rag.model.LocalModel', return_value=model):
                summary = compare(config, cases, 'validation', root/'out')
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(Retriever.calls, ['covered', 'uncovered'])
            self.assertEqual(model.calls, [False, True, False, True])
            self.assertTrue(model.enabled)
            rows = [json.loads(line) for line in (root/'out/results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({r['context_sha256'] for r in rows[:4]}), 1)
            for r in rows[:4]:
                self.assertEqual(r['response']['context_sources'], [DOC])
            for metrics in summary['variants'].values():
                self.assertEqual(metrics['generation'], 1)
                self.assertEqual(metrics['automatic_responses'], 1)
                self.assertEqual(metrics['citation_problem_rate'], 0)
