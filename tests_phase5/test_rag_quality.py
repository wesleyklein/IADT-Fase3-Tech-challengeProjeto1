import unittest
from unittest.mock import Mock
from test_comparison import FakeModel, DOC
from hospital_assistant.rag.chain import RagAssistant, content_missing
from hospital_assistant.assessment.comparison import FrozenRetriever, summarize

class RagQualityTests(unittest.TestCase):
    def test_citation_only_and_empty_are_missing_but_prose_is_not(self):
        for answer in ('', '[PR-005]', '**[PR-001]**; [PR-005].', ' ... '):
            self.assertTrue(content_missing(answer))
        self.assertFalse(content_missing('[PR-005] Não complete resultados.'))

    def test_valid_citation_with_no_content_is_flagged_separately(self):
        model = FakeModel()
        model.generate = lambda prompt: {'answer': '[PR-001]', 'truncated': False}
        response = RagAssistant(FrozenRetriever([DOC]), model).ask('q')
        self.assertEqual(response['status'], 'content_review_required')
        self.assertTrue(response['content_missing'])
        metrics = summarize([{'status': 'completed', 'seconds': 1, 'response': response}])
        self.assertEqual(metrics['citation_problem_rate'], 0)
        self.assertEqual(metrics['content_missing_rate'], 1)

    def test_base_selection_restores_adapter_on_generation_error(self):
        model = FakeModel()
        model.config = {'model_variant': 'base', 'prompt_variant': 'short'}
        def fail(prompt):
            self.assertFalse(model.enabled)
            raise RuntimeError('generation failed')
        model.generate = fail
        with self.assertRaises(RuntimeError):
            RagAssistant(FrozenRetriever([DOC]), model).ask('q')
        self.assertTrue(model.enabled)

    def test_config_selection_is_recorded_and_legacy_defaults_remain(self):
        model = FakeModel()
        model.config = {'model_variant': 'base', 'prompt_variant': 'short'}
        response = RagAssistant(FrozenRetriever([DOC]), model).ask('q')
        self.assertEqual(model.calls, [False])
        self.assertFalse(response['model']['adapter_enabled'])
        self.assertEqual(response['prompt_variant'], 'short')
        model.config = {}
        response = RagAssistant(FrozenRetriever([DOC]), model).ask('q')
        self.assertTrue(response['model']['adapter_enabled'])
        self.assertEqual(response['prompt_variant'], 'current')
        model.config = {'prompt_variant': 'typo'}
        with self.assertRaises(ValueError):
            RagAssistant(FrozenRetriever([DOC]), model)
