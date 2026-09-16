import unittest
from hospital_assistant.assessment.runner import retrieval_scores


class RetrievalMetricsTests(unittest.TestCase):
    def test_repeated_chunks_do_not_inflate_protocol_recall(self):
        result = retrieval_scores(['PR-001', 'PR-002'],
            [{'protocol_id': 'PR-009'}, {'protocol_id': 'PR-001'}, {'protocol_id': 'PR-001'}])
        self.assertEqual(result['recall'], 0.5)
        self.assertEqual(result['reciprocal_rank'], 0.5)

    def test_negative_case_has_no_positive_recall_denominator(self):
        result = retrieval_scores([], [])
        self.assertIsNone(result['recall'])
        self.assertFalse(result['false_positive'])
        self.assertTrue(retrieval_scores([], [{'protocol_id': 'PR-001'}])['false_positive'])
