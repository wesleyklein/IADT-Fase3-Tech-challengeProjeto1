"""Integração real LangGraph/SQLite, sem GPU nem download de modelos."""
import tempfile
import unittest
from pathlib import Path
from hospital_assistant.hospital import seed_database
from hospital_assistant.workflow.service import WorkflowService


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db = root / 'hospital.sqlite3'
        self.storage = root / 'workflows.sqlite3'
        seed_database(Path(__file__).resolve().parents[1] / 'data/synthetic/hospital.json', self.db)
        self.service = WorkflowService(self.storage, self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pause_persists_and_review_is_not_replayed(self):
        initial = self.service.start('exams', patient_id='SYN-P001')
        self.assertEqual(initial['status'], 'awaiting_review')
        self.assertEqual(initial['draft']['pending_exams'], ['SYN-E001'])
        self.assertIsNone(initial['draft']['exams'][0]['result'])
        reopened = WorkflowService(self.storage, self.db)
        saved = reopened.show(initial['thread_id'])
        self.assertEqual(saved['draft_hash'], initial['draft_hash'])
        with self.assertRaises(ValueError):
            reopened.review(saved['thread_id'], 'approve', 'Teste', 'hash incorreto')
        approved = reopened.review(saved['thread_id'], 'approve', 'Teste', saved['draft_hash'])
        self.assertEqual(approved['status'], 'approved_didactic')
        with self.assertRaises(ValueError):
            reopened.review(saved['thread_id'], 'reject', 'Teste', saved['draft_hash'])
        final = reopened.show(saved['thread_id'])
        self.assertEqual(sum(e['node'] == 'review' for e in final['events']), 1)

    def test_invalid_and_missing_patient_block(self):
        for patient in ('SYN-P999', "' OR 1=1 --"):
            result = self.service.start('patient', patient_id=patient)
            self.assertEqual(result['status'], 'blocked')
            self.assertNotIn('draft', result)

    def test_protocol_requires_patient_and_passes_current_record_to_rag(self):
        calls = []

        class FakeRag:
            def ask(self, question, patient_data=None):
                calls.append((question, patient_data))
                return {'answer': '[PR-002] Hemograma está pending.', 'status': 'draft'}

        self.service.rag = lambda: FakeRag()
        result = self.service.start('protocol', question='Quais exames e situações?',
                                    patient_id='SYN-P001')
        self.assertEqual(result['status'], 'awaiting_review')
        self.assertEqual(calls[0][0], 'Quais exames e situações?')
        self.assertEqual(calls[0][1]['patient']['id'], 'SYN-P001')
        self.assertEqual(calls[0][1]['exams'][0]['name'], 'Hemograma')
        self.assertEqual(calls[0][1]['exams'][0]['status'], 'pending')
        blocked = self.service.start('protocol', question='Quais exames?')
        self.assertEqual(blocked['status'], 'blocked')

    def test_rejection_persists_and_cannot_be_approved(self):
        initial = self.service.start('report', patient_id='SYN-P001')
        self.assertIn('SYN-E001', initial['draft']['missing_results'])
        rejected = self.service.review(initial['thread_id'], 'reject', 'Teste', initial['draft_hash'])
        self.assertEqual(rejected['status'], 'rejected')
        with self.assertRaises(ValueError):
            self.service.review(initial['thread_id'], 'approve', 'Teste', initial['draft_hash'])

    def test_prescription_contains_only_placeholders(self):
        result = self.service.start('prescription', patient_id='SYN-P005')
        self.assertEqual(result['status'], 'awaiting_review')
        self.assertTrue(all(value == '[PREENCHIMENTO EXCLUSIVO DO PROFISSIONAL]'
                            for value in result['draft']['form'].values()))

    def test_rag_failure_is_recorded_without_breaking_patient_lookup(self):
        def unavailable():
            raise RuntimeError('Índice indisponível para teste')
        self.service.rag = unavailable
        result = self.service.start('protocol', question='Exame pendente?', patient_id='SYN-P001')
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(any(e['node'] == 'failure' for e in result['events']))
        self.assertEqual(self.service.start('patient', patient_id='SYN-P001')['status'], 'awaiting_review')
