"""Garante que dados estruturados do paciente sejam enviados à LLM do RAG."""
import unittest
from contextlib import contextmanager
from hospital_assistant.rag.chain import RagAssistant


class Retriever:
    def search(self, question):
        return [{'protocol_id': 'PR-002', 'text': 'pending indica pendência.'}]


class Model:
    identity = {'model_id': 'fake'}
    config = {'model_variant': 'adapted', 'prompt_variant': 'short'}

    def __init__(self):
        self.model = self
        self.prompt = ''

    def fits(self, prompt):
        return True

    @contextmanager
    def disable_adapter(self):
        yield

    def generate(self, prompt):
        self.prompt = prompt.to_string()
        return {'answer': '[PR-002] Hemograma está pending.', 'truncated': False}


class PatientContextTests(unittest.TestCase):
    def test_structured_patient_and_exam_are_in_generation_prompt(self):
        model = Model()
        patient = {
            'patient': {'id': 'SYN-P001', 'display_name': 'Paciente fictício 001', 'age': 52,
                        'summary': 'Exame solicitado sem resultado.'},
            'exams': [{'name': 'Hemograma', 'status': 'pending',
                       'requested_at': '2026-09-01T10:00:00Z', 'result': None}],
        }
        response = RagAssistant(Retriever(), model).ask('Quais exames?', patient_data=patient)
        self.assertIn('SYN-P001', model.prompt)
        self.assertIn('Hemograma', model.prompt)
        self.assertIn('status=pending', model.prompt)
        self.assertEqual(response['patient_source']['tool'], 'patient_lookup')
        self.assertTrue(response['generation_performed'])

    def test_wrong_generated_status_is_replaced_by_grounded_answer(self):
        model = Model()
        model.generate = lambda prompt: {
            'answer': 'O estado atual do exame é pending.', 'truncated': False}
        patient = {
            'patient': {'id': 'SYN-P002', 'display_name': 'Paciente fictício 002', 'age': 39,
                        'summary': 'Exame concluído.'},
            'exams': [{'name': 'Glicemia', 'status': 'completed',
                       'requested_at': '2026-09-01T10:00:00Z',
                       'result': 'Resultado sintético registrado.'}],
        }
        response = RagAssistant(Retriever(), model).ask(
            'Segundo o protocolo, o que significa o estado atual?', patient_data=patient)
        self.assertEqual(response['status'], 'grounded_draft')
        self.assertTrue(response['grounding_verified'])
        self.assertIn('Glicemia', response['answer'])
        self.assertIn('status completed', response['answer'])
        self.assertIn('[PR-002]', response['answer'])
        self.assertEqual(response['original_answer'], 'O estado atual do exame é pending.')
        self.assertIn('status_not_in_patient_record', response['grounding_corrections'])
