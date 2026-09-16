import unittest
from langchain_core.documents import Document
from hospital_assistant.rag.chunks import protocol_chunks


class ProtocolChunkTests(unittest.TestCase):
    def test_each_body_chunk_preserves_title_version_and_original_offset(self):
        body = 'Selecione um identificador existente. Nunca associe por nome. ' * 8
        for newline in ('\n', '\r\n'):
            text = newline.join(['PR-001 | Identificação do paciente', 'Versão: 1.0',
                                 'SINTÉTICO', '', body])
            chunks = protocol_chunks([Document(page_content=text,
                metadata={'protocol_id': 'PR-001', 'source': 'PR-001.txt'})], 90, 15)
            self.assertGreater(len(chunks), 1)
            for chunk in chunks:
                self.assertTrue(chunk.page_content.startswith('PR-001 | Identificação do paciente\nVersão: 1.0\n'))
                start = chunk.metadata['start_index']
                excerpt = chunk.metadata['body_text']
                self.assertEqual(text[start:start + len(excerpt)], excerpt)
                self.assertNotIn('SINTÉTICO', chunk.page_content)

    def test_empty_body_and_missing_header_fail(self):
        for text in ('Sem cabeçalho', 'PR-001 | Nome\nVersão: 1.0\n\n '):
            with self.assertRaises(ValueError):
                protocol_chunks([Document(page_content=text, metadata={'protocol_id': 'PR-001'})], 100, 10)
