import tempfile
import unittest
from pathlib import Path
from hospital_assistant.common import write_jsonl
from hospital_assistant.finetuning.data import (assert_disjoint, collate_rows, encode_answer,
                                               load_split, tokenize_rows)
from hospital_assistant.finetuning.evaluation import token_f1
from hospital_assistant.finetuning.runtime import validate_config


class TokenizerDouble:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        # Prefixo constante e token EOS ao final da resposta.
        if add_generation_prompt:
            return [1, 2, 3]
        return [1, 2, 3] + [4] * len(messages[-1]["content"]) + [9]


class FinetuningTests(unittest.TestCase):
    def test_answer_mask_keeps_eos(self):
        item = encode_answer(TokenizerDouble(), {"question": "q", "answer": "ab"}, 10)
        self.assertEqual(item["labels"], [-100, -100, -100, 4, 4, 9])
        padded = collate_rows([item, {"input_ids": [1, 9], "attention_mask": [1, 1], "labels": [-100, 9]}], 9)
        self.assertEqual(padded["labels"][1], [-100, 9, -100, -100, -100, -100])

    def test_long_answer_is_excluded_not_truncated(self):
        self.assertIsNone(encode_answer(TokenizerDouble(), {"question": "q", "answer": "long"}, 5))

    def test_split_and_review_gate(self):
        row = {"id": "1", "split": "test", "question": "q", "answer": "a", "language": "pt-BR"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            write_jsonl(path, [row])
            with self.assertRaises(ValueError):
                load_split([path], "train")
            with self.assertRaises(ValueError):
                load_split([path], "test")
            row["review_status"] = "translation_reviewed"
            write_jsonl(path, [row])
            self.assertEqual(load_split([path], "test")[0]["id"], "1")

    def test_leakage_document_and_group(self):
        with self.assertRaises(ValueError):
            assert_disjoint([{"id": "a", "group_id": "same"}], [{"id": "b", "group_id": "same"}])

    def test_deterministic_filter_and_report(self):
        rows = [{"id": str(i), "question": "q", "answer": "a", "language": "en"} for i in range(5)]
        first = tokenize_rows(TokenizerDouble(), rows, 10, 42, 2)
        self.assertEqual(first, tokenize_rows(TokenizerDouble(), list(reversed(rows)), 10, 42, 2))
        self.assertEqual(first[2]["used"], 2)
        self.assertEqual(first[2]["eligible"], 5)

    def test_f1_uses_multiset_overlap(self):
        self.assertEqual(token_f1("A a b", "a a b"), 1)
        self.assertEqual(token_f1("", "a"), 0)
        self.assertAlmostEqual(token_f1("a a", "a"), 2/3)
