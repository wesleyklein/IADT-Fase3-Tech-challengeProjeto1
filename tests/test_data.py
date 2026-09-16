import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch
from pathlib import Path

from hospital_assistant.common import clean_text, digest, read_json, read_jsonl, redact_identifiers, write_jsonl
from hospital_assistant.hospital import get_patient, seed_database
from hospital_assistant.medquad import parse_collection, split_records
from hospital_assistant.synthetic import prepare_synthetic
from hospital_assistant.translations import export_reviewed

ROOT = Path(__file__).resolve().parents[1]


def record(doc, question, answer, url=None):
    return {"id": doc + question, "document_id": doc, "source_url": url or f"https://example.org/{doc}",
            "question": question, "answer": answer, "content_hash": digest(question + answer)}


class DataTests(unittest.TestCase):
    def test_cleaning_keeps_units_negation_and_accents(self):
        self.assertEqual(clean_text(" Não  usar\n 5 mg &amp; 2 mL. "), "Não usar 5 mg & 2 mL.")

    def test_partial_anonymization(self):
        text, flags = redact_identifiers("teste@example.org 123.456.789-00 (11) 99999-0000 5 mg")
        self.assertEqual(set(flags), {"email", "cpf", "phone_br"})
        self.assertNotIn("123.456.789", text)
        self.assertIn("5 mg", text)

    def test_transitive_duplicates_stay_in_one_split(self):
        rows = [record("a", "q1", "a1"), record("b", "q1", "a2"),
                record("c", "q3", "a2"), record("c", "q4", "a4")]
        result, _ = split_records(rows, 42)
        self.assertEqual(len({r["group_id"] for r in result}), 1)
        self.assertEqual(len({r["split"] for r in result}), 1)

    def test_deduplication_and_reproducibility(self):
        rows = [record("a", "same", "answer"), record("b", "same", "answer")]
        first, duplicates = split_records(rows, 42)
        self.assertEqual(duplicates, 1)
        self.assertEqual(first, split_records(rows, 42)[0])

    def test_xml_empty_and_length_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "sample"
            folder.mkdir()
            (folder / "one.xml").write_text('<Document url="https://example.org"><Focus>Example</Focus><QAPairs>'
                '<QAPair pid="1"><Question qtype="information">Question?</Question><Answer>Valid answer.</Answer></QAPair>'
                '<QAPair pid="2"><Question>Empty?</Question><Answer/></QAPair>'
                '</QAPairs></Document>', encoding="utf-8")
            rows, rejected = parse_collection(Path(tmp), {"collections": ["sample"], "excluded_collections": [],
                                                        "min_answer_chars": 5, "max_answer_chars": 100})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rejected["empty_qa"], 1)
            self.assertEqual(rows[0]["source_url"], "https://example.org")

    def test_restricted_collection_rejected(self):
        with self.assertRaises(ValueError):
            parse_collection(Path("."), {"collections": ["11_MPlusDrugs_QA"], "excluded_collections": []})

    def test_seed_idempotence_and_parameterized_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "hospital.sqlite3"
            source = ROOT / "data/synthetic/hospital.json"
            first = seed_database(source, database)
            self.assertEqual(first, seed_database(source, database))
            self.assertEqual(first, {"patients": 6, "exams": 5})
            patient = get_patient(database, "SYN-P001")
            self.assertEqual(patient["exams"][0]["status"], "pending")
            with self.assertRaises(ValueError):
                get_patient(database, "' OR 1=1 --")
            with self.assertRaises(ValueError):
                get_patient(database, "SYN-P999")

    def test_database_enforces_foreign_keys_and_result_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "db.sqlite3"
            seed_database(ROOT / "data/synthetic/hospital.json", path)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("PRAGMA foreign_keys=ON")
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute("INSERT INTO exams VALUES ('SYN-E999','SYN-P999','X','pending','2026-01-01',NULL)")
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute("UPDATE exams SET result='inventado' WHERE status='pending'")

    def test_non_synthetic_seed_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = Path(tmp) / "seed.json"
            seed.write_text('{"synthetic":false}', encoding="utf-8")
            with self.assertRaises(ValueError):
                seed_database(seed, Path(tmp) / "db.sqlite3")

    def test_connections_closed_after_seed_lookup_and_lookup_error(self):
        real_connect = sqlite3.connect
        opened = []

        def track(*args, **kwargs):
            connection = real_connect(*args, **kwargs)
            opened.append(connection)
            return connection

        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "hospital.sqlite3"
            with patch("hospital_assistant.hospital.sqlite3.connect", side_effect=track):
                seed_database(ROOT / "data/synthetic/hospital.json", database)
                get_patient(database, "SYN-P001")
                with self.assertRaises(ValueError):
                    get_patient(database, "SYN-P999")
            try:
                self.assertEqual(len(opened), 3)
                for connection in opened:
                    with self.assertRaises(sqlite3.ProgrammingError):
                        connection.execute("SELECT 1")
            finally:
                for connection in opened:
                    connection.close()

    def test_seed_error_rolls_back_and_closes_connection(self):
        real_connect = sqlite3.connect
        opened = []

        def track(*args, **kwargs):
            connection = real_connect(*args, **kwargs)
            opened.append(connection)
            return connection

        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "hospital.sqlite3"
            seed = Path(tmp) / "seed.json"
            data = read_json(ROOT / "data/synthetic/hospital.json")
            data["patients"][1]["synthetic"] = False
            seed.write_text(json.dumps(data), encoding="utf-8")
            with patch("hospital_assistant.hospital.sqlite3.connect", side_effect=track):
                with self.assertRaises(ValueError):
                    seed_database(seed, database)
            try:
                with self.assertRaises(sqlite3.ProgrammingError):
                    opened[0].execute("SELECT 1")
            finally:
                for connection in opened:
                    connection.close()
            with closing(real_connect(database)) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM patients").fetchone()[0], 0)

    def test_internal_sources_and_evaluation_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = prepare_synthetic(ROOT / "data/synthetic", tmp)
            self.assertEqual(result["internal_examples"], 24)
            self.assertEqual(result["evaluation_scenarios"], 8)
            self.assertTrue(all(r["synthetic"] for r in read_jsonl(Path(tmp) / "internal.train.jsonl")))

    def test_translation_requires_review_and_preserves_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            row = dict(record("doc", "Question?", "Answer."), split="test", language="en")
            for split in ["train", "validation", "test"]:
                write_jsonl(source / f"{split}.en.jsonl", [row] if split == "test" else [])
            reviews = Path(tmp) / "reviews.jsonl"
            review = {"id": row["id"], "source_hash": row["content_hash"], "split": "test",
                      "question_pt": "Pergunta?", "answer_pt": "Resposta.", "status": "pending",
                      "reviewer": "", "reviewed_at": ""}
            write_jsonl(reviews, [review])
            with self.assertRaises(ValueError):
                export_reviewed(source, reviews, Path(tmp) / "out")
            review.update(status="approved", reviewer="Revisor de teste", reviewed_at="2026-09-06")
            write_jsonl(reviews, [review])
            export_reviewed(source, reviews, Path(tmp) / "out")
            translated = read_jsonl(Path(tmp) / "out/test.pt.jsonl")[0]
            self.assertEqual(translated["original_answer"], "Answer.")
            review["split"] = "train"
            write_jsonl(reviews, [review])
            with self.assertRaises(ValueError):
                export_reviewed(source, reviews, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
