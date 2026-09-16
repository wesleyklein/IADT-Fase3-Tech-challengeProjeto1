"""Banco local exclusivamente sintético; consultas parametrizadas e somente leitura."""
import sqlite3
from contextlib import closing
from pathlib import Path
from .common import read_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY CHECK(version=1));
INSERT OR IGNORE INTO schema_version VALUES (1);
CREATE TABLE IF NOT EXISTS patients (
 id TEXT PRIMARY KEY CHECK(id LIKE 'SYN-P%'),
 display_name TEXT NOT NULL, age INTEGER NOT NULL CHECK(age BETWEEN 0 AND 120),
 summary TEXT NOT NULL, synthetic INTEGER NOT NULL CHECK(synthetic=1)
);
CREATE TABLE IF NOT EXISTS exams (
 id TEXT PRIMARY KEY CHECK(id LIKE 'SYN-E%'),
 patient_id TEXT NOT NULL REFERENCES patients(id),
 name TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','completed','cancelled')),
 requested_at TEXT NOT NULL, result TEXT,
 CHECK((status='completed' AND result IS NOT NULL) OR (status!='completed' AND result IS NULL))
);
CREATE INDEX IF NOT EXISTS ix_exams_patient_status ON exams(patient_id,status);
"""


def seed_database(seed_file, database):
    """Carrega pacientes/exames fictícios sem duplicar IDs já existentes no SQLite."""
    data = read_json(seed_file)
    if data.get("synthetic") is not True:
        raise ValueError("Somente fixtures declaradas sintéticas são aceitas")
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    # O contexto SQLite controla commit/rollback; closing libera o arquivo também.
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(SCHEMA)
        for patient in data["patients"]:
            if patient.get("synthetic") is not True:
                raise ValueError("Paciente não sintético")
            connection.execute("INSERT OR IGNORE INTO patients VALUES (?,?,?,?,1)",
                               (patient["id"], patient["display_name"], patient["age"], patient["summary"]))
        for exam in data["exams"]:
            connection.execute("INSERT OR IGNORE INTO exams VALUES (?,?,?,?,?,?)",
                               tuple(exam[field_name] for field_name in ["id", "patient_id", "name", "status", "requested_at", "result"]))
        return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ["patients", "exams"]}


def get_patient(database, patient_id):
    """Consulta o ID exato em modo somente leitura e inclui os exames desse paciente."""
    path = Path(database).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
        if row is None:
            raise ValueError(f"Paciente sintético não encontrado: {patient_id}")
        exams = connection.execute("SELECT * FROM exams WHERE patient_id=? ORDER BY id", (patient_id,)).fetchall()
        return {"patient": dict(row), "exams": [dict(exam) for exam in exams]}

