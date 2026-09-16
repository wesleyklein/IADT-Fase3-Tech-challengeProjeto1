"""Comandos de preparação e consulta de dados. Execute a partir da raiz do repositório."""
import argparse
import json
import sqlite3
import subprocess
import sys
from .common import read_json
from .hospital import get_patient, seed_database
from .medquad import download, prepare
from .synthetic import prepare_synthetic
from .translations import export_reviewed


def main():
    """Interpreta os argumentos do terminal e encaminha a operação ao módulo responsável."""
    parser = argparse.ArgumentParser(description="Dados do hospital acadêmico (somente sintéticos)")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--destination", default="data/raw/MedQuAD")
    fetch.add_argument("--config", default="config/data.json")
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", default="data/raw/MedQuAD")
    prep.add_argument("--output", default="data/processed/medquad")
    prep.add_argument("--config", default="config/data.json")
    seed = sub.add_parser("seed")
    seed.add_argument("--source", default="data/synthetic/hospital.json")
    seed.add_argument("--database", default="data/local/hospital.sqlite3")
    patient = sub.add_parser("patient")
    patient.add_argument("patient_id")
    patient.add_argument("--database", default="data/local/hospital.sqlite3")
    synth = sub.add_parser("synthetic")
    synth.add_argument("--source", default="data/synthetic")
    synth.add_argument("--output", default="data/processed/internal")
    trans = sub.add_parser("translations")
    trans.add_argument("--source", default="data/processed/medquad")
    trans.add_argument("--reviews", required=True)
    trans.add_argument("--output", default="data/processed/medquad-pt")
    args = parser.parse_args()
    try:
        if args.command == "download":
            download(args.destination, read_json(args.config)["revision"])
            result = {"download": "ok"}
        elif args.command == "prepare":
            result = prepare(args.source, args.output, read_json(args.config))
        elif args.command == "seed":
            result = seed_database(args.source, args.database)
        elif args.command == "patient":
            result = get_patient(args.database, args.patient_id)
        elif args.command == "synthetic":
            result = prepare_synthetic(args.source, args.output)
        else:
            result = export_reviewed(args.source, args.reviews, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, sqlite3.Error, subprocess.CalledProcessError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

