"""Importa XML oficial e separa documentos/duplicatas antes da tradução."""
import random
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from .common import clean_text, digest, redact_identifiers, write_json, write_jsonl

REPOSITORY = "https://github.com/abachaa/MedQuAD.git"
BLOCKED = {"10_MPlus_ADAM_QA", "11_MPlusDrugs_QA", "12_MPlusHerbsSupplements_QA"}


def download(destination, revision):
    """Obtém o MedQuAD e fixa a revisão Git usada como fonte dos exemplos."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Destino já existe. Use-o em prepare ou escolha outro destino.")
    subprocess.run(["git", "clone", REPOSITORY, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)


def source_revision(root, expected):
    """Confere a revisão e rejeita alterações locais que impediriam reproduzir os dados."""
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"Revisão diferente da configuração: {actual}")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ValueError("Fonte MedQuAD modificada: use checkout limpo para reproduzir o recorte.")
    return actual


def parse_collection(root, config):
    """Extrai perguntas/respostas dos XML e contabiliza registros rejeitados pelos filtros."""
    rows, rejected = [], Counter()
    collections = config["collections"]
    if set(collections) & (BLOCKED | set(config["excluded_collections"])):
        raise ValueError("Coleção excluída por ausência de respostas/restrição de redistribuição.")
    for collection in collections:
        directory = Path(root) / collection
        if not directory.is_dir():
            raise ValueError(f"Coleção ausente: {collection}")
        for path in sorted(directory.glob("*.xml")):
            try:
                document = ET.parse(path).getroot()
            except ET.ParseError:
                rejected["invalid_xml"] += 1
                continue
            url = document.get("url", "").strip()
            if not url:
                rejected["missing_source"] += 1
                continue
            document_id = f"{collection}/{path.stem}"
            for pair in document.findall(".//QAPair"):
                question_node, answer_node = pair.find("Question"), pair.find("Answer")
                question = clean_text("".join(question_node.itertext())) if question_node is not None else ""
                answer = clean_text("".join(answer_node.itertext())) if answer_node is not None else ""
                if not question or not answer:
                    rejected["empty_qa"] += 1
                    continue
                if not config["min_answer_chars"] <= len(answer) <= config["max_answer_chars"]:
                    rejected["answer_length"] += 1
                    continue
                question, q_flags = redact_identifiers(question)
                answer, a_flags = redact_identifiers(answer)
                if q_flags or a_flags:
                    # Não liberar automaticamente uma resposta alterada por anonimização.
                    rejected["identifier_detected"] += 1
                    continue
                rows.append({
                    "id": f"medquad:{document_id}:{pair.get('pid')}",
                    "document_id": document_id, "source_url": url,
                    "source_file": path.relative_to(root).as_posix(),
                    "source_file_sha256": digest(path.read_text(encoding="utf-8")),
                    "collection": collection, "focus": clean_text(document.findtext("Focus", "")),
                    "question_type": question_node.get("qtype", "unknown"),
                    "question": question, "answer": answer, "language": "en",
                    "synthetic": False, "license": "CC-BY-4.0",
                    "review_status": "automated_checks_only", "clinical_review": "pending",
                    "content_hash": digest(question.casefold() + "\n" + answer.casefold()),
                })
    if not rows:
        raise ValueError("Nenhum par elegível encontrado.")
    return rows, dict(rejected)


def split_records(rows, seed):
    """Agrupa documentos ligados por URL, pergunta ou resposta iguais.

    As conexões são transitivas: uma duplicata em dois documentos mantém ambos
    no mesmo grupo. O split por hash fica estável quando o limite muda.
    """
    # Union-find une documentos relacionados: uma duplicata transitiva também deve
    # permanecer no mesmo conjunto, evitando que a avaliação memorize o treino.
    parents = {r["document_id"]: r["document_id"] for r in rows}

    def find(x):
        """Encontra o representante do grupo de documentos, encurtando os caminhos já percorridos."""
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    seen = {}
    for row in rows:
        for key in [("url", row["source_url"].rstrip("/")),
                    ("q", row["question"].casefold()), ("a", row["answer"].casefold())]:
            doc = row["document_id"]
            if key in seen:
                a, b = find(doc), find(seen[key])
                parents[max(a, b)] = min(a, b)
            seen[key] = doc
    unique, duplicates = {}, 0
    for row in rows:
        group = find(row["document_id"])
        bucket = int(digest(f"{seed}:{group}")[:8], 16) % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        row = dict(row, group_id=group, split=split)
        if row["content_hash"] in unique:
            duplicates += 1
        else:
            unique[row["content_hash"]] = row
    return list(unique.values()), duplicates


def prepare(root, output, config):
    """Filtra, agrupa duplicatas e exporta treino, validação, teste e fila de tradução."""
    if config["limit"] < 1:
        raise ValueError("limit deve ser positivo")
    revision = source_revision(root, config["revision"])
    rows, rejected = parse_collection(root, config)
    rows, duplicates = split_records(rows, config["seed"])
    eligible = len(rows)
    random.Random(config["seed"]).shuffle(rows)
    selected = sorted(rows[:config["limit"]], key=lambda x: x["id"])
    counts = Counter(r["split"] for r in selected)
    if any(counts[s] == 0 for s in ["train", "validation", "test"]):
        raise ValueError("Recorte sem exemplos em algum split. Aumente o limite ou ajuste o recorte.")
    output = Path(output)
    for split in ["train", "validation", "test"]:
        write_jsonl(output / f"{split}.en.jsonl", [r for r in selected if r["split"] == split])
    queue = [{"id": r["id"], "source_hash": r["content_hash"], "split": r["split"],
              "source_url": r["source_url"], "question_en": r["question"], "answer_en": r["answer"],
              "question_pt": "", "answer_pt": "", "status": "pending",
              "reviewer": "", "reviewed_at": ""} for r in selected]
    # Revisões são mantidas em arquivo separado; reexecução não sobrescreve trabalho humano.
    write_jsonl(output / "translation_queue.jsonl", queue)
    manifest = {"dataset": "MedQuAD", "repository": REPOSITORY, "revision": revision,
                "config": config, "eligible_unique": eligible, "selected": len(selected),
                "counts": dict(counts), "rejected": rejected, "exact_duplicates": duplicates,
                "translation_status": "pending", "clinical_review": "pending",
                "records_sha256": digest("\n".join(r["content_hash"] for r in selected))}
    write_json(output / "manifest.json", manifest)
    return manifest

