"""Exporta somente traduções explicitamente revisadas, ligadas à fonte original."""
import datetime
from pathlib import Path
from .common import clean_text, read_jsonl, redact_identifiers, write_jsonl


def export_reviewed(source, reviews, output):
    """Exporta traduções aprovadas preservando o vínculo com a fonte e o conjunto original."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Use diretório de saída separado dos originais.")
    originals = {r["id"]: r for split in ["train", "validation", "test"]
                 for r in read_jsonl(source / f"{split}.en.jsonl")}
    approved, seen = [], set()
    for review in read_jsonl(reviews):
        if review["id"] in seen:
            raise ValueError("ID de revisão duplicado")
        seen.add(review["id"])
        if review["status"] != "approved":
            continue
        row = originals.get(review["id"])
        if not row or review["source_hash"] != row["content_hash"] or review["split"] != row["split"]:
            raise ValueError("Revisão não corresponde à fonte/split")
        if not review.get("reviewer", "").strip() or not review.get("reviewed_at", "").strip():
            raise ValueError("Tradução aprovada exige responsável e data")
        datetime.date.fromisoformat(review["reviewed_at"])
        translated_question, translated_answer = clean_text(review["question_pt"]), clean_text(review["answer_pt"])
        if not translated_question or not translated_answer or redact_identifiers(translated_question + " " + translated_answer)[1]:
            raise ValueError("Tradução vazia ou com identificadores detectados")
        approved.append(dict(row, language="pt-BR", question=translated_question, answer=translated_answer,
                             original_question=row["question"], original_answer=row["answer"],
                             translation_review={k: review[k] for k in ["reviewer", "reviewed_at"]},
                             review_status="translation_reviewed"))
    if not approved:
        raise ValueError("Nenhuma tradução aprovada. Preencha a revisão sem alterar o original.")
    for split in ["train", "validation", "test"]:
        write_jsonl(output / f"{split}.pt.jsonl", [r for r in approved if r["split"] == split])
    return {"approved": len(approved), "clinical_review": "pending"}

