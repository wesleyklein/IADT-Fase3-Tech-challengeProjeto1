"""Valida fixtures e exporta exemplos internos separados do corpus público."""
from pathlib import Path
from .common import digest, read_json, write_jsonl, write_json


def prepare_synthetic(source, output):
    """Valida as fontes dos exemplos fictícios e separa perguntas de treino e avaliação."""
    source, output = Path(source), Path(output)
    data = read_json(source / "examples.json")
    rows, seen = [], set()
    for example in data:
        if example["id"] in seen or example.get("synthetic") is not True:
            raise ValueError("Exemplo duplicado ou não sintético")
        seen.add(example["id"])
        protocol = (source / example["source_file"]).resolve()
        if not protocol.is_relative_to(source.resolve()) or not protocol.is_file():
            raise ValueError("Fonte interna inválida")
        text = protocol.read_text(encoding="utf-8")
        if not example["question"].strip() or not example["answer"].strip():
            raise ValueError("Exemplo vazio")
        rows.append(dict(example, language="pt-BR", document_id=example["source_file"],
                         source_hash=digest(text), review_status="didactic_draft",
                         clinical_review="pending", split="train",
                         messages=[{"role": "system", "content": "Você é um assistente acadêmico de hospital fictício. Cite a fonte e submeta condutas à validação humana."},
                                   {"role": "user", "content": example["question"]},
                                   {"role": "assistant", "content": example["answer"]}]))
    write_jsonl(output / "internal.train.jsonl", rows)
    scenarios = read_json(source / "evaluation_scenarios.json")
    if any(s["question"] in {r["question"] for r in rows} for s in scenarios):
        raise ValueError("Pergunta de avaliação copiada do treino")
    write_json(output / "evaluation_scenarios.json", scenarios)
    return {"internal_examples": len(rows), "evaluation_scenarios": len(scenarios),
            "clinical_review": "pending"}

