"""Contratos de dados e máscara da resposta, sem dependência de GPU."""
import random
from collections import Counter
from pathlib import Path
from ..common import digest, read_jsonl

SYSTEM = ("You are an academic assistant for a fictional hospital. Answer in the language "
          "of the question. Do not invent patient facts or sources. Patient-specific "
          "decisions require human review. This is not a clinical service.")


def load_split(paths, expected):
    """Carrega um conjunto de exemplos e verifica IDs, idioma e revisão das traduções."""
    rows, ids = [], set()
    for path in paths:
        for row in read_jsonl(path):
            if row.get("split") != expected:
                raise ValueError(f"Split inválido em {path}: esperado {expected}")
            if row["id"] in ids:
                raise ValueError("ID repetido: não misture original e tradução do mesmo exemplo")
            if not row.get("question", "").strip() or not row.get("answer", "").strip():
                raise ValueError("Pergunta/resposta vazia")
            if row.get("language") == "pt-BR" and not row.get("synthetic"):
                if row.get("review_status") != "translation_reviewed":
                    raise ValueError("Tradução pública não revisada")
            ids.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"Conjunto {expected} vazio; execute a preparação dos dados")
    return rows


def assert_disjoint(*splits):
    """Impede que documentos ou exemplos relacionados vazem entre treino e avaliação."""
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            for field in ["id", "document_id", "group_id", "content_hash"]:
                a = {r[field] for r in left if r.get(field)}
                b = {r[field] for r in right if r.get(field)}
                if a & b:
                    raise ValueError(f"Vazamento entre conjuntos: {field}")


def encode_answer(tokenizer, row, max_length):
    """Tokeniza a conversa inteira e usa -100 para excluir o prompt do cálculo da perda."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["question"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": row["answer"]}],
                                         tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat template incompatível com máscara da resposta")
    # Não treinar em respostas clínicas cortadas silenciosamente.
    if len(full) > max_length:
        return None
    if len(full) <= len(prompt):
        raise ValueError("Resposta sem tokens supervisionados")
    # -100 é o valor que a função de perda ignora: aprendemos a resposta, não a pergunta.
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prompt) + full[len(prompt):]}


def tokenize_rows(tokenizer, rows, max_length, seed, limit=None):
    """Seleciona exemplos completos que cabem no orçamento e relata os rejeitados por tamanho."""
    selected, encoded, rejected = [], [], []
    ordered = sorted(rows, key=lambda r: r["id"])
    random.Random(seed).shuffle(ordered)
    for row in ordered:
        item = encode_answer(tokenizer, row, max_length)
        if item is None:
            rejected.append(row["id"])
        else:
            selected.append(row)
            encoded.append(item)
    eligible = len(selected)
    if limit is not None:
        selected, encoded = selected[:limit], encoded[:limit]
    if not encoded:
        raise ValueError("Nenhum exemplo completo cabe em max_length; aumente o limite em GPU maior")
    report = {"input": len(rows), "eligible": eligible, "used": len(encoded),
              "too_long_ids": rejected, "used_ids": [r["id"] for r in selected],
              "languages": dict(Counter(r.get("language", "unknown") for r in selected))}
    return selected, encoded, report


def fingerprints(paths):
    """Registra hashes dos arquivos para detectar mudanças dos dados entre execuções."""
    return {str(p): digest(Path(p).read_text(encoding="utf-8")) for p in paths}


def collate_rows(rows, pad_id):
    """Padding mascara só as posições adicionadas, mesmo se pad_id == eos_id."""
    width = max(len(r["input_ids"]) for r in rows)
    return {key: [r[key] + [fill] * (width - len(r[key])) for r in rows]
            for key, fill in [("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)]}

