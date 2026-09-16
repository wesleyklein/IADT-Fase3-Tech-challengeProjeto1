"""Comparação pareada base/LoRA, mesmo contexto e quantização, sem juiz externo."""
import math
import re
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from ..common import read_json, write_json, write_jsonl
from .data import SYSTEM, assert_disjoint, fingerprints, load_split, tokenize_rows
from .diagnostics import require_cuda


def token_f1(prediction, reference):
    """Compara palavras da resposta e referência; mede sobreposição lexical, não veracidade."""
    prediction_counts, reference_counts = Counter(re.findall(r"\w+", prediction.casefold())), Counter(re.findall(r"\w+", reference.casefold()))
    overlap_count = sum((prediction_counts & reference_counts).values())
    if not prediction_counts or not reference_counts or not overlap_count:
        return 0.0
    precision, recall = overlap_count / sum(prediction_counts.values()), overlap_count / sum(reference_counts.values())
    return 2 * precision * recall / (precision + recall)


def evaluate(run, output, limit=20, max_new_tokens=128):
    """Compara base e adaptador com as mesmas perguntas e calcula perda e F1 lexical."""
    if limit < 1 or max_new_tokens < 1:
        raise ValueError("limit e max_new_tokens devem ser positivos")
    require_cuda()
    import torch
    from transformers import AutoTokenizer, set_seed
    from peft import PeftModel
    from .runtime import load_base
    run, output = Path(run), Path(output)
    metrics = read_json(run / "metrics.json")
    if metrics.get("status") != "completed":
        raise ValueError("Execução de treino não concluída")
    config = read_json(run / "resolved_config.json")
    expected = read_json(run / "data_report.json")["files_sha256"]
    if fingerprints(expected.keys()) != expected:
        raise ValueError("Dados mudaram desde o treinamento; restaure as fontes da execução")
    test_rows = load_split(config["test_files"], "test")
    assert_disjoint(load_split(config["train_files"], "train"), load_split(config["validation_files"], "validation"), test_rows)
    tokenizer = AutoTokenizer.from_pretrained(run / "adapter", trust_remote_code=False)
    rows, encoded, report = tokenize_rows(tokenizer, test_rows, config["max_length"], config["seed"], limit)
    set_seed(config["seed"])
    model = PeftModel.from_pretrained(load_base(config), run / "adapter", is_trainable=False)
    model.eval()
    outputs, summaries = [], {}
    for name in ["base", "adapted"]:
        losses, scores, tokens = [], [], []
        context = model.disable_adapter() if name == "base" else nullcontext()
        start = time.perf_counter()
        with context, torch.inference_mode():
            for row, item in zip(rows, encoded):
                batch = {key: torch.tensor([value], device="cuda") for key, value in item.items()}
                loss = model(**batch).loss.item()
                if not math.isfinite(loss):
                    raise ValueError("Avaliação retornou loss não finita")
                losses.append(loss)
                tokens.append(sum(label != -100 for label in item["labels"][1:]))
                prompt = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["question"]}], tokenize=True,
                    add_generation_prompt=True, return_tensors="pt").to("cuda")
                generated = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                    max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id)
                answer = tokenizer.decode(generated[0, prompt.shape[1]:], skip_special_tokens=True)
                score = token_f1(answer, row["answer"])
                scores.append(score)
                outputs.append({"id": row["id"], "model": name, "question": row["question"],
                                "reference": row["answer"], "answer": answer, "token_f1": score,
                                "source_url": row.get("source_url"), "language": row.get("language")})
        nll = sum(loss * count for loss, count in zip(losses, tokens)) / sum(tokens)
        summaries[name] = {"answer_nll": nll, "perplexity": math.exp(nll) if nll < 700 else None,
                           "mean_token_f1": sum(scores) / len(scores), "seconds": time.perf_counter() - start}
    write_jsonl(output / "predictions.jsonl", outputs)
    summary = {"status": "completed", "run": str(run), "training_mode": metrics["mode"],
               "model_id": config["model_id"], "revision": config["model_revision"],
               "data": report, "metrics": summaries, "max_new_tokens": max_new_tokens,
               "note": "F1 lexical e perplexidade não provam correção clínica; revisar predictions.jsonl. "
                       "Teste restrito aos exemplos completos que cabem no limite; mesmo modelo base quantizado com adaptador ligado/desligado."}
    write_json(output / "comparison.json", summary)
    return summary

