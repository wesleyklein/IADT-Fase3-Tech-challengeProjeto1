"""Avaliação de busca e geração; métricas não equivalem a correção clínica."""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from ..common import read_json, write_json


def retrieval_scores(expected, sources):
    """Calcula cobertura e posição dos protocolos esperados, sem inflar acertos por chunks repetidos."""
    ids = list(dict.fromkeys(d['protocol_id'] for d in sources))
    expected = set(expected)
    if not expected:
        return {'recall': None, 'reciprocal_rank': None, 'false_positive': bool(ids)}
    ranks = [i + 1 for i, value in enumerate(ids) if value in expected]
    return {'recall': len(expected.intersection(ids)) / len(expected),
            'reciprocal_rank': 1 / min(ranks) if ranks else 0.0, 'false_positive': None}


def average(rows, key):
    """Calcula a média dos valores aplicáveis; None significa ausência de denominador."""
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def evaluate(config_path, cases_path, split, output, generate=False):
    """Executa os casos de busca/geração e grava resultados, métricas e revisão pendente."""
    from ..rag.index import Retriever
    config = read_json(config_path)
    cases = read_json(cases_path)
    ids = [c['id'] for c in cases]
    if len(ids) != len(set(ids)) or any(c['split'] not in ('validation', 'test') for c in cases):
        raise ValueError('IDs repetidos ou splits inválidos')
    selected = [c for c in cases if c['split'] == split]
    if not selected:
        raise ValueError('Nenhum cenário selecionado')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'split': split, 'mode': 'generation' if generate else 'retrieval',
               'started_at': datetime.now(timezone.utc).isoformat(), 'planned': len(selected),
               'config': config, 'python': platform.python_version(), 'platform': platform.platform(),
               'clinical_review': 'pending', 'files_sha256': {}, 'packages': {}}
    try:
        for path in (Path(config_path), Path(cases_path), Path(config['index']) / 'index.json'):
            summary['files_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ('langchain-core', 'langchain-community', 'transformers', 'peft', 'torch', 'faiss-cpu'):
            try:
                summary['packages'][name] = version(name)
            except PackageNotFoundError:
                summary['packages'][name] = None
        write_json(output / 'summary.json', summary)
        retriever = Retriever(config)
        assistant = None
        if generate:
            from ..rag.model import LocalModel
            from ..rag.chain import RagAssistant
            assistant = RagAssistant(retriever, LocalModel(config))
        rows = []
        with (output / 'results.jsonl').open('x', encoding='utf-8') as stream, \
                (output / 'human_review.jsonl').open('x', encoding='utf-8') as reviews:
            for position, case in enumerate(selected, 1):
                print(f"[{position}/{len(selected)}] {case['id']}", flush=True)
                start = time.perf_counter()
                row = {'id': case['id'], 'question': case['question'], 'expected_protocols': case['protocols']}
                try:
                    response = assistant.ask(case['question']) if assistant else None
                    sources = response['retrieved_sources'] if response else retriever.search(case['question'])
                    row.update(status='completed', sources=sources,
                               **retrieval_scores(case['protocols'], sources))
                    if response:
                        row['response'] = response
                        # Sinais sintáticos: não avaliam a veracidade das afirmações.
                        performed = response.get('generation_performed', False)
                        row['generation_performed'] = performed
                        row['content_missing'] = response.get('content_missing', False) if performed else None
                        row['truncated'] = response.get('truncated', False) if performed else None
                        row['citation_problem'] = (not response.get('cited_sources') or bool(response.get('invalid_citations'))) if performed else None
                        reviews.write(json.dumps({'id': case['id'], 'question': case['question'],
                            'response': response, 'reviewer': None, 'reviewed_at': None,
                            'supported_by_sources': None, 'answers_question': None,
                            'appropriate_abstention': None, 'unsafe_content': None,
                            'comments': '', 'status': 'pending'}, ensure_ascii=False) + '\n')
                        reviews.flush()
                except Exception as exc:
                    row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                row['seconds'] = time.perf_counter() - start
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
        valid = [r for r in rows if r['status'] == 'completed']
        summary.update(status='completed' if len(valid) == len(rows) else 'partial',
                       completed=len(valid), failed=len(rows)-len(valid),
                       metrics={'mean_recall': average(valid, 'recall'),
                                'mrr': average(valid, 'reciprocal_rank'),
                                'false_positive_rate': average(valid, 'false_positive'),
                                'mean_seconds': average(valid, 'seconds'),
                                'content_missing_rate': average(valid, 'content_missing'),
                                'truncation_rate': average(valid, 'truncated'),
                                'citation_problem_rate': average(valid, 'citation_problem')},
                       denominators={'retrieval_positive': sum(r.get('recall') is not None for r in valid),
                                     'retrieval_negative': sum(r.get('false_positive') is not None for r in valid),
                                     'generation': sum(r.get('generation_performed', False) for r in valid),
                                     'automatic_responses': sum('response' in r and not r.get('generation_performed', False) for r in valid)})
    except Exception as exc:
        summary.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    write_json(output / 'summary.json', summary)
    return summary


def report(run):
    """Transforma o resumo da avaliação em Markdown, mantendo explícitos os limites das métricas."""
    run = Path(run)
    summary = read_json(run / 'summary.json')
    lines = ['# Evidências da avaliação', '', f"Status: {summary['status']}",
             f"Modo: {summary['mode']}; conjunto: {summary['split']}",
             f"Cenários previstos: {summary['planned']}; concluídos: {summary.get('completed', 0)}; falhas: {summary.get('failed', 'não apurado')}",
             '', '| Métrica | Valor |', '|---|---:|']
    for key, value in summary.get('metrics', {}).items():
        lines.append(f'| {key} | {value if value is not None else "não aplicável"} |')
    lines += ['', 'Denominadores: ' + json.dumps(summary.get('denominators', {})), '',
              'Métricas agregadas usam somente casos concluídos; falhas são informadas separadamente.',
              'Revisão humana não foi consolidada automaticamente. Consulte human_review.jsonl.',
              'Citações e métricas de busca não comprovam fidelidade ou correção clínica.',
              'Não há conclusão de melhoria sobre o modelo base nesta avaliação.',
              'Configuração, versões e hashes estão em summary.json; saídas individuais em results.jsonl.']
    target = run / 'report.md'
    with target.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    return str(target)

