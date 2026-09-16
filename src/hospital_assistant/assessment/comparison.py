"""Comparação 2x2: contexto comum, base/adaptador e prompt atual/curto."""
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from importlib.metadata import version
from ..common import read_json, write_json
from ..rag.chain import PROMPTS, RagAssistant



def prompt_values(question, docs):
    """Monta os campos pergunta/contexto que serão inseridos nos templates comparados."""
    return {'question': question, 'context': '\n\n'.join(f"[{d['protocol_id']}] {d['text']}" for d in docs)}


def shared_context(question, retrieved, model):
    """Seleciona um mesmo contexto que caiba nos dois prompts, tornando a comparação pareada."""
    used = []
    for doc in retrieved:
        values = prompt_values(question, used + [doc])
        if all(model.fits(prompt.invoke(values)) for prompt in PROMPTS.values()):
            used.append(doc)
    return used


class FrozenRetriever:
    """Reutiliza fontes já selecionadas, sem repetir a busca durante a comparação."""
    def __init__(self, docs):
        """Inicializa as dependências e a configuração utilizadas pelos métodos desta classe."""
        self.docs = docs

    def search(self, question):
        """Recupera os trechos mais próximos da pergunta e aplica o limite de distância configurado."""
        return self.docs


def summarize(rows):
    """Agrega métricas somente das gerações concluídas e conta abstinências automáticas à parte."""
    generated = [r for r in rows if r.get('response', {}).get('generation_performed')]
    completed = [r for r in rows if r['status'] == 'completed']
    return {'completed': len(completed), 'failed': len(rows)-len(completed),
            'generation': len(generated), 'automatic_responses': len(completed)-len(generated),
            'citation_problem_rate': sum(not r['response'].get('cited_sources') or bool(r['response'].get('invalid_citations')) for r in generated)/len(generated) if generated else None,
            'content_missing_rate': sum(r['response'].get('content_missing', False) for r in generated)/len(generated) if generated else None,
            'truncation_rate': sum(r['response']['truncated'] for r in generated)/len(generated) if generated else None,
            'mean_generation_seconds': sum(r['seconds'] for r in generated)/len(generated) if generated else None}


def compare(config_path, cases_path, split, output):
    """Executa base/adaptador com dois prompts, mantendo as fontes iguais por pergunta."""
    from ..rag.index import Retriever
    from ..rag.model import LocalModel
    config = read_json(config_path)
    all_cases = read_json(cases_path)
    if len({c['id'] for c in all_cases}) != len(all_cases):
        raise ValueError('IDs repetidos')
    cases = [c for c in all_cases if c['split'] == split]
    if not cases:
        raise ValueError('Nenhum caso selecionado')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'mode': 'comparison_2x2', 'split': split,
               'config': config, 'planned_cases': len(cases), 'planned_rows': len(cases)*4,
               'clinical_review': 'pending'}
    write_json(output / 'summary.json', summary)
    try:
        paths = [Path(config_path), Path(cases_path), Path(config['index'])/'index.json',
                 Path(config['run'])/'resolved_config.json'] + sorted((Path(config['run'])/'adapter').glob('*'))
        summary['files_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
        summary['packages'] = {p: version(p) for p in ('torch', 'transformers', 'peft', 'langchain-core')}
        summary['prompts'] = {name: [{'role': m.type, 'content': m.content} for m in prompt.invoke(
            {'question': '{question}', 'context': '{context}'}).to_messages()] for name, prompt in PROMPTS.items()}
        retriever, model = Retriever(config), LocalModel({**config, 'model_variant': 'adapted'})
        summary['model'] = model.identity
        summary['generation_config'] = model.generation.to_dict()
        rows = []
        with (output/'results.jsonl').open('x', encoding='utf-8') as stream:
            for case in cases:
                retrieved = retriever.search(case['question'])
                used = shared_context(case['question'], retrieved, model)
                # O mesmo hash nas quatro variantes comprova igualdade do contexto selecionado.
                context_hash = hashlib.sha256(json.dumps(used, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                for prompt_name, prompt in PROMPTS.items():
                    for variant in ('base', 'adapted'):
                        print(f"{case['id']} / {prompt_name} / {variant}", flush=True)
                        row = {'id': case['id'], 'question': case['question'], 'prompt': prompt_name,
                               'variant': variant, 'context_sha256': context_hash,
                               'retrieved_sources': retrieved, 'context_sources': used}
                        start = time.perf_counter()
                        try:
                            with model.model.disable_adapter() if variant == 'base' else nullcontext():
                                response = RagAssistant(FrozenRetriever(used), model, prompt).ask(case['question'])
                            response['prompt_variant'] = prompt_name
                            response['model'] = {**model.identity, 'adapter_enabled': variant == 'adapted'}
                            if not used and retrieved:
                                response['reason'] = 'context_budget'
                            row.update(status='completed', response=response)
                        except Exception as exc:
                            row.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                        row['seconds'] = time.perf_counter()-start
                        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                        stream.flush()
                        rows.append(row)
        summary['variants'] = {f'{p}/{v}': summarize([r for r in rows if r['prompt']==p and r['variant']==v])
                               for p in PROMPTS for v in ('base','adapted')}
        summary['status'] = 'completed' if all(r['status']=='completed' for r in rows) else 'partial'
        with (output/'human_review.jsonl').open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps({**row, 'reviewer': None, 'reviewed_at': None,
                    'answers_question': None, 'supported_by_sources': None,
                    'comments': '', 'review_status': 'pending'}, ensure_ascii=False)+'\n')
    except Exception as exc:
        summary.update(status='failed', error_type=type(exc).__name__, error=str(exc))
    write_json(output/'summary.json', summary)
    return summary

