"""Rotas explícitas, ferramentas somente leitura e revisão via interrupt."""
import hashlib
import json
import operator
import re
from datetime import datetime, timezone
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from ..hospital import get_patient

ACTIONS = ('protocol', 'patient', 'exams', 'report', 'prescription')


def event(node, **details):
    return {'node': node, 'at': datetime.now(timezone.utc).isoformat(), **details}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class State(TypedDict, total=False):
    action: str
    question: str
    patient_id: str
    patient_data: dict
    draft: dict
    draft_hash: str
    decision: dict
    status: str
    error: str
    events: Annotated[list[dict], operator.add]


def build_graph(checkpointer, database, rag_factory):
    """Conecta paciente, protocolos, LLM e revisão humana em um grafo persistente."""
    @tool
    def patient_lookup(patient_id: str) -> dict:
        """Consulta paciente sintético pelo ID exato, incluindo exames registrados."""
        if not re.fullmatch(r'SYN-P\d{3}', patient_id):
            raise ValueError('Informe um ID no formato SYN-P001')
        return get_patient(database, patient_id)

    @tool
    def protocol_lookup(question: str, patient_data: dict) -> dict:
        """Responde com protocolos e com os dados atualizados do paciente no contexto da LLM."""
        return rag_factory().ask(question, patient_data=patient_data)

    def validate(state):
        """Toda ação exige paciente; protocolo também exige a pergunta que orienta o RAG."""
        error = None
        if state['action'] not in ACTIONS:
            error = 'Ação desconhecida'
        elif not re.fullmatch(r'SYN-P\d{3}', state.get('patient_id', '')):
            error = 'Informe um paciente sintético no formato SYN-P001'
        elif state['action'] == 'protocol' and not state.get('question', '').strip():
            error = 'Informe a pergunta sobre o paciente e os protocolos'
        return {'status': 'blocked' if error else 'running', 'error': error or '',
                'events': [event('validate', accepted=not bool(error))]}

    def lookup(state):
        try:
            data = patient_lookup.invoke({'patient_id': state['patient_id']})
            return {'patient_data': data, 'events': [event('patient_lookup', patient_id=state['patient_id'])]}
        except ValueError as exc:
            return {'status': 'blocked', 'error': str(exc), 'events': [event('patient_lookup', found=False)]}

    def draft(state):
        """Protocolos passam pela LLM com paciente; demais ações continuam determinísticas."""
        action = state['action']
        data = state['patient_data']
        if action == 'protocol':
            result = protocol_lookup.invoke({'question': state['question'], 'patient_data': data})
        else:
            result = {'kind': action, 'patient': data['patient'], 'synthetic': True,
                      'label': 'RASCUNHO DIDÁTICO — SEM VALIDADE CLÍNICA',
                      'source': {'tool': 'patient_lookup', 'snapshot_sha256': digest(data)}}
            if action in ('patient', 'exams', 'report'):
                result['exams'] = data['exams']
                result['pending_exams'] = [e['id'] for e in data['exams'] if e['status'] == 'pending']
            if action == 'report':
                result['limitations'] = 'Somente resultados registrados; sem interpretação diagnóstica.'
                result['missing_results'] = [e['id'] for e in data['exams'] if e['result'] is None]
                result['professional_review'] = '[PREENCHIMENTO DO PROFISSIONAL]'
            if action == 'prescription':
                result['form'] = {key: '[PREENCHIMENTO EXCLUSIVO DO PROFISSIONAL]'
                    for key in ('medicamento', 'dose', 'via', 'frequência', 'duração', 'assinatura')}
                result['limitations'] = 'Formulário vazio, sem medicamento ou esquema terapêutico.'
        return {'draft': result, 'draft_hash': digest(result), 'status': 'awaiting_review',
                'events': [event('draft', action=action)]}

    def review(state):
        decision = interrupt({'draft': state['draft'], 'draft_hash': state['draft_hash'],
                              'message': 'Revise o conteúdo e aprove ou rejeite o rascunho didático.'})
        if (not isinstance(decision, dict) or decision.get('decision') not in ('approve', 'reject')
                or not isinstance(decision.get('reviewer'), str) or not decision['reviewer'].strip()
                or decision.get('draft_hash') != state['draft_hash']):
            raise ValueError('Decisão inválida ou referente a outro rascunho')
        approved = decision['decision'] == 'approve'
        return {'decision': decision, 'status': 'approved_didactic' if approved else 'rejected',
                'events': [event('review', decision=decision['decision'], reviewer=decision['reviewer'],
                                 draft_hash=state['draft_hash'])]}

    graph = StateGraph(State)
    graph.add_node('validate', validate)
    graph.add_node('lookup', lookup)
    graph.add_node('prepare_draft', draft)
    graph.add_node('review', review)
    graph.add_edge(START, 'validate')
    graph.add_conditional_edges('validate', lambda s: 'end' if s['status'] == 'blocked' else 'lookup',
                                {'end': END, 'lookup': 'lookup'})
    graph.add_conditional_edges('lookup', lambda s: 'end' if s['status'] == 'blocked' else 'prepare_draft',
                                {'end': END, 'prepare_draft': 'prepare_draft'})
    graph.add_edge('prepare_draft', 'review')
    graph.add_edge('review', END)
    return graph.compile(checkpointer=checkpointer)
