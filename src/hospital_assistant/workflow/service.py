"""Serviço compartilhado por CLI/UI, checkpoints e exclusão de mutações concorrentes."""
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4
from filelock import FileLock
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from .graph import build_graph, event


class WorkflowService:
    """Fachada usada pela tela e CLI para iniciar, consultar e revisar fluxos persistidos."""
    def __init__(self, storage='data/local/workflows.sqlite3', database='data/local/hospital.sqlite3',
                 rag_config='config/rag.json', rag_overrides=None):
        """Inicializa as dependências e a configuração utilizadas pelos métodos desta classe."""
        self.storage, self.database, self.rag_config = Path(storage), database, rag_config
        self.rag_overrides = dict(rag_overrides or {})
        self._rag = None

    def rag(self):
        """Carrega o RAG sob demanda e reutiliza a instância nas solicitações deste serviço."""
        if self._rag is None:
            from ..common import read_json
            from ..rag.index import Retriever
            from ..rag.model import LocalModel
            from ..rag.chain import RagAssistant
            config = {**read_json(self.rag_config), **self.rag_overrides}
            self._rag = RagAssistant(Retriever(config), LocalModel(config))
        return self._rag

    @contextmanager
    def graph(self):
        """Abre checkpoints SQLite sob bloqueio de arquivo e fecha os recursos ao sair."""
        self.storage.parent.mkdir(parents=True, exist_ok=True)
        # Serializa start/resume entre processos locais e fecha SQLite no Windows.
        # O bloqueio atual engloba a execução do grafo, inclusive geração lenta.
        # Outra solicitação pode atingir o timeout; isto não indica corrupção do banco.
        with FileLock(str(self.storage) + '.lock', timeout=2):
            with SqliteSaver.from_conn_string(str(self.storage)) as saver:
                yield build_graph(saver, self.database, self.rag)

    @staticmethod
    def config(thread):
        """Valida o UUID e monta a configuração que identifica a conversa nos checkpoints."""
        UUID(thread)
        return {'configurable': {'thread_id': thread}, 'recursion_limit': 12}

    @staticmethod
    def snapshot(graph, config, thread):
        """Lê o estado persistido e os próximos nós, permitindo retomar a mesma solicitação."""
        state = graph.get_state(config)
        if not state.values:
            raise ValueError('Solicitação não encontrada')
        return {'thread_id': thread, **state.values, 'next_nodes': list(state.next)}

    def start(self, action, question='', patient_id=''):
        """Cria uma solicitação e executa o grafo até a pausa humana ou um estado de falha."""
        thread = str(uuid4())
        config = self.config(thread)
        with self.graph() as graph:
            try:
                graph.invoke({'action': action, 'question': question, 'patient_id': patient_id,
                              'events': [event('start')], 'status': 'created'}, config)
            except Exception as exc:
                graph.update_state(config, {'status': 'failed', 'error': str(exc),
                    'events': [event('failure', error_type=type(exc).__name__)]})
            return self.snapshot(graph, config, thread)

    def show(self, thread):
        """Recupera a solicitação existente sem executar novamente a geração."""
        with self.graph() as graph:
            return self.snapshot(graph, self.config(thread), thread)

    def review(self, thread, decision, reviewer, draft_hash, comment=''):
        """Registra a decisão humana vinculada ao hash do rascunho; uma rejeição encerra a proposta."""
        if decision not in ('approve', 'reject') or not reviewer.strip():
            raise ValueError('Informe decisão e nome do revisor')
        with self.graph() as graph:
            config = self.config(thread)
            state = self.snapshot(graph, config, thread)
            if state['status'] != 'awaiting_review' or state['next_nodes'] != ['review']:
                raise ValueError('Solicitação não aguarda revisão')
            if draft_hash != state['draft_hash']:
                raise ValueError('O rascunho mudou; consulte novamente antes de decidir')
            graph.invoke(Command(resume={'decision': decision, 'reviewer': reviewer.strip(),
                'comment': comment, 'draft_hash': draft_hash}), config)
            return self.snapshot(graph, config, thread)

