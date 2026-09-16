"""Índice FAISS reconstruído de JSON: não desserializa pickle."""
import hashlib
import math
import re
from pathlib import Path
from ..common import read_json, write_json


def validate(config):
    """Verifica os campos obrigatórios antes de permitir a próxima operação."""
    for key in ('chunk_size', 'top_k', 'max_input_tokens', 'max_new_tokens'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} deve ser inteiro positivo')
    if not 0 <= config['chunk_overlap'] < config['chunk_size']:
        raise ValueError('chunk_overlap inválido')
    if not math.isfinite(config['max_distance']) or not 0 <= config['max_distance'] <= 4:
        raise ValueError('max_distance deve estar entre 0 e 4 (distância L2 quadrática)')
    if config['device'] not in ('cpu', 'cuda'):
        raise ValueError('device deve ser cpu ou cuda')


def embeddings(model, revision):
    """Cria o encoder multilíngue em CPU; vetores normalizados representam os trechos."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model,
        model_kwargs={'device': 'cpu', 'revision': revision, 'trust_remote_code': False},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 8})


def build(config):
    """Gera embeddings e salva o índice JSON com fontes, hashes e revisão do encoder."""
    from huggingface_hub import HfApi
    from langchain_community.document_loaders import TextLoader
    from .chunks import protocol_chunks
    validate(config)
    target = Path(config['index'])
    if target.exists():
        raise ValueError('Índice já existe; escolha outra pasta na configuração')
    files = sorted(Path(config['protocols']).glob('PR-*.txt'))
    if not files:
        raise ValueError('Nenhum protocolo encontrado')
    docs, hashes = [], {}
    for path in files:
        if not re.fullmatch(r'PR-\d{3}', path.stem):
            raise ValueError(f'Nome de protocolo inválido: {path.name}')
        loaded = TextLoader(str(path), encoding='utf-8').load()
        if not loaded[0].page_content.strip():
            raise ValueError(f'Protocolo vazio: {path.name}')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded[0].metadata = {'protocol_id': path.stem, 'source': path.name,
                              'sha256': hashes[path.name], 'synthetic': True}
        docs.extend(loaded)
    chunks = protocol_chunks(docs, config['chunk_size'], config['chunk_overlap'])
    revision = HfApi().model_info(config['embedding_model'], revision=config['embedding_revision']).sha
    encoder = embeddings(config['embedding_model'], revision)
    # Respeita também o limite do encoder; nunca aceita truncamento silencioso.
    from transformers import AutoTokenizer
    token_counter = AutoTokenizer.from_pretrained(config['embedding_model'], revision=revision)
    for doc in chunks:
        length = len(token_counter.encode(doc.page_content))
        if length > 128:
            raise ValueError('Trecho excede o encoder; reduza chunk_size')
    vectors = encoder.embed_documents([doc.page_content for doc in chunks])
    payload = {'schema': 2, 'chunk_strategy': 'protocol-title-version-body-v2', 'embedding_model': config['embedding_model'],
        'embedding_revision': revision, 'sources_sha256': hashes,
        'chunk_size': config['chunk_size'], 'chunk_overlap': config['chunk_overlap'],
        'chunks': [{'text': d.page_content, 'metadata': d.metadata, 'vector': v}
                   for d, v in zip(chunks, vectors)]}
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / 'index.json', payload)
    return {'status': 'completed', 'documents': len(docs), 'chunks': len(chunks),
            'index': str(target), 'embedding_revision': revision}


class Retriever:
    """Reconstrói a busca FAISS a partir de vetores JSON e verifica a integridade das fontes."""
    def __init__(self, config):
        """Inicializa as dependências e a configuração utilizadas pelos métodos desta classe."""
        from langchain_community.vectorstores import FAISS
        validate(config)
        data = read_json(Path(config['index']) / 'index.json')
        if data['schema'] not in (1, 2):
            raise ValueError('Versão de índice incompatível')
        current = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path(config['protocols']).glob('PR-*.txt')}
        if current != data['sources_sha256']:
            raise ValueError('Protocolos mudaram; crie um novo índice')
        if data['embedding_model'] != config['embedding_model']:
            raise ValueError('Modelo de embeddings difere do índice')
        from transformers import AutoTokenizer
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            data['embedding_model'], revision=data['embedding_revision'])
        encoder = embeddings(data['embedding_model'], data['embedding_revision'])
        chunks = data['chunks']
        self.store = FAISS.from_embeddings(
            [(c['text'], c['vector']) for c in chunks], encoder,
            metadatas=[c['metadata'] for c in chunks])
        self.config = config

    def search(self, question):
        """Recupera os trechos mais próximos da pergunta e aplica o limite de distância configurado."""
        if not question.strip():
            raise ValueError('Pergunta vazia')
        if len(self.query_tokenizer.encode(question)) > 128:
            raise ValueError('Pergunta excede 128 tokens do encoder; reformule de forma mais curta')
        return [{'text': doc.page_content, **doc.metadata, 'distance': float(score)}
                for doc, score in self.store.similarity_search_with_score(question, k=self.config['top_k'])
                if float(score) <= self.config['max_distance']]

