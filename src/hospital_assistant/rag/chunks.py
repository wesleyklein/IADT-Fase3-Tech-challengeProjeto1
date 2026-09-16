"""Trechos do corpo com título/versão repetidos e posições na fonte original."""
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def protocol_chunks(docs, chunk_size, chunk_overlap):
    """Divide o corpo dos protocolos e repete título/versão para contextualizar cada trecho."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, add_start_index=True)
    result = []
    for doc in docs:
        text = doc.page_content
        separator = re.search(r'\r?\n[ \t]*\r?\n', text)
        if separator is None:
            raise ValueError('Protocolo precisa de cabeçalho e corpo separados por linha vazia')
        header = text[:separator.start()].splitlines()
        title = header[0].strip()
        versions = [line.strip() for line in header if line.strip().startswith('Versão:')]
        if not title.startswith(doc.metadata['protocol_id'] + ' | ') or len(versions) != 1:
            raise ValueError('Título ou versão ausente/inválido no protocolo')
        body_start = separator.end()
        body = text[body_start:]
        if not body.strip():
            raise ValueError('Corpo do protocolo vazio')
        for part in splitter.create_documents([body]):
            metadata = {**doc.metadata, 'title': title, 'version': versions[0],
                        'start_index': body_start + part.metadata['start_index'],
                        'body_text': part.page_content}
            result.append(Document(page_content=f'{title}\n{versions[0]}\n{part.page_content}',
                                   metadata=metadata))
    return result

