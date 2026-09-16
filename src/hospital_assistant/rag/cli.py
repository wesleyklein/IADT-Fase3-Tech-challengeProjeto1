"""Indexação, inspeção da busca e consulta única ou chat pelo terminal."""
import argparse
import json
import sys
from pathlib import Path
from ..common import read_json


def emit(result, output=None):
    """Exibe JSON e, quando solicitado, grava uma cópia sem sobrescrever a saída existente."""
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text + '\n')
    print(text)


def main():
    """Interpreta os argumentos do terminal e encaminha a operação ao módulo responsável."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/rag.json')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('index')
    for command in ('retrieve', 'ask'):
        p = sub.add_parser(command)
        p.add_argument('question')
        p.add_argument('--output')
    sub.add_parser('chat')
    args = parser.parse_args()
    try:
        from .index import build, Retriever, validate
        config = read_json(args.config)
        validate(config)
        if getattr(args, 'output', None) and Path(args.output).exists():
            raise ValueError('Arquivo de saída já existe; escolha outro nome')
        if args.command == 'index':
            emit(build(config))
            return 0
        retriever = Retriever(config)
        if args.command == 'retrieve':
            emit({'question': args.question, 'sources': retriever.search(args.question)}, args.output)
            return 0
        from .model import LocalModel
        from .chain import RagAssistant
        assistant = RagAssistant(retriever, LocalModel(config))
        if args.command == 'ask':
            emit(assistant.ask(args.question), args.output)
        else:
            print('Chat didático. Cada pergunta é independente. Digite /sair para encerrar.')
            while True:
                try:
                    question = input('Você: ').strip()
                except EOFError:
                    break
                if question == '/sair':
                    break
                if question:
                    emit(assistant.ask(question))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        emit({'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)})
        return 1


if __name__ == '__main__':
    sys.exit(main())

