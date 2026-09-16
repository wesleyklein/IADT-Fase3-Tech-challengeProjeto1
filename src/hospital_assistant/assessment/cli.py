"""Avaliação reproduzível e relatório das evidências geradas."""
import argparse
import json
import sys


def main():
    """Interpreta os argumentos do terminal e encaminha a operação ao módulo responsável."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('retrieve', 'generate', 'compare'):
        p = sub.add_parser(name)
        p.add_argument('--config', default='config/rag.json')
        p.add_argument('--cases', default='data/evaluation/rag_cases.json')
        p.add_argument('--split', choices=['validation', 'test'], default='validation')
        p.add_argument('--output', required=True)
    p = sub.add_parser('report')
    p.add_argument('--run', required=True)
    args = parser.parse_args()
    try:
        from .runner import evaluate, report
        if args.command == 'report':
            print(report(args.run))
            return 0
        if args.command == 'compare':
            from .comparison import compare
            result = compare(args.config, args.cases, args.split, args.output)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result['status'] == 'completed' else 1
        result = evaluate(args.config, args.cases, args.split, args.output, args.command == 'generate')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['status'] == 'completed' else 1
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    sys.exit(main())

