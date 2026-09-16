"""Execução e revisão persistente do fluxo hospitalar didático."""
import argparse
import json
import sys


def main():
    """Interpreta os argumentos do terminal e encaminha a operação ao módulo responsável."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', default='data/local/workflows.sqlite3')
    parser.add_argument('--database', default='data/local/hospital.sqlite3')
    parser.add_argument('--rag-config', default='config/rag.json')
    sub = parser.add_subparsers(dest='command', required=True)
    start = sub.add_parser('start')
    start.add_argument('action', choices=['protocol', 'patient', 'exams', 'report', 'prescription'])
    start.add_argument('--question', default='')
    start.add_argument('--patient-id', default='')
    show = sub.add_parser('show')
    show.add_argument('thread')
    review = sub.add_parser('review')
    review.add_argument('thread')
    review.add_argument('--decision', required=True, choices=['approve', 'reject'])
    review.add_argument('--reviewer', required=True)
    review.add_argument('--draft-hash', required=True)
    review.add_argument('--comment', default='')
    args = parser.parse_args()
    try:
        from .service import WorkflowService
        service = WorkflowService(args.storage, args.database, args.rag_config)
        if args.command == 'start':
            result = service.start(args.action, args.question, args.patient_id)
        elif args.command == 'show':
            result = service.show(args.thread)
        else:
            result = service.review(args.thread, args.decision, args.reviewer, args.draft_hash, args.comment)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result['status'] in ('failed', 'blocked') else 0
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    sys.exit(main())

