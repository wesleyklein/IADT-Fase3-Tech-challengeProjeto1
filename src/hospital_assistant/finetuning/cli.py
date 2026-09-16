"""CLI de treinamento e avaliação com relatórios persistentes e saídas não sobrescritas."""
import argparse
import json
from pathlib import Path
from ..common import write_json
from .diagnostics import diagnose


def main():
    """Interpreta os argumentos do terminal e encaminha a operação ao módulo responsável."""
    parser = argparse.ArgumentParser(description="Diagnóstico e QLoRA do hospital acadêmico")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--output", default="runs/doctor.json")
    for name in ["probe", "train"]:
        p = sub.add_parser(name)
        p.add_argument("--config", default="config/qlora.json")
        p.add_argument("--output", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--limit", type=int, default=20)
    evaluation.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.command == "doctor":
        result = diagnose()
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ready_for_probe"] else 1
    output = Path(args.output)
    if output.exists():
        parser.error("Diretório de saída já existe; escolha outro para preservar a execução anterior")
    output.mkdir(parents=True)
    try:
        if args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(args.run, output, args.limit, args.max_new_tokens)
        else:
            from .runtime import train
            result = train(args.config, output, probe=args.command == "probe")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        error = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc),
                 "hint": "Consulte doctor.json. Se faltar VRAM no probe, use GPU maior; não é falha dos dados."}
        write_json(output / "failure.json", error)
        print(json.dumps(error, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

