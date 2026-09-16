"""Diagnóstico sem baixar modelos nem instalar/alterar drivers."""
import importlib.metadata
import platform
import subprocess

PACKAGES = ["torch", "transformers", "peft", "accelerate", "bitsandbytes"]


def diagnose():
    """Inspeciona bibliotecas e CUDA; disponibilidade não garante memória suficiente para treinar."""
    result = {"python": platform.python_version(), "platform": platform.system(),
              "packages": {}, "ready_for_probe": False, "issues": []}
    for package in PACKAGES:
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
            result["issues"].append(f"Instale a dependência: {package}")
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        result["nvidia_smi"] = query.stdout.strip() if query.returncode == 0 else "indisponível"
    except (OSError, subprocess.TimeoutExpired):
        result["nvidia_smi"] = "comando indisponível (não prova ausência de GPU)"
    try:
        import torch
        result["torch_cuda_build"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if not result["cuda_available"]:
            result["issues"].append("CUDA indisponível: verificar driver NVIDIA e instalação PyTorch CUDA")
        else:
            prop = torch.cuda.get_device_properties(0)
            result["gpu"] = {"name": prop.name, "vram_mib": round(prop.total_memory / 2**20),
                             "compute_capability": [prop.major, prop.minor],
                             "torch_architectures": torch.cuda.get_arch_list()}
            # Executa um kernel: cuda.is_available por si só não valida a arquitetura.
            torch.ones(1, device="cuda").add_(1)
            torch.cuda.synchronize()
            if prop.major < 6:
                result["issues"].append("Este perfil foi preparado para compute capability >= 6")
    except Exception as exc:
        result["issues"].append(f"Falha ao inicializar PyTorch/CUDA: {type(exc).__name__}: {exc}")
    result["ready_for_probe"] = not result["issues"]
    result["note"] = "Diagnóstico não garante que QLoRA caiba na VRAM; execute probe para medir."
    return result


def require_cuda():
    """Interrompe a operação quando o diagnóstico não permite usar a GPU CUDA."""
    report = diagnose()
    if not report["ready_for_probe"]:
        raise ValueError("Ambiente não pronto. Execute hospital-train doctor e consulte issues.")
    return report

