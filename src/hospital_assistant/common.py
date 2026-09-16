"""Serialização e limpeza sem modificar doses, unidades ou negações."""
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path


def clean_text(text: str) -> str:
    """Normaliza Unicode, entidades HTML e espaços, preservando acentos e negações."""
    return " ".join(unicodedata.normalize("NFC", html.unescape(text)).split())


def clean_html(text: str) -> str:
    # BeautifulSoup é importado apenas para entradas HTML; a leitura XML segue outro caminho.
    """Remove marcação, scripts e estilos antes de normalizar o texto visível."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return clean_text(soup.get_text(" "))


def digest(value: str) -> str:
    """Calcula SHA-256 para identificar conteúdo; hash não é criptografia reversível."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def redact_identifiers(text: str) -> tuple[str, dict]:
    """Proteção parcial: não detecta nomes, endereços ou todos identificadores."""
    counts = {}
    for name, pattern in {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "cpf": r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)",
        "phone_br": r"\(\d{2}\)\s*\d{4,5}-\d{4}\b",
    }.items():
        text, count = re.subn(pattern, f"[{name.upper()}_REMOVIDO]", text)
        if count:
            counts[name] = count
    return text, counts


def read_json(path):
    """Lê um documento JSON em UTF-8 e devolve seus objetos Python."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    """Lê um objeto JSON por linha, ignorando linhas vazias."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    """Cria a pasta necessária e grava JSON legível, preservando acentos."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    """Grava registros independentes, um por linha, para processamento de datasets."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

