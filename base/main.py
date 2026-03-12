import os
import tempfile
from pathlib import Path
import exiftool

TIPOS_IMAGEM = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tiff", ".tif", ".gif", ".bmp", ".raw"}
TIPOS_VIDEO  = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}

BINARIOS = {
    "item 0", "item 1", "item 2", "item 3",
    "item 1 sig tst 2 tst tokens val",
    "item 1r vals ocsp vals",
    "item 1 pad", "item 1 pad 2",
    "created assertions hash",
    "hash", "pad",
}


def _detectar_tipo(nome):
    ext = Path(nome).suffix.lower()
    if ext in TIPOS_IMAGEM:
        return "imagem"
    if ext in TIPOS_VIDEO:
        return "video"
    raise ValueError(f"Tipo nao suportado: {ext}")``


def _normalizar(dados):
    resultado = {}
    for chave, valor in dados.items():
        chave_clean = chave.split(":")[-1]
        if chave_clean.lower() in {"sourcefile", "exiftoolversion"}:
            continue
        if chave_clean.lower() in BINARIOS:
            continue
        if isinstance(valor, str) and "(Binary data" in valor:
            continue
        resultado[chave_clean] = valor
    return resultado
 

def extrair_metadados(conteudo: bytes, nome_arquivo: str) -> dict:
    tipo = _detectar_tipo(nome_arquivo)
    suffix = Path(nome_arquivo).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(conteudo)
        tmp_path = tmp.name

    try:
        with exiftool.ExifToolHelper() as et:
            dados_brutos = et.get_metadata(tmp_path)
            meta = _normalizar(dados_brutos[0]) if dados_brutos else {}
    finally:
        os.unlink(tmp_path)

    meta["_file_name"]    = nome_arquivo
    meta["_file_type"]    = tipo
    meta["_file_size_kb"] = round(len(conteudo) / 1024, 2)

    return meta



#   http://localhost:5000/