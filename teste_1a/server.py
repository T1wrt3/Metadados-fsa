from pathlib import Path
from flask import Flask, request, jsonify, send_file
from functools import wraps
from dotenv import load_dotenv
from main import (
    extrair_metadados,
    ela,
    verificar_c2pa,
    reverse_search,
    analise,
    calcular_score_alerta,
)
import os

load_dotenv()

app = Flask(__name__)
BASE_DIR: Path = Path(__file__).parent.resolve()

TIPOS_IMAGEM: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
    ".tiff", ".tif", ".gif", ".bmp", ".raw",
})


def require_file(f):
    """Decorator para validar e extrair o arquivo da requisição."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "file" not in request.files:
            return jsonify({"erro": "Nenhum arquivo enviado."}), 400

        arquivo = request.files["file"]
        if not arquivo.filename:
            return jsonify({"erro": "Nome de arquivo vazio."}), 400

        conteudo: bytes = arquivo.read()
        nome: str = arquivo.filename
        return f(conteudo, nome, *args, **kwargs)
    return decorated_function


@app.route("/")
def index():
    return send_file(BASE_DIR / "app.html")


@app.route("/analyze/quick", methods=["POST"])
@require_file
def analyze_quick(conteudo: bytes, nome: str):
    try:
        exif = extrair_metadados(conteudo, nome)
        return jsonify({"exif": exif})
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    except Exception as e:
        return jsonify({"erro": f"Erro inesperado: {e}"}), 500


@app.route("/analyze/deep", methods=["POST"])
@require_file
def analyze_deep(conteudo: bytes, nome: str):
    try:
        sufixo: str = Path(nome).suffix.lower()
        e_imagem: bool = sufixo in TIPOS_IMAGEM

        if not e_imagem:
            return jsonify({"erro": "Analise profunda disponivel apenas para imagens."}), 400

        exif = extrair_metadados(conteudo, nome)
        ela_b64 = ela(conteudo, sufixo)
        c2pa = verificar_c2pa(conteudo, sufixo)
        forense_avancada = analise(conteudo)

        alerta = calcular_score_alerta(exif, c2pa, forense_avancada)

        return jsonify({
            "ela": ela_b64,
            "forense_avancada": forense_avancada,
            "c2pa": c2pa,
            "alerta": alerta,
        })
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    except Exception as e:
        return jsonify({"erro": f"Erro inesperado: {e}"}), 500


@app.route("/reverse", methods=["POST"])
@require_file
def reverse(conteudo: bytes, nome: str):
    try:
        resultados = reverse_search(conteudo)
        return jsonify({"resultados": resultados})
    except Exception as e:
        return jsonify({"erro": f"Erro inesperado: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
