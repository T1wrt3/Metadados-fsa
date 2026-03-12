from pathlib import Path
from flask import Flask, request, jsonify, send_file
from main import extrair_metadados, ela, score_ia, verificar_c2pa, reverse_search
import os

app = Flask(__name__)
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

TIPOS_IMAGEM: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
    ".tiff", ".tif", ".gif", ".bmp", ".raw",
})


@app.route("/")
def index():
    """Serve o frontend principal."""
    return send_file(os.path.join(BASE_DIR, "app.html"))


@app.route("/metadata", methods=["POST"])
def metadata():
    """Extrai e retorna metadados EXIF/XMP do arquivo enviado."""
    if "file" not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado."}), 400

    arquivo = request.files["file"]

    if not arquivo.filename:
        return jsonify({"erro": "Nome de arquivo vazio."}), 400

    try:
        conteudo = arquivo.read()
        meta = extrair_metadados(conteudo, arquivo.filename)
        return jsonify(meta)
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    """Executa analise completa: EXIF, ELA, score de IA e verificacao C2PA."""
    if "file" not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado."}), 400

    arquivo = request.files["file"]

    if not arquivo.filename:
        return jsonify({"erro": "Nome de arquivo vazio."}), 400

    try:
        conteudo: bytes = arquivo.read()
        nome: str = arquivo.filename
        sufixo: str = Path(nome).suffix.lower()
        e_imagem: bool = sufixo in TIPOS_IMAGEM

        exif = extrair_metadados(conteudo, nome)

        ela_b64 = None
        c2pa: dict = {}

        if e_imagem:
            try:
                ela_b64 = ela(conteudo, sufixo)
            except Exception:
                ela_b64 = None

            c2pa = verificar_c2pa(conteudo, sufixo)

        score = score_ia(conteudo) if e_imagem else {}

        return jsonify({
            "exif": exif,
            "ela": ela_b64,
            "score_ia": score,
            "c2pa": c2pa,
        })
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/reverse", methods=["POST"])
def reverse():
    """Executa busca reversa da imagem enviada via SerpAPI."""
    if "file" not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado."}), 400

    arquivo = request.files["file"]

    if not arquivo.filename:
        return jsonify({"erro": "Nome de arquivo vazio."}), 400

    try:
        conteudo: bytes = arquivo.read()
        resultados = reverse_search(conteudo)
        return jsonify({"resultados": resultados})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
