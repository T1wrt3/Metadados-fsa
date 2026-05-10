import csv
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ELA_QUALIDADE = 90
ELA_FATOR_AMPLIFICACAO = 15
ELA_VALOR_MAX = 255

PROJETO = Path(__file__).parent.resolve()
BASE_DATASET = PROJETO / "archive" / "real_vs_fake"
SAIDA_DIR = PROJETO / "archive"

COLUNAS = [
    "arquivo",
    "rotulo",
    "ela_media",
    "ela_desvio",
    "variancia_ruido",
    "fft_simetria",
    "corr_rg",
    "corr_rb",
    "corr_gb",
    "aberracao_cromatica",
    "gradiente_media",
    "gradiente_desvio",
]


def _ela_features(conteudo):
    try:
        imagem_original = Image.open(io.BytesIO(conteudo)).convert("RGB")

        buffer_recomp = io.BytesIO()
        imagem_original.save(buffer_recomp, format="JPEG", quality=ELA_QUALIDADE)
        buffer_recomp.seek(0)
        imagem_recomprimida = Image.open(buffer_recomp).convert("RGB")

        diferenca = ImageChops.difference(imagem_original, imagem_recomprimida)
        arr = np.array(diferenca, dtype=np.float32)
        arr = np.clip(arr * ELA_FATOR_AMPLIFICACAO, 0, ELA_VALOR_MAX)

        return float(np.mean(arr)), float(np.std(arr))
    except Exception:
        return 0.0, 0.0


def _ruido_variancia(img_cinza):
    kernel_srm = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) / 4.0
    residuo = cv2.filter2D(img_cinza, -1, kernel_srm)

    tamanho_bloco = 16
    h, w = residuo.shape
    n_linhas = h // tamanho_bloco
    n_colunas = w // tamanho_bloco

    if n_linhas == 0 or n_colunas == 0:
        return 0.0

    variancias = []
    for idx_linha in range(n_linhas):
        for idx_coluna in range(n_colunas):
            r = idx_linha * tamanho_bloco
            c = idx_coluna * tamanho_bloco
            bloco = residuo[r:r + tamanho_bloco, c:c + tamanho_bloco]
            variancias.append(float(np.var(bloco)))

    return float(np.mean(variancias))


def _fft_simetria(img_cinza):
    f = np.fft.fft2(img_cinza.astype(np.float32))
    f_deslocado = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(f_deslocado))

    h, w = magnitude.shape
    metade_superior = magnitude[:h // 2, :]
    metade_inferior = np.flipud(magnitude[h // 2:, :])
    min_h = min(metade_superior.shape[0], metade_inferior.shape[0])
    diff = np.abs(metade_superior[:min_h] - metade_inferior[:min_h])
    simetria = float(1.0 - (np.mean(diff) / (np.max(magnitude) + 1e-9)))

    return simetria


def _correlacao_rgb(img_cv):
    b, g, r = cv2.split(img_cv.astype(np.float32))

    def corr(a, b_):
        a_plano = a.flatten()
        b_plano = b_.flatten()
        if np.std(a_plano) < 1e-9 or np.std(b_plano) < 1e-9:
            return 1.0
        return float(np.corrcoef(a_plano, b_plano)[0, 1])

    return corr(r, g), corr(r, b), corr(g, b)


def _aberracao_cromatica(img_cv):
    img_suavizada = cv2.GaussianBlur(img_cv, (3, 3), 0)
    b, g, r = cv2.split(img_suavizada)

    bordas_r = cv2.Canny(r, 30, 90)
    bordas_g = cv2.Canny(g, 30, 90)
    bordas_b = cv2.Canny(b, 30, 90)

    mascara = (bordas_r > 0) | (bordas_g > 0) | (bordas_b > 0)
    total_pixels = float(np.count_nonzero(mascara)) + 1e-9

    diff_rg = np.count_nonzero(cv2.bitwise_xor(bordas_r, bordas_g) & mascara)
    diff_rb = np.count_nonzero(cv2.bitwise_xor(bordas_r, bordas_b) & mascara)
    diff_gb = np.count_nonzero(cv2.bitwise_xor(bordas_g, bordas_b) & mascara)

    desalinhamento_total = float(diff_rg + diff_rb + diff_gb)
    return desalinhamento_total / (2.0 * total_pixels)


def _gradiente_features(img_cv):
    img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
    canal_y, _, _ = cv2.split(img_yuv)

    sobel_x = cv2.Sobel(canal_y, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(canal_y, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    return float(np.mean(magnitude)), float(np.std(magnitude))


def extrair_de_imagem(caminho):
    try:
        conteudo = caminho.read_bytes()

        imagem_np = np.frombuffer(conteudo, np.uint8)
        img_cv = cv2.imdecode(imagem_np, cv2.IMREAD_COLOR)
        if img_cv is None:
            return None

        img_cv = cv2.resize(img_cv, (256, 256))
        conteudo_redim = cv2.imencode(".jpg", img_cv)[1].tobytes()

        img_cinza = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        ela_media, ela_desvio = _ela_features(conteudo_redim)
        variancia = _ruido_variancia(img_cinza)
        simetria = _fft_simetria(img_cinza)
        corr_rg, corr_rb, corr_gb = _correlacao_rgb(img_cv)
        aber = _aberracao_cromatica(img_cv)
        grad_media, grad_desvio = _gradiente_features(img_cv)

        return [
            ela_media,
            ela_desvio,
            variancia,
            simetria,
            corr_rg,
            corr_rb,
            corr_gb,
            aber,
            grad_media,
            grad_desvio,
        ]
    except Exception as exc:
        logger.warning("Falha ao processar %s: %s", caminho.name, exc)
        return None


def processar_pasta(conjunto):
    pasta_real = BASE_DATASET / conjunto / "real"
    pasta_fake = BASE_DATASET / conjunto / "fake"
    arquivo_saida = SAIDA_DIR / f"features_{conjunto}.csv"

    if not pasta_real.exists() or not pasta_fake.exists():
        print(f"Pasta {conjunto} nao encontrada, pulando...")
        return

    imagens = []
    for img in pasta_real.glob("*"):
        if img.is_file():
            imagens.append((img, 0))
    for img in pasta_fake.glob("*"):
        if img.is_file():
            imagens.append((img, 1))

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)

    with open(arquivo_saida, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(COLUNAS)

        for caminho, rotulo in tqdm(imagens, desc=f"Processando {conjunto}"):
            features = extrair_de_imagem(caminho)
            if features is None:
                continue
            linha = [caminho.name, rotulo] + [round(v, 6) for v in features]
            escritor.writerow(linha)

    logger.info("Salvo em %s", arquivo_saida)


if __name__ == "__main__":
    for conjunto in ["train", "valid", "test"]:
        processar_pasta(conjunto)
    print("Features extraidas com sucesso.")