import os
import base64
import logging
import tempfile
from pathlib import Path
from typing import Any

import exiftool
import numpy as np
import requests
from PIL import Image, ImageChops
import io
import cv2
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

TIPOS_IMAGEM = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
    ".tiff", ".tif", ".gif", ".bmp", ".raw",
})

TIPOS_VIDEO = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
})

CAMPOS_BINARIOS = frozenset({
    "item 0", "item 1", "item 2", "item 3",
    "item 1 sig tst 2 tst tokens val",
    "item 1r vals ocsp vals",
    "item 1 pad", "item 1 pad 2",
    "created assertions hash",
    "hash", "pad",
})

CAMPOS_IGNORADOS = frozenset({"sourcefile", "exiftoolversion"})

ELA_QUALIDADE = 90
ELA_FATOR_AMPLIFICACAO = 15
ELA_VALOR_MAX = 255

SERPAPI_URL = "https://serpapi.com/search"
SERPAPI_ENGINE = "google_reverse_image"
SERPAPI_MAX_RESULTADOS = 5

CAMPOS_IA = {
    "software": [
        "stable diffusion", "midjourney", "dall-e", "dalle",
        "novelai", "comfyui", "automatic1111", "invoke",
        "diffusers", "dreamstudio", "leonardo", "playground",
        "adobe firefly", "firefly", "bing image creator",
        "copilot", "chatgpt", "gpt", "gemini", "ideogram",
    ],
    "chaves": [
        "software", "generator", "creator", "creatortool",
        "historysoftwareagent", "comment", "usercomment",
        "description", "imagedescription", "xmptoolkit",
        "model", "source", "aimodel", "prompt",
        "negativeprompt", "sampler", "cfgscale", "steps",
        "seed", "denoisingstrength", "clipskip",
        "aigeneratedcontent", "aiassisted",
    ],
}

CAMPOS_CAMERA = [
    "make", "model", "lensmodel", "lensmake", "lensinfo",
    "focallength", "focallengthin35mmformat",
    "fnumber", "aperture", "aperturevalue",
    "exposuretime", "shutterspeedvalue", "shutterspeed",
    "iso", "isospeedratings", "photographicsensitivity",
    "flash", "flashfired", "flashmode",
    "whitebalance", "meteringmode", "exposuremode",
    "exposureprogram", "exposurecompensation",
    "scenecapturetype", "digitalzoomratio",
    "gaincontrol", "contrast", "saturation", "sharpness",
    "subjectdistance", "subjectdistancerange",
    "serialnumber", "bodyserialnumber", "lensserialnumber",
    "internalserialnumber",
]

CAMPOS_ARQUIVO = [
    "filename", "directory", "filesize", "filemodifydate",
    "fileaccessdate", "filecreatedate", "filetype",
    "filetypeextension", "mimetype", "imagewidth",
    "imageheight", "imagesize", "megapixels",
    "bitdepth", "colorspace", "colortype",
    "compression", "quality", "encoding",
    "xresolution", "yresolution", "resolutionunit",
    "datetimeoriginal", "createdate", "modifydate",
    "offsettime", "offsettimeoriginal",
    "gpslatitude", "gpslongitude", "gpsaltitude",
    "gpsposition",
]


def _carregar_chaves_serpapi():
    chaves = []
    indice = 1
    while True:
        chave = os.environ.get(f"SERPAPI_KEY_{indice}", "")
        if not chave:
            break
        chaves.append(chave)
        indice += 1
    if not chaves:
        chave_unica = os.environ.get("SERPAPI_KEY", "")
        if chave_unica:
            chaves.append(chave_unica)
    return chaves


def _testar_chave_serpapi(chave):
    try:
        resposta = requests.get(
            "https://serpapi.com/account",
            params={"api_key": chave},
            timeout=10,
        )
        if resposta.status_code != 200:
            return False
        dados = resposta.json()
        restantes = dados.get("total_searches_left", 0)
        return restantes > 0
    except Exception:
        return False


def _obter_chave_serpapi_valida():
    chaves = _carregar_chaves_serpapi()
    if not chaves:
        raise ValueError("Nenhuma chave SerpApi configurada no .env")
    for chave in chaves:
        if _testar_chave_serpapi(chave):
            return chave
    raise ValueError("Todas as chaves SerpApi estao esgotadas")


def _detectar_tipo(nome):
    ext = Path(nome).suffix.lower()
    if ext in TIPOS_IMAGEM:
        return "imagem"
    if ext in TIPOS_VIDEO:
        return "video"
    raise ValueError("Tipo nao suportado")


def _normalizar(dados):
    resultado = {}
    for chave, valor in dados.items():
        chave_limpa = chave.split(":")[-1]
        if chave_limpa.lower() in CAMPOS_IGNORADOS:
            continue
        if chave_limpa.lower() in CAMPOS_BINARIOS:
            continue
        if isinstance(valor, str) and "(Binary data" in valor:
            continue
        resultado[chave_limpa] = valor
    return resultado


def extrair_metadados(conteudo, nome_arquivo):
    tipo = _detectar_tipo(nome_arquivo)
    sufixo = Path(nome_arquivo).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as tmp:
        tmp.write(conteudo)
        caminho_tmp = tmp.name

    meta = {}
    try:
        with exiftool.ExifToolHelper() as et:
            dados_brutos = et.get_metadata(caminho_tmp)
            meta = _normalizar(dados_brutos[0]) if dados_brutos else {}
    except Exception as exc:
        logger.error("Falha ao extrair metadados com ExifTool: %s", exc)
    finally:
        os.unlink(caminho_tmp)

    meta["_nome_arquivo"] = nome_arquivo
    meta["_tipo_arquivo"] = tipo
    meta["_tamanho_kb"] = round(len(conteudo) / 1024, 2)

    return meta


def categorizar_metadados(meta):
    ia = {}
    camera = {}
    arquivo = {}
    outros = {}
    indicios_ia = []

    for chave, valor in meta.items():
        if chave.startswith("_"):
            arquivo[chave] = valor
            continue

        chave_lower = chave.lower().replace(" ", "").replace("_", "")

        encontrado = False

        for campo_ia in CAMPOS_IA["chaves"]:
            if campo_ia in chave_lower:
                ia[chave] = valor
                encontrado = True

                if isinstance(valor, str):
                    valor_lower = valor.lower()
                    for termo in CAMPOS_IA["software"]:
                        if termo in valor_lower:
                            indicios_ia.append({
                                "campo": chave,
                                "valor": valor,
                                "motivo": f"Contem referencia a ferramenta de IA: {termo}",
                            })
                            break

                    palavras_suspeitas = [
                        "generated", "artificial", "synthetic", "ai",
                        "neural", "diffusion", "gan",
                    ]
                    for palavra in palavras_suspeitas:
                        if palavra in valor_lower:
                            indicios_ia.append({
                                "campo": chave,
                                "valor": valor,
                                "motivo": f"Contem termo associado a geracao artificial: {palavra}",
                            })
                            break
                break

        if encontrado:
            continue

        for campo_cam in CAMPOS_CAMERA:
            if campo_cam in chave_lower:
                camera[chave] = valor
                encontrado = True
                break

        if encontrado:
            continue

        for campo_arq in CAMPOS_ARQUIVO:
            if campo_arq in chave_lower:
                arquivo[chave] = valor
                encontrado = True
                break

        if encontrado:
            continue

        outros[chave] = valor

    if not ia and not camera.get("Make") and not camera.get("Model"):
        indicios_ia.append({
            "campo": "Geral",
            "valor": "Ausente",
            "motivo": "Nenhum metadado de IA ou camera encontrado — metadados podem ter sido removidos",
        })

    return {
        "ia": ia,
        "indicios_ia": indicios_ia,
        "camera": camera,
        "arquivo": arquivo,
        "outros": outros,
    }


def ela(conteudo, sufixo):
    imagem_original = Image.open(io.BytesIO(conteudo)).convert("RGB")

    buffer_recomp = io.BytesIO()
    imagem_original.save(buffer_recomp, format="JPEG", quality=ELA_QUALIDADE)
    buffer_recomp.seek(0)
    imagem_recomprimida = Image.open(buffer_recomp).convert("RGB")

    diferenca = ImageChops.difference(imagem_original, imagem_recomprimida)
    arr = np.array(diferenca, dtype=np.float32)
    arr = np.clip(arr * ELA_FATOR_AMPLIFICACAO, 0, ELA_VALOR_MAX).astype(np.uint8)

    imagem_ela = Image.fromarray(arr)
    buffer_saida = io.BytesIO()
    imagem_ela.save(buffer_saida, format="PNG")

    return base64.b64encode(buffer_saida.getvalue()).decode()


def analise_forense(conteudo):
    try:
        imagem_np = np.frombuffer(conteudo, np.uint8)
        img_cv = cv2.imdecode(imagem_np, cv2.IMREAD_COLOR)
        if img_cv is None:
            return {}

        img_cinza = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
        canal_y, _, _ = cv2.split(img_yuv)

        sobel_x = cv2.Sobel(canal_y, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(canal_y, cv2.CV_64F, 0, 1, ksize=3)
        magnitude_grad = np.sqrt(sobel_x**2 + sobel_y**2)
        grad_normalizado = cv2.normalize(
            magnitude_grad, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        _, buffer_grad = cv2.imencode(".png", grad_normalizado)
        mapa_gradiente_b64 = base64.b64encode(buffer_grad).decode("utf-8")

        kernel_srm = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) / 4.0
        residuo_ruido = cv2.filter2D(img_cinza, -1, kernel_srm)

        tamanho_bloco = 16
        h, w = residuo_ruido.shape
        n_linhas = h // tamanho_bloco
        n_colunas = w // tamanho_bloco

        mapa_calor = np.zeros((n_linhas, n_colunas), dtype=np.float32)
        for idx_linha in range(n_linhas):
            for idx_coluna in range(n_colunas):
                r = idx_linha * tamanho_bloco
                c = idx_coluna * tamanho_bloco
                bloco = residuo_ruido[r:r + tamanho_bloco, c:c + tamanho_bloco]
                mapa_calor[idx_linha, idx_coluna] = float(np.var(bloco))

        mapa_calor_norm = cv2.normalize(
            mapa_calor, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        mapa_calor_cor = cv2.applyColorMap(mapa_calor_norm, cv2.COLORMAP_JET)
        mapa_calor_redim = cv2.resize(
            mapa_calor_cor, (w, h), interpolation=cv2.INTER_NEAREST
        )
        _, buffer_mapa_calor = cv2.imencode(".png", mapa_calor_redim)
        mapa_ruido_b64 = base64.b64encode(buffer_mapa_calor).decode("utf-8")

        fft_b64_visual, _ = _analisar_fft(img_cinza)

        img_cv_256 = cv2.resize(img_cv, (256, 256))
        conteudo_256 = cv2.imencode(".jpg", img_cv_256)[1].tobytes()
        img_cinza_256 = cv2.cvtColor(img_cv_256, cv2.COLOR_BGR2GRAY)

        imagem_original_256 = Image.open(io.BytesIO(conteudo_256)).convert("RGB")
        buffer_recomp = io.BytesIO()
        imagem_original_256.save(buffer_recomp, format="JPEG", quality=ELA_QUALIDADE)
        buffer_recomp.seek(0)
        imagem_recomprimida = Image.open(buffer_recomp).convert("RGB")
        diferenca = ImageChops.difference(imagem_original_256, imagem_recomprimida)
        arr_ela = np.array(diferenca, dtype=np.float32)
        arr_ela = np.clip(arr_ela * ELA_FATOR_AMPLIFICACAO, 0, ELA_VALOR_MAX)

        img_yuv_256 = cv2.cvtColor(img_cv_256, cv2.COLOR_BGR2YUV)
        canal_y_256, _, _ = cv2.split(img_yuv_256)
        sobel_x_256 = cv2.Sobel(canal_y_256, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y_256 = cv2.Sobel(canal_y_256, cv2.CV_64F, 0, 1, ksize=3)
        magnitude_grad_256 = np.sqrt(sobel_x_256**2 + sobel_y_256**2)

        kernel_srm_256 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]) / 4.0
        residuo_256 = cv2.filter2D(img_cinza_256, -1, kernel_srm_256)
        h256, w256 = residuo_256.shape
        n_linhas_256 = h256 // tamanho_bloco
        n_colunas_256 = w256 // tamanho_bloco
        variancias_256 = []
        for idx_linha in range(n_linhas_256):
            for idx_coluna in range(n_colunas_256):
                ri = idx_linha * tamanho_bloco
                ci = idx_coluna * tamanho_bloco
                bloco = residuo_256[ri:ri + tamanho_bloco, ci:ci + tamanho_bloco]
                variancias_256.append(float(np.var(bloco)))
        variancia_256 = float(np.mean(variancias_256)) if variancias_256 else 0.0

        _, fft_simetria_256 = _analisar_fft(img_cinza_256)
        corr_rgb_256 = _correlacao_rgb(img_cv_256)
        aber_256 = _aberracao_cromatica(img_cv_256)

        ela_media = round(float(np.mean(arr_ela)), 4)
        ela_desvio = round(float(np.std(arr_ela)), 4)
        variancia_ruido = round(variancia_256, 4)
        fft_sim = round(fft_simetria_256, 4)
        aber = round(aber_256, 4)
        grad_media = round(float(np.mean(magnitude_grad_256)), 4)
        grad_desvio = round(float(np.std(magnitude_grad_256)), 4)

        interpretacao = _interpretar_resultados(
            ela_media, ela_desvio, variancia_ruido,
            fft_sim, corr_rgb_256, aber, grad_media, grad_desvio,
        )

        return {
            "mapa_gradiente_b64": mapa_gradiente_b64,
            "mapa_ruido_b64": mapa_ruido_b64,
            "variancia_ruido": variancia_ruido,
            "fft_mapa_b64": fft_b64_visual,
            "fft_simetria": fft_sim,
            "correlacao_rgb": corr_rgb_256,
            "aberracao_cromatica": aber,
            "gradiente_media": grad_media,
            "gradiente_desvio": grad_desvio,
            "ela_media": ela_media,
            "ela_desvio": ela_desvio,
            "interpretacao": interpretacao,
        }
    except Exception as exc:
        logger.error("Falha na analise forense: %s", exc)
        return {}



def _interpretar_resultados(ela_media, ela_desvio, variancia_ruido,
                            fft_simetria, corr_rgb, aberracao, grad_media, grad_desvio):
    conclusoes = []

    if ela_media < 15:
        conclusoes.append({
            "area": "ELA",
            "indicador": "Baixa variacao ELA",
            "valor": ela_media,
            "interpretacao": "A imagem apresenta niveis de compressao muito uniformes. Isso pode indicar que a imagem foi gerada sinteticamente ou recomprimida multiplas vezes.",
            "severidade": "alta",
        })
    elif ela_media < 30:
        conclusoes.append({
            "area": "ELA",
            "indicador": "Variacao ELA moderada",
            "valor": ela_media,
            "interpretacao": "A imagem apresenta variacao moderada nos niveis de compressao. Consistente com imagens naturais que passaram por algum processamento.",
            "severidade": "media",
        })
    else:
        conclusoes.append({
            "area": "ELA",
            "indicador": "Variacao ELA elevada",
            "valor": ela_media,
            "interpretacao": "A imagem apresenta variacao significativa nos niveis de compressao. Pode indicar regioes editadas ou colagem de diferentes fontes.",
            "severidade": "alta",
        })

    if ela_desvio < 10:
        conclusoes.append({
            "area": "ELA",
            "indicador": "Desvio ELA baixo",
            "valor": ela_desvio,
            "interpretacao": "A distribuicao do erro de compressao e muito homogenea. Comum em imagens geradas por IA, que nao passam por compressao JPEG real.",
            "severidade": "alta",
        })

    if variancia_ruido < 3:
        conclusoes.append({
            "area": "Ruido",
            "indicador": "Ruido muito uniforme",
            "valor": variancia_ruido,
            "interpretacao": "O ruido da imagem e extremamente uniforme. Imagens de cameras reais possuem ruido com variacao natural. Forte indicacao de geracao artificial.",
            "severidade": "alta",
        })
    elif variancia_ruido < 7:
        conclusoes.append({
            "area": "Ruido",
            "indicador": "Ruido pouco variado",
            "valor": variancia_ruido,
            "interpretacao": "O ruido apresenta pouca variacao entre regioes. Pode indicar suavizacao excessiva ou geracao artificial.",
            "severidade": "media",
        })
    elif variancia_ruido > 80:
        conclusoes.append({
            "area": "Ruido",
            "indicador": "Ruido excessivo",
            "valor": variancia_ruido,
            "interpretacao": "O ruido apresenta variacao muito alta entre regioes. Pode indicar colagem de diferentes fontes com caracteristicas de ruido distintas.",
            "severidade": "media",
        })
    else:
        conclusoes.append({
            "area": "Ruido",
            "indicador": "Ruido dentro do esperado",
            "valor": variancia_ruido,
            "interpretacao": "A distribuicao de ruido e consistente com uma imagem capturada por camera real.",
            "severidade": "baixa",
        })

    if fft_simetria > 0.98:
        conclusoes.append({
            "area": "FFT",
            "indicador": "Simetria espectral muito alta",
            "valor": fft_simetria,
            "interpretacao": "O espectro de frequencias e quase perfeitamente simetrico. Isso e raro em fotografias naturais e comum em imagens geradas por redes neurais.",
            "severidade": "alta",
        })
    elif fft_simetria > 0.95:
        conclusoes.append({
            "area": "FFT",
            "indicador": "Simetria espectral elevada",
            "valor": fft_simetria,
            "interpretacao": "O espectro de frequencias apresenta simetria acima do comum. Merece atencao.",
            "severidade": "media",
        })
    else:
        conclusoes.append({
            "area": "FFT",
            "indicador": "Simetria espectral normal",
            "valor": fft_simetria,
            "interpretacao": "O espectro de frequencias apresenta assimetria natural, consistente com fotografias reais.",
            "severidade": "baixa",
        })

    media_corr = 0.0
    if corr_rgb:
        rg = abs(corr_rgb.get("rg", 0))
        rb = abs(corr_rgb.get("rb", 0))
        gb = abs(corr_rgb.get("gb", 0))
        media_corr = (rg + rb + gb) / 3.0

        if media_corr > 0.98:
            conclusoes.append({
                "area": "Correlacao RGB",
                "indicador": "Correlacao entre canais extremamente alta",
                "valor": round(media_corr, 4),
                "interpretacao": "Os canais de cor estao quase identicos. Isso pode indicar uma imagem quase monocromatica ou geracao artificial com pouca variacao de cor.",
                "severidade": "alta",
            })
        elif media_corr > 0.92:
            conclusoes.append({
                "area": "Correlacao RGB",
                "indicador": "Correlacao entre canais elevada",
                "valor": round(media_corr, 4),
                "interpretacao": "Os canais de cor apresentam correlacao acima do comum. Pode indicar processamento artificial.",
                "severidade": "media",
            })
        else:
            conclusoes.append({
                "area": "Correlacao RGB",
                "indicador": "Correlacao entre canais normal",
                "valor": round(media_corr, 4),
                "interpretacao": "A relacao entre os canais de cor e consistente com uma fotografia natural.",
                "severidade": "baixa",
            })

    if aberracao < 0.03:
        conclusoes.append({
            "area": "Aberracao Cromatica",
            "indicador": "Aberracao cromatica ausente",
            "valor": aberracao,
            "interpretacao": "A imagem nao apresenta aberracao cromatica. Cameras reais produzem algum grau de aberracao devido as propriedades opticas das lentes. A ausencia total e comum em imagens geradas por IA.",
            "severidade": "alta",
        })
    elif aberracao < 0.08:
        conclusoes.append({
            "area": "Aberracao Cromatica",
            "indicador": "Aberracao cromatica baixa",
            "valor": aberracao,
            "interpretacao": "A imagem apresenta pouca aberracao cromatica. Pode ser uma lente de alta qualidade ou indicar geracao artificial.",
            "severidade": "media",
        })
    elif aberracao > 0.20:
        conclusoes.append({
            "area": "Aberracao Cromatica",
            "indicador": "Aberracao cromatica excessiva",
            "valor": aberracao,
            "interpretacao": "A imagem apresenta aberracao cromatica acima do normal. Pode indicar manipulacao nas bordas ou uso de lentes de baixa qualidade.",
            "severidade": "media",
        })
    else:
        conclusoes.append({
            "area": "Aberracao Cromatica",
            "indicador": "Aberracao cromatica dentro do esperado",
            "valor": aberracao,
            "interpretacao": "O nivel de aberracao cromatica e consistente com uma fotografia capturada por camera real.",
            "severidade": "baixa",
        })

    if grad_media < 5:
        conclusoes.append({
            "area": "Gradiente",
            "indicador": "Transicoes muito suaves",
            "valor": grad_media,
            "interpretacao": "A imagem apresenta transicoes extremamente suaves entre regioes. Pode indicar suavizacao artificial ou geracao por IA.",
            "severidade": "media",
        })
    elif grad_desvio < 8:
        conclusoes.append({
            "area": "Gradiente",
            "indicador": "Variacao de bordas baixa",
            "valor": grad_desvio,
            "interpretacao": "A variacao nas transicoes e baixa, indicando uniformidade incomum. Fotografias naturais tendem a ter maior variacao.",
            "severidade": "media",
        })
    else:
        conclusoes.append({
            "area": "Gradiente",
            "indicador": "Transicoes dentro do esperado",
            "valor": grad_media,
            "interpretacao": "O padrao de transicoes e bordas e consistente com uma fotografia natural.",
            "severidade": "baixa",
        })

    total = len(conclusoes)
    altas = sum(1 for c in conclusoes if c["severidade"] == "alta")
    medias = sum(1 for c in conclusoes if c["severidade"] == "media")

    if altas >= 3:
        resumo = "Multiplos indicadores apontam forte possibilidade de manipulacao ou geracao artificial."
    elif altas >= 1 and medias >= 2:
        resumo = "Alguns indicadores sugerem possivel manipulacao. Recomenda-se analise complementar."
    elif altas >= 1:
        resumo = "Ao menos um indicador apresenta anomalia significativa. Verificacao adicional recomendada."
    elif medias >= 2:
        resumo = "Pequenas inconsistencias detectadas. A imagem pode ter sofrido processamento leve."
    else:
        resumo = "Os indicadores analisados sao consistentes com uma imagem autentica."

    return {
        "conclusoes": conclusoes,
        "resumo": resumo,
        "contagem": {
            "alta": altas,
            "media": medias,
            "baixa": total - altas - medias,
        },
    }


def _analisar_fft(img_cinza):
    f = np.fft.fft2(img_cinza.astype(np.float32))
    f_deslocado = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(f_deslocado))

    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    magnitude_cor = cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_INFERNO)
    _, buffer_fft = cv2.imencode(".png", magnitude_cor)
    fft_b64 = base64.b64encode(buffer_fft).decode()

    h, w = magnitude.shape
    metade_superior = magnitude[:h // 2, :]
    metade_inferior = np.flipud(magnitude[h // 2:, :])
    min_h = min(metade_superior.shape[0], metade_inferior.shape[0])
    diff = np.abs(metade_superior[:min_h] - metade_inferior[:min_h])
    simetria = float(1.0 - (np.mean(diff) / (np.max(magnitude) + 1e-9)))

    return fft_b64, simetria


def _correlacao_rgb(img_cv):
    b, g, r = cv2.split(img_cv.astype(np.float32))

    def corr(a, b_):
        a_plano = a.flatten()
        b_plano = b_.flatten()
        if np.std(a_plano) < 1e-9 or np.std(b_plano) < 1e-9:
            return 1.0
        return float(np.corrcoef(a_plano, b_plano)[0, 1])

    return {
        "rg": round(corr(r, g), 4),
        "rb": round(corr(r, b), 4),
        "gb": round(corr(g, b), 4),
    }


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


def busca_reversa(conteudo):
    chave = _obter_chave_serpapi_valida()

    try:
        resposta = requests.post(
            SERPAPI_URL,
            data={
                "engine": SERPAPI_ENGINE,
                "api_key": chave,
            },
            files={"image": ("imagem.jpg", conteudo, "image/jpeg")},
            timeout=30,
        )
        resposta.raise_for_status()
        dados = resposta.json()

        if "error" in dados:
            raise Exception(dados["error"])

        resultados_brutos = dados.get("image_results", [])[:SERPAPI_MAX_RESULTADOS]
        return [
            {
                "titulo": item.get("title"),
                "url": item.get("link"),
                "fonte": item.get("source"),
            }
            for item in resultados_brutos
        ]
    except Exception as exc:
        logger.error("Falha na busca reversa: %s", exc)
        raise