import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJETO = Path(__file__).parent.resolve()
BASE_DATASET = PROJETO / "archive" / "real_vs_fake"
BASE_FEATURES = PROJETO / "archive"
CAMINHO_MODELO = PROJETO / "modelo_salvo.pth"
CAMINHO_NORMALIZADOR = PROJETO / "normalizador.joblib"

COLUNAS_FEATURES = [
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

NUM_FEATURES_FORENSES = len(COLUNAS_FEATURES)
TAMANHO_IMAGEM = 224
BATCH_SIZE = 32
NUM_EPOCAS = 30
TAXA_APRENDIZADO = 0.00003

TRANSFORMACAO_TREINO = transforms.Compose([
    transforms.Resize((TAMANHO_IMAGEM, TAMANHO_IMAGEM)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRANSFORMACAO_AVALIACAO = transforms.Compose([
    transforms.Resize((TAMANHO_IMAGEM, TAMANHO_IMAGEM)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DatasetHibrido(Dataset):
    def __init__(self, conjunto, transformacao, normalizador=None):
        self.transformacao = transformacao
        self.normalizador = normalizador

        csv_path = BASE_FEATURES / f"features_{conjunto}.csv"
        self.df = pd.read_csv(csv_path)

        self.pasta_real = BASE_DATASET / conjunto / "real"
        self.pasta_fake = BASE_DATASET / conjunto / "fake"

        self.mapa_features = {}
        for _, linha in self.df.iterrows():
            nome = linha["arquivo"]
            valores = [float(linha[col]) for col in COLUNAS_FEATURES]
            self.mapa_features[nome] = np.array(valores, dtype=np.float32)

        self.amostras = []
        for _, linha in self.df.iterrows():
            nome = linha["arquivo"]
            rotulo = int(linha["rotulo"])
            if rotulo == 0:
                caminho = self.pasta_real / nome
            else:
                caminho = self.pasta_fake / nome
            if caminho.exists() and nome in self.mapa_features:
                self.amostras.append((caminho, rotulo, nome))

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, idx):
        caminho, rotulo, nome = self.amostras[idx]

        imagem = Image.open(caminho).convert("RGB")
        imagem = self.transformacao(imagem)

        features = self.mapa_features[nome].copy()

        if self.normalizador is not None:
            features = self.normalizador.transform(features.reshape(1, -1)).flatten()

        return imagem, torch.tensor(features, dtype=torch.float32), torch.tensor(rotulo, dtype=torch.float32)


class ModeloHibrido(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_features_cnn = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()

        self.camada_forense = nn.Sequential(
            nn.Linear(NUM_FEATURES_FORENSES, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        tamanho_fusao = num_features_cnn + 32

        self.classificador = nn.Sequential(
            nn.Linear(tamanho_fusao, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, imagem, features_forenses):
        saida_cnn = self.cnn(imagem)
        saida_forense = self.camada_forense(features_forenses)
        fusao = torch.cat([saida_cnn, saida_forense], dim=1)
        return self.classificador(fusao).squeeze(1)


def criar_normalizador():
    from sklearn.preprocessing import StandardScaler

    csv_treino = BASE_FEATURES / "features_train.csv"
    df = pd.read_csv(csv_treino)

    normalizador = StandardScaler()
    normalizador.fit(df[COLUNAS_FEATURES].values)
    joblib.dump(normalizador, CAMINHO_NORMALIZADOR)

    print(f"Normalizador criado com {len(df)} amostras de treino")
    print(f"Medias: {normalizador.mean_}")
    print(f"Escalas: {normalizador.scale_}")

    return normalizador


def treinar():
    print(f"Dispositivo: {DISPOSITIVO}")

    normalizador = criar_normalizador()

    dataset_treino = DatasetHibrido("train", TRANSFORMACAO_TREINO, normalizador)
    dataset_valid = DatasetHibrido("valid", TRANSFORMACAO_AVALIACAO, normalizador)

    print(f"Treino: {len(dataset_treino)} amostras")
    print(f"Valid: {len(dataset_valid)} amostras")

    loader_treino = DataLoader(
        dataset_treino,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    loader_valid = DataLoader(
        dataset_valid,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    modelo = ModeloHibrido().to(DISPOSITIVO)
    criterio = nn.BCEWithLogitsLoss()
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=TAXA_APRENDIZADO, weight_decay=1e-3)
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=NUM_EPOCAS)

    melhor_acuracia = 0.0
    paciencia = 7
    sem_melhora = 0

    for epoca in range(NUM_EPOCAS):
        modelo.train()
        perda_total = 0.0
        acertos = 0
        total = 0

        barra = tqdm(loader_treino, desc=f"Epoca {epoca + 1}/{NUM_EPOCAS}")
        for imagens, features, rotulos in barra:
            imagens = imagens.to(DISPOSITIVO)
            features = features.to(DISPOSITIVO)
            rotulos = rotulos.to(DISPOSITIVO)

            otimizador.zero_grad()
            saidas = modelo(imagens, features)
            perda = criterio(saidas, rotulos)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()

            perda_total += perda.item()
            predicoes = (torch.sigmoid(saidas) > 0.5).float()
            acertos += (predicoes == rotulos).sum().item()
            total += rotulos.size(0)

            barra.set_postfix(perda=f"{perda.item():.4f}", acc=f"{acertos/total:.4f}")

        agendador.step()

        modelo.eval()
        acertos_valid = 0
        total_valid = 0

        with torch.no_grad():
            for imagens, features, rotulos in loader_valid:
                imagens = imagens.to(DISPOSITIVO)
                features = features.to(DISPOSITIVO)
                rotulos = rotulos.to(DISPOSITIVO)

                saidas = modelo(imagens, features)
                predicoes = (torch.sigmoid(saidas) > 0.5).float()
                acertos_valid += (predicoes == rotulos).sum().item()
                total_valid += rotulos.size(0)

        acc_valid = acertos_valid / total_valid if total_valid > 0 else 0
        acc_treino = acertos / total if total > 0 else 0
        print(f"Epoca {epoca + 1} | Treino: {acc_treino:.4f} | Valid: {acc_valid:.4f}")

        if acc_valid > melhor_acuracia:
            melhor_acuracia = acc_valid
            torch.save(modelo.state_dict(), CAMINHO_MODELO)
            print(f"Modelo salvo com acuracia: {melhor_acuracia:.4f}")
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                print(f"Early stopping na epoca {epoca + 1}")
                break

    print(f"Treinamento concluido. Melhor acuracia: {melhor_acuracia:.4f}")


def testar():
    normalizador = joblib.load(CAMINHO_NORMALIZADOR)

    dataset_teste = DatasetHibrido("test", TRANSFORMACAO_AVALIACAO, normalizador)
    loader_teste = DataLoader(
        dataset_teste,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    modelo = ModeloHibrido().to(DISPOSITIVO)
    modelo.load_state_dict(torch.load(CAMINHO_MODELO, map_location=DISPOSITIVO))
    modelo.eval()

    acertos = 0
    total = 0
    verdadeiros_positivos = 0
    falsos_positivos = 0
    falsos_negativos = 0

    with torch.no_grad():
        for imagens, features, rotulos in tqdm(loader_teste, desc="Testando"):
            imagens = imagens.to(DISPOSITIVO)
            features = features.to(DISPOSITIVO)
            rotulos = rotulos.to(DISPOSITIVO)

            saidas = modelo(imagens, features)
            predicoes = (torch.sigmoid(saidas) > 0.5).float()

            acertos += (predicoes == rotulos).sum().item()
            total += rotulos.size(0)

            verdadeiros_positivos += ((predicoes == 1) & (rotulos == 1)).sum().item()
            falsos_positivos += ((predicoes == 1) & (rotulos == 0)).sum().item()
            falsos_negativos += ((predicoes == 0) & (rotulos == 1)).sum().item()

    acuracia = acertos / total if total > 0 else 0
    precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos + 1e-9)
    recall = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos + 1e-9)
    f1 = 2 * (precisao * recall) / (precisao + recall + 1e-9)

    print(f"Acuracia: {acuracia:.4f}")
    print(f"Precisao: {precisao:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


def classificar(conteudo_bytes, features_forenses):
    normalizador = joblib.load(CAMINHO_NORMALIZADOR)

    features_array = np.array([features_forenses], dtype=np.float32)
    features_norm = normalizador.transform(features_array)
    features_tensor = torch.tensor(features_norm, dtype=torch.float32).to(DISPOSITIVO)

    imagem = Image.open(__import__("io").BytesIO(conteudo_bytes)).convert("RGB")
    imagem_tensor = TRANSFORMACAO_AVALIACAO(imagem).unsqueeze(0).to(DISPOSITIVO)

    modelo = ModeloHibrido().to(DISPOSITIVO)
    modelo.load_state_dict(torch.load(CAMINHO_MODELO, map_location=DISPOSITIVO))
    modelo.eval()

    with torch.no_grad():
        saida = modelo(imagem_tensor, features_tensor)
        probabilidade = torch.sigmoid(saida).item()

    score = round(probabilidade * 100, 2)

    if score >= 70:
        nivel = "Alto"
    elif score >= 40:
        nivel = "Medio"
    else:
        nivel = "Baixo"

    return {
        "score": score,
        "nivel": nivel,
        "probabilidade": round(probabilidade, 4),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python modelozudo.py [treinar|testar]")
        sys.exit(1)

    comando = sys.argv[1]

    if comando == "treinar":
        treinar()
    elif comando == "testar":
        testar()