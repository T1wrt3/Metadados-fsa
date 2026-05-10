import pandas as pd
import numpy as np
from pathlib import Path

PROJETO = Path(__file__).parent.resolve()

csv_treino_original = PROJETO / "archive" / "features_train.csv"
df = pd.read_csv(csv_treino_original)

reais = df[df["rotulo"] == 0]
fakes = df[df["rotulo"] == 1]

print("Os dados ja estao normalizados?")
print(f"  ela_media - media: {df['ela_media'].mean():.4f}, std: {df['ela_media'].std():.4f}")
print(f"  Se media~0 e std~1, ja foram normalizados.\n")

print("Preciso re-extrair features brutas de algumas imagens do dataset.")
print("Pegando 3 reais e 3 fakes do dataset...\n")

from extrair_features import extrair_de_imagem

pasta_real = PROJETO / "archive" / "real_vs_fake" / "train" / "real"
pasta_fake = PROJETO / "archive" / "real_vs_fake" / "train" / "fake"

print("=== REAIS DO DATASET (valores brutos) ===")
count = 0
for img in pasta_real.iterdir():
    if count >= 3:
        break
    feat = extrair_de_imagem(img)
    if feat:
        print(f"  {img.name}")
        print(f"    ela={feat[0]:.2f}  ruido={feat[2]:.2f}  grad={feat[8]:.2f}  aber={feat[7]:.4f}")
        count += 1

print("\n=== FAKES DO DATASET (valores brutos) ===")
count = 0
for img in pasta_fake.iterdir():
    if count >= 3:
        break
    feat = extrair_de_imagem(img)
    if feat:
        print(f"  {img.name}")
        print(f"    ela={feat[0]:.2f}  ruido={feat[2]:.2f}  grad={feat[8]:.2f}  aber={feat[7]:.4f}")
        count += 1

print("\n=== SUAS IMAGENS (valores brutos) ===")
print("  REAL_1:  ela=49.45  ruido=325.69  grad=99.05  aber=0.41")
print("  IA_GPT:  ela=40.24  ruido=104.71  grad=79.30  aber=0.69")
print("  REAL_2:  ela=26.59  ruido=7.39    grad=63.77  aber=0.53")