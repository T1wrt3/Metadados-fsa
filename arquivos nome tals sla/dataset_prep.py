from pathlib import Path
import pandas as pd
import shutil
import random

random.seed(42)

PROJETO = Path(__file__).parent.resolve()
DATASET = PROJETO / "dataset"
DESTINO = PROJETO / "archive" / "real_vs_fake"

FONTES = [
    {
        "csv": DATASET / "val_labels.csv",
        "pasta": DATASET / "val_images" / "val_images",
    },
    {
        "csv": DATASET / "val_hard_labels.csv",
        "pasta": DATASET / "val_images_hard" / "val_images_hard",
    },
]

SPLIT = (0.70, 0.15, 0.15)


def main():
    todas_reais = []
    todas_fakes = []

    for fonte in FONTES:
        csv_path = fonte["csv"]
        pasta = fonte["pasta"]

        if not csv_path.exists():
            print(f"CSV nao encontrado: {csv_path}")
            return
        if not pasta.exists():
            print(f"Pasta nao encontrada: {pasta}")
            return

        df = pd.read_csv(csv_path)

        for _, linha in df.iterrows():
            nome = linha["image_name"]
            label = int(linha["label"])
            caminho = pasta / nome

            if not caminho.exists():
                continue

            if label == 0:
                todas_reais.append(caminho)
            else:
                todas_fakes.append(caminho)

    print(f"Total real: {len(todas_reais)}")
    print(f"Total fake: {len(todas_fakes)}")
    print(f"Total geral: {len(todas_reais) + len(todas_fakes)}")

    random.shuffle(todas_reais)
    random.shuffle(todas_fakes)

    if DESTINO.exists():
        shutil.rmtree(DESTINO)

    def dividir(lista):
        n = len(lista)
        n_train = int(n * SPLIT[0])
        n_valid = int(n * SPLIT[1])
        return {
            "train": lista[:n_train],
            "valid": lista[n_train:n_train + n_valid],
            "test": lista[n_train + n_valid:],
        }

    splits_real = dividir(todas_reais)
    splits_fake = dividir(todas_fakes)

    for split_name in ["train", "valid", "test"]:
        for classe, lista in [("real", splits_real[split_name]), ("fake", splits_fake[split_name])]:
            pasta_dest = DESTINO / split_name / classe
            pasta_dest.mkdir(parents=True, exist_ok=True)

            copiados = 0
            for caminho in lista:
                shutil.copy2(caminho, pasta_dest / caminho.name)
                copiados += 1

            print(f"  {split_name}/{classe}: {copiados} imagens")

    print("\nDataset preparado com sucesso.")


if __name__ == "__main__":
    main()