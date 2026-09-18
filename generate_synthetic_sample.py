import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from config import (
    CVAE_CONFIG,
    DATA_PATH,
    DATASET_NAME,
    DEVICE,
    IMAGES_PATH,
    MODELS_PATH,
    SYNTHETIC_SAMPLE_SIZE,
)
from models import CVAE
from utils import load_dataset


def load_model(dataset_name: str, config: dict[str, Any]) -> torch.nn.Module:
    files = sorted(
        [file for file in os.listdir(MODELS_PATH) if dataset_name in file],
        reverse=True,
    )

    model = CVAE(**config).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODELS_PATH}/{files[0]}", weights_only=True))
    model.eval()

    return model


def generate_synthetic_sample(
    model: torch.nn.Module,
    sample_size: int,
    latent_dim: int,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = sample_size // num_classes
    samples = []
    sample_labels = []

    model.eval()

    with torch.inference_mode():
        for i in range(num_classes):
            z = torch.randn(size, latent_dim, device=DEVICE)
            labels = torch.full((size,), i, dtype=torch.long, device=DEVICE)
            y = F.one_hot(labels, num_classes=num_classes).float()

            X_synth = model.decode(z, y)
            samples.append(X_synth)
            sample_labels.append(labels)

    return torch.cat(samples), torch.cat(sample_labels)


def plot_synthetic_sample(sample: np.ndarray, labels: np.ndarray, img_name: str):
    pca = PCA(n_components=2)
    sample_2d = pca.fit_transform(sample)

    sns.scatterplot(
        x=sample_2d[:, 0],
        y=sample_2d[:, 1],
        hue=labels,
        palette="tab10",
    )

    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title(f"{DATASET_NAME} synthetic sample")
    plt.grid()

    os.makedirs(IMAGES_PATH, exist_ok=True)
    plt.savefig(os.path.join(IMAGES_PATH, img_name))
    plt.close()


def main():
    X, y = load_dataset(DATASET_NAME)
    num_classes = len(set(y))

    CVAE_CONFIG["input_dim"] = X.shape[1]
    CVAE_CONFIG["num_classes"] = num_classes

    model = load_model(DATASET_NAME, CVAE_CONFIG)

    synthetic_sample, labels = generate_synthetic_sample(
        model,
        SYNTHETIC_SAMPLE_SIZE,
        CVAE_CONFIG.get("latent_dim", -1),
        num_classes,
    )

    os.makedirs(DATA_PATH, exist_ok=True)
    torch.save(synthetic_sample, f"{DATA_PATH}/{DATASET_NAME}_synthetic.pt")

    plot_synthetic_sample(
        synthetic_sample.numpy(force=True),
        labels.numpy(force=True),
        f"{DATASET_NAME}_synthetic_sample.png",
    )


if __name__ == "__main__":
    main()
