import os
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    CVAE_CONFIG,
    DATA_PATH,
    DATASET_NAME,
    DEVICE,
    IMAGES_PATH,
    MODELS_PATH,
    NUM_EPOCHS,
    RANDOM_STATE,
)
from models import CVAE
from utils import load_dataset


def plot_losses(
    losses: list[float],
    recon_losses: list[float],
    kl_losses: list[float],
    img_name: str,
):
    epochs = range(1, len(losses) + 1)

    sns.lineplot(x=epochs, y=losses, label="Loss")
    sns.lineplot(x=epochs, y=recon_losses, label="Reconstruction")
    sns.lineplot(x=epochs, y=kl_losses, label="KL")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CVAE Losses")
    plt.grid()

    os.makedirs(IMAGES_PATH, exist_ok=True)
    plt.savefig(os.path.join(IMAGES_PATH, img_name))
    plt.close()


def plot_sample(sample: np.ndarray, labels: np.ndarray, img_name: str):
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
    plt.title(f"{DATASET_NAME} sample")
    plt.grid()

    os.makedirs(IMAGES_PATH, exist_ok=True)
    plt.savefig(os.path.join(IMAGES_PATH, img_name))
    plt.close()


def train_one_epoch(
    epoch_index: int,
    train_loader: torch.utils.data.DataLoader,
    device: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
):
    total_loss = 0.0
    total_kl = 0.0
    total_recon = 0.0

    model.train()

    for x, y in train_loader:
        x = x.to(device)

        y = (
            torch.nn.functional.one_hot(
                y,
                num_classes=num_classes,
            )
            .float()
            .to(device)
        )

        x_hat, mu, logvar = model(x, y)

        reconstruction_loss = nn.functional.mse_loss(
            x_hat,
            x,
        )

        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = reconstruction_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_kl += kl_loss.item()
        total_recon += reconstruction_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_kl = total_kl / len(train_loader)
    avg_recon = total_recon / len(train_loader)

    if epoch_index % 10 == 0:
        print(
            f"Epoch {epoch_index + 1}: "
            f"loss={avg_loss:.4f}, "
            f"recon={avg_recon:.4f}, "
            f"kl={avg_kl:.4f}",
        )

    return avg_loss, avg_recon, avg_kl


def main():
    X, y = load_dataset(DATASET_NAME)
    num_classes = len(set(y))

    CVAE_CONFIG["input_dim"] = X.shape[1]
    CVAE_CONFIG["num_classes"] = num_classes

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    os.makedirs(DATA_PATH, exist_ok=True)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    torch.save(train_dataset, f"{DATA_PATH}/{DATASET_NAME}_train.pt")
    torch.save(test_dataset, f"{DATA_PATH}/{DATASET_NAME}_test.pt")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print(f"Using: {DEVICE}")

    model = CVAE(**CVAE_CONFIG).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []

    recon_losses = []
    kl_losses = []

    for epoch in range(NUM_EPOCHS):
        loss, recon, kl = train_one_epoch(
            epoch,
            train_loader,
            DEVICE,
            model,
            optimizer,
            num_classes=num_classes,
        )

        train_losses.append(loss)
        recon_losses.append(recon)
        kl_losses.append(kl)

    plot_losses(train_losses, recon_losses, kl_losses, "cvae_losses.png")

    plot_sample(
        X_train,
        y_train,
        f"{DATASET_NAME}_sample.png",
    )

    os.makedirs(MODELS_PATH, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    model_name = f"cvae_{DATASET_NAME}_{timestamp}.pth"
    torch.save(model.state_dict(), os.path.join(MODELS_PATH, model_name))


if __name__ == "__main__":
    main()
