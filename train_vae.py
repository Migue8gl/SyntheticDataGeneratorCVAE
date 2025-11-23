import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from utils import set_all_seeds


def load_dataset(name):
    if name == "iris":
        d = load_iris()
    elif name == "wine":
        d = load_wine()
    elif name == "breast_cancer":
        d = load_breast_cancer()
    else:
        raise ValueError(name)
    return d.data, d.target, len(set(d.target)), d.feature_names


class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super().__init__()
        conditional_dim = input_dim + num_classes
        self.enc1 = nn.Linear(conditional_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec1 = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, y):
        h = self.relu(self.enc1(torch.cat([x, y], 1)))
        h = self.relu(self.enc2(h))
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, y):
        h = self.relu(self.dec1(torch.cat([z, y], 1)))
        h = self.relu(self.dec2(h))
        return self.sigmoid(self.dec_out(h))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparam(mu, logvar)
        return self.decode(z, y), mu, logvar


def loss_fn(recon, x, mu, logvar, beta):
    mse = nn.functional.mse_loss(recon, x, reduction="sum")
    kld = -0.5 * torch.mean(1 + logvar - mu**2 - torch.exp(logvar))
    return mse + beta * kld, mse, kld


def evaluate(model, dataloader, device, beta):
    model.eval()
    tot_l = tot_m = tot_k = 0
    with torch.inference_mode():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            recon, mu, logvar = model(xb, yb)
            loss, mse, kl = loss_fn(recon, xb, mu, logvar, beta)
            tot_l += loss.item() * xb.size(0)
            tot_m += mse.item() * xb.size(0)
            tot_k += kl.item() * xb.size(0)
    n = len(dataloader.dataset)
    return tot_l / n, tot_m / n, tot_k / n


def save_learning_curves_plot(losses, eval_losses, epochs, path):
    df = pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": losses["loss"],
            "val_loss": eval_losses["loss"],
            "train_mse": losses["mse"],
            "val_mse": eval_losses["mse"],
            "train_kld": losses["kld"],
            "val_kld": eval_losses["kld"],
        },
    )
    df_long = df.melt(id_vars="epoch", var_name="metric", value_name="value")
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    for i, (name, title) in enumerate(
        [("loss", "Loss"), ("mse", "Binary Cross Entropy"), ("kld", "KL Divergence")],
        1,
    ):
        plt.subplot(3, 1, i)
        subset = df_long[df_long["metric"].isin([f"train_{name}", f"val_{name}"])]
        sns.lineplot(data=subset, x="epoch", y="value", hue="metric")
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    cfg = OmegaConf.load("config/config.yaml")
    set_all_seeds(cfg.experiment.seed)
    device = torch.device(cfg.experiment.device)

    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    os.makedirs(cfg.paths.image_dir, exist_ok=True)

    X, y, num_classes, feature_names = load_dataset(cfg.dataset.name)
    input_dim = X.shape[1]
    hidden_dim = int(input_dim * cfg.model.hidden_dim_factor)
    latent_dim = int(input_dim * cfg.model.latent_dim_factor)

    print(f"Training CVAE on {cfg.dataset.name} dataset")
    print(f"Input dim: {input_dim}, Classes: {num_classes}")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=cfg.dataset.test_size,
        stratify=y,
        random_state=cfg.experiment.seed,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=cfg.dataset.val_size,
        stratify=y_train_val,
        random_state=cfg.experiment.seed,
    )

    print(
        f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}",
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open(
        os.path.join(
            cfg.paths.model_dir,
            cfg.paths.scaler_file.format(dataset=cfg.dataset.name),
        ),
        "wb",
    ) as f:
        pickle.dump(scaler, f)

    np.save(
        os.path.join(cfg.paths.data_dir, f"X_test_{cfg.dataset.name}.npy"),
        X_test_scaled,
    )
    np.save(os.path.join(cfg.paths.data_dir, f"y_test_{cfg.dataset.name}.npy"), y_test)
    np.save(
        os.path.join(cfg.paths.data_dir, f"X_train_{cfg.dataset.name}.npy"),
        X_train_scaled,
    )
    np.save(
        os.path.join(cfg.paths.data_dir, f"y_train_{cfg.dataset.name}.npy"),
        y_train,
    )

    with open(
        os.path.join(
            cfg.paths.model_dir,
            cfg.paths.config_file.format(dataset=cfg.dataset.name),
        ),
        "wb",
    ) as f:
        pickle.dump(
            {
                "input_dim": input_dim,
                "num_classes": num_classes,
                "feature_names": feature_names,
            },
            f,
        )

    y_train_oh = torch.eye(num_classes)[y_train].float()
    y_val_oh = torch.eye(num_classes)[y_val].float()
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)

    train_dl = DataLoader(
        TensorDataset(X_train_t, y_train_oh),
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    val_dl = DataLoader(
        TensorDataset(X_val_t, y_val_oh),
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    model = CVAE(input_dim, hidden_dim, latent_dim, num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode=cfg.lr_scheduler.mode,
        factor=cfg.lr_scheduler.factor,
        patience=cfg.lr_scheduler.patience,
    )

    best = float("inf")
    patience = 0

    losses: Dict[str, List[float]] = {"loss": [], "mse": [], "kld": []}
    eval_losses: Dict[str, List[float]] = {"loss": [], "mse": [], "kld": []}
    epochs = []

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        tl = tm = tk = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb, yb)
            loss, mse, kld = loss_fn(recon, xb, mu, logvar, cfg.loss.beta)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.gradient_clip_norm,
            )
            opt.step()
            tl += loss.item() * xb.size(0)
            tm += mse.item() * xb.size(0)
            tk += kld.item() * xb.size(0)

        n = len(train_dl.dataset)
        losses["loss"].append(tl / n)
        losses["mse"].append(tm / n)
        losses["kld"].append(tk / n)

        vl, vm, vk = evaluate(model, val_dl, device, cfg.loss.beta)
        eval_losses["loss"].append(vl)
        eval_losses["mse"].append(vm)
        eval_losses["kld"].append(vk)
        epochs.append(epoch)

        scheduler.step(vl)

        if epoch % cfg.logging.print_every == 0:
            print(
                f"Epoch {epoch:4d} - Train Loss: {tl / n:.4f} | Train MSE: {tm / n:.4f} | Train KLD: {tk / n:.4f} | "
                f"Val Loss: {vl:.4f} | Val MSE: {vm:.4f} | Val KLD: {vk:.4f}",
            )

        if vl < best:
            best = vl
            patience = 0
            torch.save(
                model.state_dict(),
                os.path.join(
                    cfg.paths.model_dir,
                    cfg.paths.model_file.format(dataset=cfg.dataset.name),
                ),
            )
        else:
            patience += 1
            if cfg.early_stopping.enabled and patience >= cfg.early_stopping.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(
        torch.load(
            os.path.join(
                cfg.paths.model_dir,
                cfg.paths.model_file.format(dataset=cfg.dataset.name),
            ),
        ),
    )

    save_learning_curves_plot(
        losses,
        eval_losses,
        epochs,
        os.path.join(cfg.paths.image_dir, f"learning_curves_{cfg.dataset.name}.png"),
    )

    print(
        f"CVAE saved to "
        f"{os.path.join(cfg.paths.model_dir, cfg.paths.model_file.format(dataset=cfg.dataset.name))} "
        f"(Best val loss: {best:.4f})",
    )


if __name__ == "__main__":
    main()
