import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import OmegaConf
from umap import UMAP

from train_vae import CVAE

SYNTHETIC_DATA_SIZE = 600


def plot_real_vs_synthetic(
    real_data,
    real_labels,
    synthetic_data,
    synthetic_labels,
    path,
):
    num_features = real_data.shape[1]
    unique_classes = np.unique(real_labels)
    palette = sns.color_palette("husl", len(unique_classes))
    num_cols = 3
    num_rows = ((num_features * 2) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 6 * num_rows))
    axes = axes.flatten()
    plot_idx = 0
    for feat_idx in range(num_features):
        ax = axes[plot_idx]
        for idx, class_label in enumerate(unique_classes):
            real_mask = real_labels == class_label
            ax.hist(
                real_data[real_mask, feat_idx],
                bins=30,
                alpha=0.6,
                label=f"Class {class_label}",
                color=palette[idx],
                density=True,
            )
        ax.set_title(
            f"Real Feature {feat_idx + 1} Distribution",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(f"Feature {feat_idx + 1} Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    for feat_idx in range(num_features):
        ax = axes[plot_idx]
        for idx, class_label in enumerate(unique_classes):
            synth_mask = synthetic_labels == class_label
            ax.hist(
                synthetic_data[synth_mask, feat_idx],
                bins=30,
                alpha=0.6,
                label=f"Class {class_label}",
                color=palette[idx],
                density=True,
            )
        ax.set_title(
            f"Synthetic Feature {feat_idx + 1} Distribution",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(f"Feature {feat_idx + 1} Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison_overlays(
    real_data,
    real_labels,
    synthetic_data,
    synthetic_labels,
    path,
):
    num_features = real_data.shape[1]
    unique_classes = np.unique(real_labels)
    palette = sns.color_palette("husl", len(unique_classes))
    num_cols = min(3, num_features)
    num_rows = (num_features + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows))
    if num_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for feat_idx in range(num_features):
        ax = axes[feat_idx]
        for idx, class_label in enumerate(unique_classes):
            real_mask = real_labels == class_label
            synth_mask = synthetic_labels == class_label
            ax.hist(
                real_data[real_mask, feat_idx],
                bins=30,
                alpha=0.4,
                label=f"Real Class {class_label}",
                color=palette[idx],
                density=True,
                edgecolor="black",
                linewidth=1.5,
            )
            ax.hist(
                synthetic_data[synth_mask, feat_idx],
                bins=30,
                alpha=0.4,
                label=f"Synth Class {class_label}",
                color=palette[idx],
                density=True,
                linestyle="--",
                histtype="step",
                linewidth=2,
            )
        ax.set_title(
            f"Feature {feat_idx + 1}: Real vs Synthetic",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel(f"Feature {feat_idx + 1} Value", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
    for idx in range(num_features, len(axes)):
        axes[idx].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_sample_visualization(
    real_data,
    real_labels,
    synthetic_data,
    synthetic_labels,
    path,
    n_samples=200,
):
    unique_classes = np.unique(real_labels)
    palette = sns.color_palette("husl", len(unique_classes))
    class_colors = {cls: palette[i] for i, cls in enumerate(unique_classes)}
    if real_data.shape[1] == 1:
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        x_real = np.arange(min(len(real_data), n_samples))
        x_synth = np.arange(min(len(synthetic_data), n_samples))
        for class_label in unique_classes:
            mask_real = real_labels == class_label
            mask_synth = synthetic_labels == class_label
            samples_real = real_data[mask_real][:n_samples, 0]
            samples_synth = synthetic_data[mask_synth][:n_samples, 0]
            ax.scatter(
                x_real[: len(samples_real)],
                samples_real,
                c=[class_colors[class_label]],
                label=f"Real Class {class_label}",
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
            ax.scatter(
                x_synth[: len(samples_synth)] + len(real_data),
                samples_synth,
                c=[class_colors[class_label]],
                label=f"Synth Class {class_label}",
                alpha=0.6,
                s=50,
                marker="^",
                edgecolors="black",
                linewidth=0.5,
            )
        ax.axvline(
            x=len(real_data),
            color="red",
            linestyle="--",
            linewidth=2,
            label="Real/Synth Boundary",
        )
        ax.set_title("Real vs Synthetic Data Samples", fontsize=16, fontweight="bold")
        ax.set_xlabel("Sample Index", fontsize=13)
        ax.set_ylabel("Feature Value", fontsize=13)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
    elif real_data.shape[1] == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax = axes[0]
        for class_label in unique_classes:
            mask = real_labels == class_label
            samples = real_data[mask][:n_samples]
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                c=[class_colors[class_label]],
                label=f"Class {class_label}",
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
        ax.set_title("Real Data Samples (2D)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Feature 1", fontsize=13)
        ax.set_ylabel("Feature 2", fontsize=13)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        for class_label in unique_classes:
            mask = synthetic_labels == class_label
            samples = synthetic_data[mask][:n_samples]
            ax.scatter(
                samples[:, 0],
                samples[:, 1],
                c=[class_colors[class_label]],
                label=f"Class {class_label}",
                alpha=0.6,
                s=50,
                marker="^",
                edgecolors="black",
                linewidth=0.5,
            )
        ax.set_title("Synthetic Data Samples (2D)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Feature 1", fontsize=13)
        ax.set_ylabel("Feature 2", fontsize=13)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        real_subset_indices = []
        synth_subset_indices = []
        for class_label in unique_classes:
            real_mask = np.where(real_labels == class_label)[0]
            synth_mask = np.where(synthetic_labels == class_label)[0]
            real_subset_indices.extend(real_mask[: min(n_samples, len(real_mask))])
            synth_subset_indices.extend(synth_mask[: min(n_samples, len(synth_mask))])
        real_subset = real_data[real_subset_indices]
        real_labels_subset = real_labels[real_subset_indices]
        synth_subset = synthetic_data[synth_subset_indices]
        synth_labels_subset = synthetic_labels[synth_subset_indices]
        combined_data = np.vstack([real_subset, synth_subset])
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(combined_data)
        real_embedding = embedding[: len(real_subset)]
        synth_embedding = embedding[len(real_subset) :]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax = axes[0]
        for class_label in unique_classes:
            mask = real_labels_subset == class_label
            ax.scatter(
                real_embedding[mask, 0],
                real_embedding[mask, 1],
                c=[class_colors[class_label]],
                label=f"Class {class_label}",
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )
        ax.set_title("Real Data Samples (UMAP)", fontsize=16, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=13)
        ax.set_ylabel("UMAP 2", fontsize=13)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax = axes[1]
        for class_label in unique_classes:
            mask = synth_labels_subset == class_label
            ax.scatter(
                synth_embedding[mask, 0],
                synth_embedding[mask, 1],
                c=[class_colors[class_label]],
                label=f"Class {class_label}",
                alpha=0.6,
                s=50,
                marker="^",
                edgecolors="black",
                linewidth=0.5,
            )
        ax.set_title("Synthetic Data Samples (UMAP)", fontsize=16, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=13)
        ax.set_ylabel("UMAP 2", fontsize=13)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()


def sample_n(model, class_idx, num_classes, latent_dim, n):
    y = torch.nn.functional.one_hot(torch.tensor([class_idx] * n), num_classes).float()
    with torch.inference_mode():
        z = torch.randn(n, latent_dim, device=y.device)
        return model.decode(z, y)


def compute_and_save_statistics(
    X_train,
    y_train,
    samples,
    synthetic_labels,
    output_path,
):
    stats_data = []

    stats_data.append(
        {
            "category": "Overall",
            "data_type": "Synthetic",
            "class": "All",
            "metric": "std",
            "value": samples.std(),
        },
    )
    stats_data.append(
        {
            "category": "Overall",
            "data_type": "Synthetic",
            "class": "All",
            "metric": "mean",
            "value": samples.mean(),
        },
    )
    stats_data.append(
        {
            "category": "Overall",
            "data_type": "Synthetic",
            "class": "All",
            "metric": "shape_samples",
            "value": samples.shape[0],
        },
    )
    stats_data.append(
        {
            "category": "Overall",
            "data_type": "Synthetic",
            "class": "All",
            "metric": "shape_features",
            "value": samples.shape[1],
        },
    )
    stats_data.append(
        {
            "category": "Overall",
            "data_type": "Synthetic",
            "class": "All",
            "metric": "num_classes",
            "value": len(np.unique(synthetic_labels)),
        },
    )

    stats_data.append(
        {
            "category": "Overall",
            "data_type": "Real",
            "class": "All",
            "metric": "std",
            "value": X_train.std(),
        },
    )
    stats_data.append(
        {
            "category": "Overall",
            "data_type": "Real",
            "class": "All",
            "metric": "mean",
            "value": X_train.mean(),
        },
    )
    stats_data.append(
        {
            "category": "Overall",
            "data_type": "Real",
            "class": "All",
            "metric": "shape_samples",
            "value": X_train.shape[0],
        },
    )
    stats_data.append(
        {
            "category": "Overall",
            "data_type": "Real",
            "class": "All",
            "metric": "shape_features",
            "value": X_train.shape[1],
        },
    )
    stats_data.append(
        {
            "category": "Overall",
            "data_type": "Real",
            "class": "All",
            "metric": "num_classes",
            "value": len(np.unique(y_train)),
        },
    )

    for class_idx in sorted(set(y_train)):
        real_mask = y_train == class_idx
        synth_mask = synthetic_labels == class_idx

        stats_data.append(
            {
                "category": "Class-wise",
                "data_type": "Real",
                "class": int(class_idx),
                "metric": "sample_count",
                "value": real_mask.sum(),
            },
        )
        stats_data.append(
            {
                "category": "Class-wise",
                "data_type": "Real",
                "class": int(class_idx),
                "metric": "mean",
                "value": X_train[real_mask].mean(),
            },
        )
        stats_data.append(
            {
                "category": "Class-wise",
                "data_type": "Real",
                "class": int(class_idx),
                "metric": "std",
                "value": X_train[real_mask].std(),
            },
        )

        # Synthetic data class statistics
        stats_data.append(
            {
                "category": "Class-wise",
                "data_type": "Synthetic",
                "class": int(class_idx),
                "metric": "sample_count",
                "value": synth_mask.sum(),
            },
        )
        stats_data.append(
            {
                "category": "Class-wise",
                "data_type": "Synthetic",
                "class": int(class_idx),
                "metric": "mean",
                "value": samples[synth_mask].mean(),
            },
        )
        stats_data.append(
            {
                "category": "Class-wise",
                "data_type": "Synthetic",
                "class": int(class_idx),
                "metric": "std",
                "value": samples[synth_mask].std(),
            },
        )

    # Create DataFrame and save to CSV
    df = pd.DataFrame(stats_data)
    df.to_csv(output_path, index=False)

    print(f"\nStatistics saved to: {output_path}")
    print(f"Total rows: {len(df)}")

    return df


def main():
    cfg = OmegaConf.load("config/config.yaml")
    X_test = np.load(
        os.path.join(cfg.paths.data_dir, f"X_test_{cfg.dataset.name}.npy"),
    )
    y_test = np.load(os.path.join(cfg.paths.data_dir, f"y_test_{cfg.dataset.name}.npy"))
    X_train = np.load(
        os.path.join(cfg.paths.data_dir, f"X_train_{cfg.dataset.name}.npy"),
    )
    y_train = np.load(
        os.path.join(cfg.paths.data_dir, f"y_train_{cfg.dataset.name}.npy"),
    )
    input_dim = X_test.shape[1]
    hidden_dim = int(input_dim * cfg.model.hidden_dim_factor)
    latent_dim = int(input_dim * cfg.model.latent_dim_factor)
    num_classes = len(set(y_test))
    model = CVAE(input_dim, hidden_dim, latent_dim, num_classes)
    dataset_name = cfg.dataset.name
    model.load_state_dict(
        torch.load(
            os.path.join(
                cfg.paths.model_dir,
                cfg.paths.model_file.format(dataset=dataset_name),
            ),
        ),
    )
    model.eval()
    all_samples = []
    all_labels = []
    samples_each_class = SYNTHETIC_DATA_SIZE // num_classes
    for class_idx in sorted(set(y_test)):
        class_samples = sample_n(
            model,
            class_idx=class_idx,
            num_classes=num_classes,
            latent_dim=latent_dim,
            n=samples_each_class,
        )
        all_samples.append(class_samples)
        all_labels.append(np.full(samples_each_class, class_idx))
    samples = torch.cat(all_samples, dim=0).cpu().numpy()
    synthetic_labels = np.concatenate(all_labels)

    os.makedirs("img", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    compute_and_save_statistics(
        X_train,
        y_train,
        samples,
        synthetic_labels,
        f"results/statistics_{dataset_name}.csv",
    )
    plot_real_vs_synthetic(
        X_train,
        y_train,
        samples,
        synthetic_labels,
        "img/synthetic_vs_real_separate.png",
    )
    plot_comparison_overlays(
        X_train,
        y_train,
        samples,
        synthetic_labels,
        "img/synthetic_vs_real_overlay.png",
    )
    plot_sample_visualization(
        X_train,
        y_train,
        samples,
        synthetic_labels,
        "img/synthetic_vs_real_samples.png",
        n_samples=200,
    )


if __name__ == "__main__":
    main()
