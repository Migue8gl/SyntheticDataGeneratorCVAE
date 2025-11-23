import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils import set_all_seeds


def create_confusion_matrix(clf, X_test, y_test, title, path):
    disp = ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        cmap=plt.cm.Blues,
        normalize="true",
    )
    disp.ax_.set_title(title)

    plt.savefig(path)


def save_metrics(results_dir, filename, y_true, y_pred, cv_scores, title_prefix=""):
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    report_dict = classification_report(y_true, y_pred, output_dict=True)

    flat_metrics = {
        "title": title_prefix,
        "cv_scores": ",".join([f"{s:.4f}" for s in cv_scores]),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "accuracy": report_dict["accuracy"],
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_precision": report_dict["weighted avg"]["precision"],
        "weighted_recall": report_dict["weighted avg"]["recall"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "y_true": ",".join(map(str, y_true)),
        "y_pred": ",".join(map(str, y_pred)),
    }

    for cls, vals in report_dict.items():
        if cls in ["accuracy", "macro avg", "weighted avg"]:
            continue
        flat_metrics[f"class_{cls}_precision"] = vals["precision"]
        flat_metrics[f"class_{cls}_recall"] = vals["recall"]
        flat_metrics[f"class_{cls}_f1"] = vals["f1-score"]
        flat_metrics[f"class_{cls}_support"] = vals["support"]

    write_header = not os.path.exists(filepath)

    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat_metrics)

    return filepath


def main():
    cfg = OmegaConf.load("config/config.yaml")
    set_all_seeds(cfg.experiment.seed)

    X_test = np.load(os.path.join(cfg.paths.data_dir, f"X_test_{cfg.dataset.name}.npy"))
    y_test = np.load(os.path.join(cfg.paths.data_dir, f"y_test_{cfg.dataset.name}.npy"))
    X_train = np.load(
        os.path.join(cfg.paths.data_dir, f"X_train_{cfg.dataset.name}.npy"),
    )
    y_train = np.load(
        os.path.join(cfg.paths.data_dir, f"y_train_{cfg.dataset.name}.npy"),
    )

    X_synthetic = np.load(
        os.path.join(cfg.paths.data_dir, f"X_synthetic_{cfg.dataset.name}.npy"),
    )
    y_synthetic = np.load(
        os.path.join(cfg.paths.data_dir, f"y_synthetic_{cfg.dataset.name}.npy"),
    )

    k = cfg.experiment.size_k_folds
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.experiment.seed)

    print("\nReal Data\n")

    model_real = RandomForestClassifier(random_state=cfg.experiment.seed)

    cv_scores = cross_val_score(model_real, X_train, y_train, cv=kf, scoring="accuracy")

    print(f"\n--- Cross Validation ({k} folds) ---")
    print(f"Scores: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean():.4f}")
    print(f"Std deviation: {cv_scores.std():.4f}")

    model_real.fit(X_train, y_train)
    create_confusion_matrix(
        model_real,
        X_test,
        y_test,
        "Confusion matrix on Real Data",
        os.path.join(cfg.paths.image_dir, f"confusion_matrix_real_{cfg.dataset.name}"),
    )
    y_pred = model_real.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    print("\nSynthetic Data\n")

    model_syn = RandomForestClassifier(random_state=cfg.experiment.seed)

    cv_scores = cross_val_score(
        model_syn,
        X_synthetic,
        y_synthetic,
        cv=kf,
        scoring="accuracy",
    )

    print(f"\n--- Cross Validation ({k} folds) ---")
    print(f"Scores: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean():.4f}")
    print(f"Std deviation: {cv_scores.std():.4f}")

    model_syn.fit(X_synthetic, y_synthetic)
    create_confusion_matrix(
        model_syn,
        X_test,
        y_test,
        "Confusion matrix on Synthetic Data",
        os.path.join(cfg.paths.image_dir, f"confusion_matrix_syn_{cfg.dataset.name}"),
    )
    y_pred_syn = model_syn.predict(X_test)

    print("\n--- Classification Report (Trained on Synthetic → Test on Real) ---")
    print(classification_report(y_test, y_pred_syn))

    save_metrics(
        results_dir=cfg.paths.results_dir,
        filename=f"metrics_{cfg.dataset.name}.csv",
        y_true=y_test,
        y_pred=y_pred,
        cv_scores=cv_scores,
        title_prefix="real_data",
    )

    save_metrics(
        results_dir=cfg.paths.results_dir,
        filename=f"metrics_{cfg.dataset.name}.csv",
        y_true=y_test,
        y_pred=y_pred_syn,
        cv_scores=cv_scores,
        title_prefix="synthetic_data",
    )


if __name__ == "__main__":
    main()
