import os

import numpy as np
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt


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


def main():
    cfg = OmegaConf.load("config/config.yaml")

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


if __name__ == "__main__":
    main()
