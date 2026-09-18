from sklearn.datasets import load_iris


def load_dataset(name: str):
    if name == "iris":
        return load_iris(return_X_y=True)
