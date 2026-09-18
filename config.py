import torch

RANDOM_STATE = 42
BATCH_SIZE = 4
NUM_EPOCHS = 300
DATASET_NAME = "iris"
CVAE_CONFIG = {
    "hidden_dim": 10,
    "latent_dim": 2,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYNTHETIC_SAMPLE_SIZE = 300
DATA_PATH = "./data"
MODELS_PATH = "./models"
IMAGES_PATH = "./img"
