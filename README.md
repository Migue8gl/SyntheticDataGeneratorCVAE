# Synthetic Data Generator

This repository provides a pipeline for generating synthetic tabular data using a Conditional Variational Autoencoder (CVAE), and evaluating the quality of the generated data by training classifiers on both real and synthetic datasets. The project is designed for reproducible experiments on classic datasets such as Iris, Wine, and Breast Cancer.

## Features

- **CVAE Training:** Train a Conditional Variational Autoencoder on a selected dataset.
- **Synthetic Data Generation:** Generate synthetic samples conditioned on class labels.
- **Evaluation:** Compare classifier performance on real vs. synthetic data.
- **Visualization:** Visualize feature distributions and sample embeddings.
- **Experiment Management:** Easy-to-use bash script for running and cleaning experiments.

## Repository Structure

```
.
├── config/
│   └── config.yaml         # Main configuration file
├── data/                   # Stores processed and synthetic data
├── img/                    # Stores generated plots and images
├── models/                 # Stores trained model files and scalers
├── results/                # Stores experiment results and metrics
├── train_vae.py            # CVAE training script
├── generate_synthetic.py   # Synthetic data generation and visualization
├── synthetic_training.py   # Evaluation of real vs synthetic data
├── utils.py                # Utility functions (e.g., seed setting)
├── experiment.sh           # Main experiment runner and cleaner
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/SyntheticDataGenerator.git
   cd SyntheticDataGenerator
   ```

2. **Install [uv](https://github.com/astral-sh/uv) (if not already installed):**
   ```bash
   pip install uv
   ```

3. **Create and activate the virtual environment:**
   The `experiment.sh` script will automatically create a `.venv` folder using Python 3.12 if it does not exist.
   ```bash
   bash experiment.sh --run
   ```
   - If `.venv` does not exist, it will be created and dependencies installed.
   - If `.venv` exists, it will be activated and the pipeline will run.

## Usage

### Make the Script Executable

Before running the experiment script, you must give it execute permissions:

```bash
chmod +x experiment.sh
```

### Run the Full Experiment Pipeline

```bash
./experiment.sh --run
```

This will:
1. Activate the virtual environment.
2. Train the CVAE model on the selected dataset.
3. Generate synthetic data.
4. Evaluate classifier performance on both real and synthetic data.
5. Save results, metrics, and visualizations in the appropriate folders.

### Clean Experiment Artifacts

To remove all generated data, results, images, and models:

```bash
./experiment.sh --clean
```

This will delete the `data/`, `results/`, `img/`, and `models/` directories.

## Configuration

All experiment settings are managed in `config/config.yaml`. You can change:
- Dataset (`iris`, `wine`, `breast_cancer`)
- Model hyperparameters
- Training settings
- Paths for saving outputs

## Requirements

- Python 3.12
- [uv](https://github.com/astral-sh/uv) for fast environment management
- All Python dependencies are listed in `requirements.txt`

## How It Works

1. **train_vae.py:** Loads the dataset, preprocesses it, trains the CVAE, and saves the model and scaler.
2. **generate_synthetic.py:** Loads the trained CVAE, generates synthetic samples, and visualizes distributions.
3. **synthetic_training.py:** Trains a classifier on both real and synthetic data, compares performance, and saves metrics.
4. **experiment.sh:** Orchestrates the entire workflow and manages the environment.

## Visualization

- Feature distributions and overlays are saved in the `img/` folder.
- UMAP visualizations for high-dimensional data are also provided.

## Extending

You can add new datasets or models by modifying the scripts and updating the configuration file.

## Troubleshooting

- If you encounter issues with the virtual environment, delete the `.venv` folder and rerun the script.
- Make sure Python 3.12 is installed on your system.

## License

This project is licensed under the MIT License.

---

**For questions or contributions, please open an issue or pull request.**
