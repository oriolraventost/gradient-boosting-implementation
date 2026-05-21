# Gradient Boosting Implementation

A modular implementation of Gradient Boosting built from scratch using PyTorch and Scikit-Learn base learners. In addition to traditional decision trees, it allows Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs) to be used as base weak learners.

---

## Project Structure

```text
├── config/
│   ├── modeling.yaml          # Runtime configuration and model hyperparameters
│   └── datasets.yaml          # Datasets description
├── data/
│   └── dataset_key/           # Dataset-specific directories (e.g., heart, scores)
│       ├── raw/               # Single raw source file (data.csv)
│       └── processed/         # Processed splits (train.csv, valid.csv, test.csv)
├── models/
│   └── dataset_key/           # Training artifacts
│       ├── info.json          # Evaluation metrics and model hyperparameters
│       ├── predictions.csv    # Model predictions
│       └── YYYY_MM_DD_HH_MM.joblib  # Serialized model artifacts
├── notebooks/                 # Comparison of different modeling techniques on basic datasets
│   ├── cifar10.ipynb
│   ├── mnist.ipynb
│   ├── scores.ipynb
│   └── heart.ipynb
├── src/                       # Core framework source code
│   ├── __init__.py            # Initialization and global configuration
│   ├── main.py                # Pipeline entry point
│   ├── data_manager.py        # Data loading and saving methods (I/O)
│   ├── processor.py           # Feature engineering and preprocessing
│   ├── gb_classifier.py       # Gradient Boosting classifier
│   ├── gb_regressor.py        # Gradient Boosting regressor
│   ├── nn_regressor.py        # MLP-based learner module
│   ├── cnn_regressor.py       # CNN-based learner module
│   └── utils.py               # Utility functions
├── .gitignore                 # Git exclusions
├── requirements.txt           # Project dependencies
└── README.md                  # Documentation
```

---

## Extensibility & Configuration

The framework is designed to be easy to configure and extend to new datasets.

### 1. Customizing Model Hyperparameters

You can adjust the boosting engine and neural network settings in `config/modeling.yaml` without changing the source code:

```yaml
gradient_boosting:
  n_estimators: 1200  # Number of boosting iterations
  learning_rate: 0.1  # Optimization step size

neural_network:
  hidden_size:
    - 32
    - 16
    - 8
```

### 2. Onboarding New Tabular Datasets

To add a new tabular dataset:

* Create the dataset folder and stage the data:

```text
data/
└── new_dataset_key/
    └── raw/
        └── data.csv
```

* Define the schema in `config/datasets.yaml`:

```yaml
new_dataset_key:
  id_column: "user_id"
  target: "target_column_name"  # Target column name in the dataset
  problem_type: "regression"    # or "classification"
  cat_cols:
    - "categorical_feature_1"
  num_cols:
    - "numerical_feature_1"
```

* Activate the dataset in `config/modeling.yaml` by setting `active_dataset` to `"new_dataset_key"`.

### 3. Supported Tabular Datasets

The tabular datasets already configured in `config/datasets.yaml` (`heart` and `scores`) can be downloaded from:

- **Playground Series S6E1**: [https://www.kaggle.com/competitions/playground-series-s6e1](https://www.kaggle.com/competitions/playground-series-s6e1)
- **Playground Series S6E2**: [https://www.kaggle.com/competitions/playground-series-s6e2](https://www.kaggle.com/competitions/playground-series-s6e2)

After downloading, place the CSV as `data/scores/raw/data.csv` or `data/heart/raw/data.csv`.

### 4. Using Computer Vision Datasets

For computer vision tasks, you do not need to create a `raw/` directory with `data.csv`. The implementation currently supports only MNIST and CIFAR-10.

To use one of these datasets:

* In `config/datasets.yaml`, ensure the dataset is defined with `problem_type: "computer_vision"` (already included by default).

* In `config/modeling.yaml`:

```yaml
main:
  active_dataset: "mnist"      # or "cifar10"
  run_preprocess: true         # must be True to load and preprocess the images
  train_and_predict: true
```

The framework will load and preprocess the built-in MNIST or CIFAR-10 dataset. No dataset folder needs to be created for these.

---

## Workflow Execution

### 1. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Configure the Pipeline

Set execution parameters in `config/modeling.yaml`:

```yaml
main:
  active_dataset: "scores"
  run_preprocess: true
  train_and_predict: true
```

### 3. Execute the Pipeline

Run the main entry point:

```bash
python -m src.main
```

### 4. Review Outputs

After execution, results are stored in `models/dataset_key/`:

* `info.json`: Validation metrics (accuracy, R², etc.).
* `predictions.csv`: Model predictions on the test split.
* `.joblib` files: Serialized models for reuse and deployment.

---

## Disclaimer

This repository is not perfect. There may be bugs and edge cases that are not handled. Use it at your own risk, and feel free to open issues or submit pull requests if you find problems or have suggestions for improvement.