## Project Structure

```text
├── configs/                    # YAML configuration files
│   ├── main.yaml               # Main module
│   ├── gradient_boosting.yaml  # Gradient boosting parameters
│   └── datasets.yaml           # Datasets names and columns
├── data/                       # Data storage (git-ignored)
│   ├── raw/                    # As downloaded from external site
│   │   ├── dataset_train.csv
│   │   └── dataset_test.csv
│   └── processed/              # Processed ready to train
│   │   ├── dataset_train.csv
│   │   └── dataset_test.csv
│   └── predictions/            # Predictions generated
│   │   └── dataset_preds.csv
├── models/                     # Models storage (git-ignored)
│   ├── dataset/
│   │   └── YYYY_MM_DD_HH_MM.joblib
│   └── registry.json
├── src/                        # Source code
│   ├── __init__.py             # Constant variables (paths)
│   ├── gb_classifier.py        # Gradient boosting classifier
│   ├── gb_regressor.py         # Gradient boosting regressor
│   ├── main.py                 # Entrypoint
│   ├── nn_regressor.py         # Neural network regressor
│   ├── processor.py            # Data processor (raw -> processed)
│   ├── regression.py           # XGBoost regression
│   ├── tree.py                 # XGBoost tree
│   └── utils.py                # Helper functions
├── .gitignore                  # Standard git ignore file
├── requirements.txt            # Project dependencies
└── README.md                   # Instructions
```