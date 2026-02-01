import pandas as pd
import yaml
import json

from datetime import datetime
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src import RAW_DATA_PATH, PROCESSED_DATA_PATH, MODELS_REGISTRY_PATH, PREDICTIONS_DATA_PATH
from src.processor import Processor
from src.nn_regressor import NNRegressor
from src.gb_regressor import GBRegressor

def main():
    '''Main workflow with validated baseline comparison.'''
    with open("configs/main.yaml", "r") as f:
        main_config = yaml.safe_load(f)
    
    with open("configs/datasets.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)[main_config["dataset_name"]]

    with open("configs/models.yaml", "r") as f:
        models_config = yaml.safe_load(f)
    
    gb_config = models_config["gradient_boosting"]

    raw_train_data_path = Path(RAW_DATA_PATH) / f"{main_config['dataset_name']}_train.csv"
    raw_test_data_path = Path(RAW_DATA_PATH) / f"{main_config['dataset_name']}_test.csv"

    processed_train_data_path = Path(PROCESSED_DATA_PATH) / f"{main_config['dataset_name']}_train.csv"
    processed_test_data_path = Path(PROCESSED_DATA_PATH) / f"{main_config['dataset_name']}_test.csv"
    
    raw_train_data = pd.read_csv(raw_train_data_path, index_col=dataset_config["id_column"])
    raw_test_data = pd.read_csv(raw_test_data_path, index_col=dataset_config["id_column"])

    if main_config["preprocess"]:
        processor = Processor(dataset_config)
        processed_train_data, processed_test_data = processor.run(raw_train_data, raw_test_data)

        processed_train_data.to_csv(processed_train_data_path, index=False)
        processed_test_data.to_csv(processed_test_data_path, index=False)
    
    else:
        processed_train_data = pd.read_csv(processed_train_data_path)
        processed_test_data = pd.read_csv(processed_test_data_path)

    gradient_boosting_model = GBRegressor(weak_learner_key=gb_config["weak_learner"], dataset_config=dataset_config)

    if main_config["train"]:
        X_train, X_valid, y_train, y_valid = train_test_split(
            processed_train_data.drop(columns=[dataset_config["target"]]), 
            processed_train_data[dataset_config["target"]],
            test_size=0.2
        )
        
        if main_config["use_gradient_boosting"]:
            weak_learner_config = models_config[gb_config["weak_learner"]]
            
            gradient_boosting_model.fit(
                X_train, y_train, X_valid, y_valid,
                upsilon=gb_config["upsilon"],
                learning_rate=gb_config["learning_rate"],
                patience=gb_config["patience"],
                max_iter=gb_config["max_iter"],
                init_params=weak_learner_config["init_params"] or {},
                fit_params=weak_learner_config["fit_params"] or {}
            )
            
            timestamp = datetime.now()
            formatted_timestamp = timestamp.strftime("%Y_%m_%d_%H_%M")
            gradient_boosting_model.save_model(f"models/{main_config['dataset_name']}/{formatted_timestamp}.joblib")

            try:
                with open(MODELS_REGISTRY_PATH, 'r') as file:
                    registry = json.load(file)
            
            except (json.JSONDecodeError, FileNotFoundError):
                registry = {}
            
            current_best_loss = registry.get(main_config["dataset_name"], {}).get(gb_config["weak_learner"], {}).get("validation_loss", float('inf'))
            if gradient_boosting_model.best_loss < current_best_loss:
                registry.setdefault(main_config["dataset_name"], {})[gb_config["weak_learner"]] = {
                    "best": formatted_timestamp,
                    "validation_loss": gradient_boosting_model.best_loss
                }
            
            with open("models/registry.json", 'w', encoding='utf-8') as file:
                json.dump(registry, file, indent=4)
        
        else:
            print("--- Executing Standalone Baseline Training ---")
            
            nn_init_params = models_config["neural_network"]["init_params"] or {}
            nn_fit_params = models_config["neural_network"]["fit_params"] or {}
            
            nn_model = NNRegressor(
                cat_cols=dataset_config["cat_cols"],
                num_cols=dataset_config["num_cols"],
                cat_cardinalities=dataset_config["cat_cardinalities"],
                **nn_init_params
            )

            nn_model.fit(X_train, y_train, **nn_fit_params )
            nn_val_mse = mean_squared_error(y_valid, nn_model.predict(X_valid))

            dt_init_params = models_config["decision_tree"]["init_params"] or {}
            dt_fit_params = models_config["decision_tree"]["fit_params"] or {}
            
            dt_model = DecisionTreeRegressor(**dt_init_params)
            dt_model.fit(X_train, y_train, **dt_fit_params)
            dt_val_mse = mean_squared_error(y_valid, dt_model.predict(X_valid))

            print(f"\n{'Model':<20} | {'Validation MSE':<10}\n" + "-"*35)
            print(f"{'Decision Tree':<20} | {dt_val_mse:.6f}")
            print(f"{'Neural Network':<20} | {nn_val_mse:.6f}\n")

    if main_config["predict"]:
        if main_config["train"]:
            preds = gradient_boosting_model.predict(processed_test_data)
        else:
            try:
                with open(MODELS_REGISTRY_PATH, 'r') as file:
                    registry = json.load(file)
            
            except (json.JSONDecodeError, FileNotFoundError):
                registry = {}
            
            best_model_filename = registry.get(main_config["dataset_name"], {}).get(gb_config["weak_learner"], {}).get("best")
            best_model_path = Path(PROCESSED_DATA_PATH) / main_config['dataset_name'] / best_model_filename
            
            gradient_boosting_model = GBRegressor(weak_learner_key=gb_config["weak_learner"])

            gradient_boosting_model.load_model(best_model_path)
            gradient_boosting_model.predict(processed_test_data)
        
        output_path = Path(PREDICTIONS_DATA_PATH) / f"{main_config['dataset_name']}_preds.csv"
        preds.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

if __name__=="__main__":
    main()