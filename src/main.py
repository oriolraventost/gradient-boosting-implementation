import pandas as pd
import yaml
import json

from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

from src import *
from src.processor import Processor
from src.gb_regressor import GBRegressor

def main():
    '''Main workflow for loading data, training and generating predictions with gradient boosting.'''
    with open(MAIN_CONFIG_PATH, "r") as f:
        main_config = yaml.safe_load(f)
    
    with open(GRADIENT_BOOSTING_CONFIG_PATH, "r") as f:
        gb_config = yaml.safe_load(f)
    
    with open(DATASETS_CONFIG_PATH, "r") as f:
        dataset_config = yaml.safe_load(f)[main_config["dataset_name"]]

    raw_train_data_path = Path(RAW_DATA_PATH) / f"{main_config['dataset_name']}_train.csv"
    raw_test_data_path = Path(RAW_DATA_PATH) / f"{main_config['dataset_name']}_test.csv"

    processed_train_data_path = Path(PROCESSED_DATA_PATH) / f"{main_config['dataset_name']}_train.csv"
    processed_test_data_path = Path(PROCESSED_DATA_PATH) / f"{main_config['dataset_name']}_test.csv"
    
    raw_train_data = pd.read_csv(raw_train_data_path, index_col=dataset_config["id"])
    raw_test_data = pd.read_csv(raw_test_data_path, index_col=dataset_config["id"])

    if main_config["preprocess"]:
        processor = Processor()
        processed_train_data, processed_test_data = processor.run(
            train_data=raw_train_data,
            test_data=raw_test_data,
            target=dataset_config["target"],
            id=dataset_config["id"]
        )

        processed_train_data.to_csv(processed_train_data_path, index=False)
        processed_test_data.to_csv(processed_test_data_path, index=False)
    
    else:
        processed_train_data = pd.read_csv(processed_train_data_path)
        processed_test_data = pd.read_csv(processed_test_data_path)

    model = GBRegressor()

    if main_config["train"]:
        X_train, X_valid, y_train, y_valid = train_test_split(
            processed_train_data.drop(columns=[dataset_config["target"]]), 
            processed_train_data[dataset_config["target"]],
            test_size=0.2,
            random_state=42
        )
            
        model.fit(
            X_train, y_train, X_valid, y_valid,
            upsilon=gb_config["upsilon"],
            learning_rate=gb_config["learning_rate"],
            patience=gb_config["patience"],
            max_iter=gb_config["max_iter"],
            max_depth=gb_config["max_depth"]
        )
            
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y_%m_%d_%H_%M")
        model.save_model(f"models/{main_config['dataset_name']}/{formatted_timestamp}.joblib")

        try:
            with open(MODELS_REGISTRY_PATH, 'r') as file:
                registry = json.load(file)
            
        except (json.JSONDecodeError, FileNotFoundError):
            registry = {}
            
        current_best_loss = registry.get(main_config["dataset_name"], {}).get("validation_loss", float('inf'))
        if model.best_loss < current_best_loss:
            registry.setdefault(main_config["dataset_name"], {}) == {
                "best": formatted_timestamp,
                "validation_loss": model.best_loss
            }
            
        with open("models/registry.json", 'w', encoding='utf-8') as file:
            json.dump(registry, file, indent=4)

    if main_config["predict"]:
       
        if main_config["train"]:
            preds = model.predict(processed_test_data)
        
        else:
            try:
                with open(MODELS_REGISTRY_PATH, 'r') as file:
                    registry = json.load(file)
            
            except (json.JSONDecodeError, FileNotFoundError):
                registry = {}
            
            best_model_filename = registry.get(main_config["dataset_name"], {}).get("best")
            best_model_path = Path(MODELS_PATH) / main_config['dataset_name'] / f"{best_model_filename}.joblib"

            model.load_model(best_model_path)
            preds = model.predict(processed_test_data)
        
        output_path = Path(PREDICTIONS_DATA_PATH) / f"{main_config['dataset_name']}_preds.csv"
        preds.to_csv(output_path, index=False)

if __name__=="__main__":
    main()