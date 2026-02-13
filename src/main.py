import pandas as pd
import yaml
import json

from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

from src import *
from src.data_loader import DataLoader
from src.processor import Processor
from src.gb_regressor import GBRegressor
from src.gb_classifier import GBClassifier
from src.utils import preprocess

def main():
    """Main workflow for loading data, training, and generating predictions
    with gradient boosting.
    """
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    data_loader = DataLoader(dataset_name=config["main"]["active_dataset"])
    raw_train, raw_test = data_loader.load_raw_data()

    if config["main"]["run_preprocess"]:
        processor = Processor()
        processed_train_data, processed_test_data = preprocess(
            raw_train_features,
            raw_train_labels,
            raw_test_features,
            dataset_config["target"],
            dataset_config["id"]
        )

        processed_train_data.to_csv(processed_train_data_path, index=False)
        processed_test_data.to_csv(processed_test_data_path, index=False)
    
    else:
        processed_train_data = pd.read_csv(processed_train_data_path)
        processed_test_data = pd.read_csv(processed_test_data_path)

    if main_config["problem_type"] == "regression":
        model = GBRegressor(**gradient_boosting_config)

    elif main_config["problem_type"] == "classification":
        model = GBClassifier(**gradient_boosting_config)

    if main_config["train"]:
        X_train, X_valid, y_train, y_valid = train_test_split(
            processed_train_data.drop(columns=[dataset_config["target"]]), 
            processed_train_data[dataset_config["target"]],
            test_size=0.2,
            random_state=42
        )
        
        start_time = datetime.now()
        formatted_start_timestamp = start_time.strftime("%Y_%m_%d_%H_%M")

        model.fit(X_train, y_train, X_valid, y_valid)
            
        end_time = datetime.now()
        formatted_end_timestamp = end_time.strftime("%Y_%m_%d_%H_%M")

        model_dir = Path(MODELS_PATH)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{formatted_end_timestamp}.joblib"
        model.save_model(str(model_path))
        
        try:
            with open(MODELS_REGISTRY_PATH, 'r') as file:
                registry = json.load(file)
            
        except (json.JSONDecodeError, FileNotFoundError):
            registry = {}
            
        current_best_loss = registry.get(main_config["dataset_name"], {}).get("validation_loss", float('inf'))
        if model.best_loss < current_best_loss:
            dataset_entry = registry.setdefault(main_config["dataset_name"], {})
            dataset_entry.update({
                "best": formatted_start_timestamp,
                "validation_loss": model.best_loss,
                "start_time": formatted_start_timestamp,
                "end_time": formatted_end_timestamp,
                **gradient_boosting_config,
            })
            
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
            
            best_model_filename = registry.get("best")
            best_model_path = Path(MODELS_PATH) / f"{best_model_filename}.joblib"

            model.load_model(best_model_path)
            preds = model.predict(processed_test_data)
        
        output_path = Path(PREDICTIONS_DATA_PATH) / "preds.csv"
        submission[dataset_config["target"]] = preds.astype(int)
        submission.to_csv(output_path, index=False)

if __name__=="__main__":
    main()