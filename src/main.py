import pandas as pd
import yaml

from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src import PROCESSED_DATA_PATH
from src.processor import Processor
from src.nn_regressor import NNRegressor
from src.gb_regressor import GBRegressor

def main():
    '''Main workflow with validated baseline comparison.'''
    with open("configs/models.yaml", "r") as f:
        models_config = yaml.safe_load(f)
    
    main_config = models_config["main"]
    gb_config = models_config["gradient_boosting"]
    
    with open("configs/datasets.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)[main_config["dataset_name"]]

    train_filename = f"{main_config['dataset_name']}_train.csv"
    test_filename = f"{main_config['dataset_name']}_test.csv"
    train_path = Path(PROCESSED_DATA_PATH) / train_filename
    
    if not train_path.exists():
        processor = Processor()
        processor.run(train_filename, test_filename)

    df_train = pd.read_csv(train_path)
    X_full = df_train.drop(columns=[dataset_config["target"]])
    y_full = df_train[dataset_config["target"]]

    if not main_config["predict_only"]:
        if main_config["use_gradient_boosting"]:
            weak_learner_config = models_config[gb_config["weak_learner"]]
            trainer = GBRegressionTrainer()
            trainer.run(
                train_filename,
                dataset_config["target"],
                gb_config["weak_learner"],
                gb_config["upsilon"],
                gb_config["learning_rate"],
                gb_config["patience"],
                gb_config["max_iter"],
                init_params=weak_learner_config["init_params"],
                fit_params=weak_learner_config["fit_params"]
            )
        else:
            print("--- Executing Standalone Baseline Training ---")
            
            X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2)

            print(len(X_train))

            dt_init_params = models_config["decision_tree"]["init_params"] or {}
            dt_fit_params = models_config["decision_tree"]["fit_params"] or {}
            dt_model = DecisionTreeRegressor(**dt_init_params)
            dt_model.fit(X_train, y_train, **dt_fit_params)
            dt_val_mse = mean_squared_error(y_val, dt_model.predict(X_val))
            
            nn_fit_params = models_config["neural_network"]["fit_params"] or {}
            nn_model = NNRegressor(input_size=X_train.shape[1])
            nn_model.fit(
                X_train, y_train, 
                epochs=nn_fit_params["epochs"], 
                learning_rate=nn_fit_params["learning_rate"],
                patience=nn_fit_params["patience"]
            )
            nn_val_mse = mean_squared_error(y_val, nn_model.predict(X_val))

            print("\n" + "="*30)
            print(f"{'Model':<20} | {'Validation MSE':<10}")
            print("-" * 35)
            print(f"{'Decision Tree':<20} | {dt_val_mse:.6f}")
            print(f"{'Neural Network':<20} | {nn_val_mse:.6f}")
            print("="*30 + "\n")

    if not main_config["train_only"]:
        predictor = GBRegressionPredictor()
        predictor.run(test_filename)

if __name__=="__main__":
    main()