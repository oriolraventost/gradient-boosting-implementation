import yaml

from datetime import datetime
from sklearn.model_selection import train_test_split

from src import *
from src.data_loader import DataLoader
from src.processor import Processor
from src.gb_regressor import GBRegressor
from src.gb_classifier import GBClassifier

def main():
    """Main workflow for loading data, training, and generating predictions
    with gradient boosting.
    """
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    active_dataset = config["main"]["active_dataset"]
    active_dataset_config = config["datasets"][active_dataset]

    weak_learner_key = config["gradient_boosting"]["weak_learner_key"]
    weak_learner_config = config[weak_learner_key]

    data_loader = DataLoader(
            active_dataset,
            active_dataset_config["id_column"],
            active_dataset_config["target"]
        )

    if config["main"]["run_preprocess"]:
        raw_train, raw_test = data_loader.load_raw_data()

        processor = Processor(
            active_dataset_config["target"],
            active_dataset_config["id_column"]
        )

        train, test = processor.run(raw_train, raw_test)

        data_loader.save_processed_data(train, test)
    
    if config["main"]["train_and_predict"]:
        train, test = data_loader.load_processed_data()
    
        if active_dataset_config["problem_type"] == "regression":
            model = GBRegressor(
                **config["gradient_boosting"],
                weak_learner_config=weak_learner_config
            )

        else:       
            model = GBClassifier(
                **config["gradient_boosting"],
                weak_learner_config=weak_learner_config
            )

        X_train, X_valid, y_train, y_valid = train_test_split(
            train.drop(columns=[active_dataset_config["target"]]), 
            train[active_dataset_config["target"]],
            test_size=0.2,
            random_state=42
        )
        
        start_time = datetime.now()
        start_timestamp = start_time.strftime("%Y_%m_%d_%H_%M")

        model.fit(X_train, y_train, X_valid, y_valid)
            
        end_time = datetime.now()
        end_timestamp = end_time.strftime("%Y_%m_%d_%H_%M")
        
        data_loader.save_model(
            model,
            test,
            start_timestamp,
            end_timestamp,
            config["gradient_boosting"],
            weak_learner_config
        )

if __name__=="__main__":
    main()