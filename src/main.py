import yaml

from pathlib import Path

from src import PROCESSED_DATA_PATH
from src.processor import Processor
from src.gb_regression_trainer import GBRegressionTrainer
from src.gb_regression_predictor import GBRegressionPredictor

def main():
    '''Main workflow'''
    with open("configs/models.yaml", "r") as f:
        models_config = yaml.safe_load(f)
        gradient_boosting_config = models_config["gradient_boosting"]
        weak_learner = gradient_boosting_config["weak_learner"]
        weak_learner_config = models_config[weak_learner]
    
    with open("configs/datasets.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)[gradient_boosting_config["dataset_name"]]
    
    train_filename = f"{gradient_boosting_config["dataset_name"]}_train.csv"
    test_filename = f"{gradient_boosting_config["dataset_name"]}_test.csv"

    train_path = Path(f"{PROCESSED_DATA_PATH}/{train_filename}")
    test_path = Path(f"{PROCESSED_DATA_PATH}/{test_filename}")

    if not (train_path.exists() and test_path.exists()):
        processor = Processor()
        processor.run(train_filename, test_filename)
    
    if not gradient_boosting_config["predict_only"]:
        print('HI')
        gb_regression_trainer = GBRegressionTrainer()
        gb_regression_trainer.run(
            train_filename,
            dataset_config["target"],
            gradient_boosting_config["weak_learner"],
            gradient_boosting_config["upsilon"],
            gradient_boosting_config["learning_rate"],
            gradient_boosting_config["patience"],
            gradient_boosting_config["max_iter"],
            init_params=weak_learner_config["init_params"],
            fit_params=weak_learner_config["fit_params"]
        )

    if not gradient_boosting_config["train_only"]:
        gb_regression_predictor = GBRegressionPredictor()
        gb_regression_predictor.run(test_filename)

if __name__=="__main__":
    main()