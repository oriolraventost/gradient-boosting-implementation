from src import *
from src.data_manager import DataManager
from src.processor import Processor
from src.gb_regressor import GBRegressor
from src.gb_classifier import GBClassifier

def main():
    """Main workflow for loading data, training, and
    generating predictions with gradient boosting.
    """
    data_manager = DataManager()
    
    datasets_config, modeling_config = data_manager.load_config()

    active_dataset = modeling_config["main"]["active_dataset"]
    active_dataset_config = datasets_config[active_dataset]

    gradient_boosting_config = modeling_config["gradient_boosting"]
    
    weak_learner_key = gradient_boosting_config["weak_learner_key"]
    weak_learner_config = modeling_config[weak_learner_key]

    processor = Processor(**active_dataset_config)

    if modeling_config["main"]["run_preprocess"]:  
        raw_train, raw_test = data_manager.load_raw_data()
        
        train, valid, test = processor.transform_features(raw_train, raw_test)
        data_manager.save_processed_data(train, valid, test)
    
    else:
        train, valid, test = data_manager.load_processed_data()
    
    if modeling_config["main"]["train_and_predict"]:
        target = active_dataset_config["target"]
        
        X_train, X_valid = (
            train.drop(columns=[target]).values,
            valid.drop(columns=[target]).values
        )

        y_train, y_valid = processor.transform_target(
            train[target],
            valid[target]
        )
        
        if active_dataset_config["problem_type"] == "regression":
            model = GBRegressor(
                **gradient_boosting_config,
                weak_learner_config=weak_learner_config
            )
            
        else:       
            model = GBClassifier(
                **gradient_boosting_config,
                weak_learner_config=weak_learner_config
            )

        model.fit(X_train, y_train, X_valid, y_valid)

        data_manager.save_model(
            gradient_boosting_config,
            weak_learner_config,
            test,
            model,
            processor.target_transformer
        )

if __name__=="__main__":
    main()