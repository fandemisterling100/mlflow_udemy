import warnings
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import pickle

from sklearn.metrics import accuracy_score

# Import model selection libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the iris csv file from the URL
    data = pd.read_csv("iris.csv")

    data.to_csv("data/iris.csv", index=False)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train.to_csv("data/train.csv")
    test.to_csv("data/test.csv")
    # The predicted column is "variety"
    train_x = train.drop(["variety"], axis=1)
    test_x = test.drop(["variety"], axis=1)
    train_y = train[["variety"]]
    test_y = test[["variety"]]

    mlflow.set_tracking_uri(uri="")

    print("The set tracking uri is ", mlflow.get_tracking_uri())
    exp = mlflow.set_experiment(experiment_name="iris_assignment")

    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    mlflow.start_run()
    tags = {
        "engineering": "ML platform",
        "release.candidate":"RC1",
        "release.version": "2.0",
        "owner": "mjaramillo",
        "assignment": "iris",
    }

    mlflow.set_tags(tags)

    # Define the hyperparameter grid
    grid = {'n_estimators': [10, 50, 100, 200],
                'max_depth': [8, 9, 10, 11, 12,13, 14, 15],
                'min_samples_split': [2, 3, 4, 5]}

    # Initialize the model
    rf = RandomForestClassifier(random_state=0)

    # Repeated stratified kfold
    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)

    # Initialize RandomSearchCV
    random_search = RandomizedSearchCV(rf, grid,cv=rskf, n_iter=10, n_jobs=-1)

    # Fit the RandomSearchCV to the training data
    random_search.fit(train_x, train_y)

    # Select the best hyperparameters
    best_params = random_search.best_params_
    print("Best hyperparameters: ", best_params)
    
    # Initialize model with best parameters
    rf_model2 = RandomForestClassifier(n_estimators = best_params['n_estimators'],
                                    min_samples_leaf= best_params['min_samples_split'],
                                    max_depth = best_params['max_depth'],
                                    random_state=0)
    
    # Fit the model to the training data.
    rf_model2.fit(train_x, train_y)
        
    # make predictions on the test data
    y_pred_train = rf_model2.predict(train_x)
    y_pred_test = rf_model2.predict(test_x)

    acc_train = accuracy_score(y_true = train_y, y_pred = y_pred_train)
    acc_test = accuracy_score(y_true = test_y, y_pred = y_pred_test)
    
    print("ACC Train: %s" % acc_train)
    print("ACC Test: %s" % acc_test)
    
    #log parameters
    params ={
        "n_estimators": best_params['n_estimators'],
        "min_samples_split": best_params['min_samples_split'],
        "max_depth": best_params['max_depth'],
        "random_state": 0,
    }
    mlflow.log_params(params)
    #log metrics
    metrics = {
        "acc_train": acc_train,
        "acc_test": acc_test, 
    }
    mlflow.log_metrics(metrics)
    #log model
    mlflow.sklearn.log_model(rf_model2, "Iris - Random Forest Model")
    with open('data/model.pkl','wb') as f:
        pickle.dump(rf_model2,f)

    mlflow.log_artifacts("data/")

    artifacts_uri=mlflow.get_artifact_uri()
    print("The artifact path is",artifacts_uri )
    mlflow.end_run()
    run = mlflow.last_active_run()
    print("Active run id is {}".format(run.info.run_id))
    print("Active run name is {}".format(run.info.run_name))