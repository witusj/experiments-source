import time
import json
import numpy as np
import optuna
import xgboost as xgb
from xgboost.callback import TrainingCallback
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from functions import random_combination_with_replacement, create_neighbors_list, calculate_objective
import logging
from typing import Tuple, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomCallback(TrainingCallback):
    def __init__(self, period: int = 10):
        self.period = period

    def after_iteration(self, model, epoch: int, evals_log: Dict[str, Any]) -> bool:
        if (epoch + 1) % self.period == 0:
            logger.info(f"Epoch {epoch}, Evaluation log: {evals_log['validation_0']['logloss'][epoch]}")
        return False

def fit_and_score(estimator, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Tuple[Any, float, float]:
    """Fit the estimator on the train set and score it on both sets."""
    estimator.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    train_score = estimator.score(X_train, y_train)
    test_score = estimator.score(X_test, y_test)
    return estimator, train_score, test_score

def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Objective function for Optuna hyperparameter optimization."""
    param = {
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'early_stopping_rounds': 9,
        'callbacks': [CustomCallback(period=10)]
    }

    clf = xgb.XGBClassifier(**param)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)
    results = []
    
    start = time.time()
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        est, train_score, test_score = fit_and_score(clone(clf), X_train, X_test, y_train, y_test)
        results.append(test_score)
    end = time.time()

    logger.info(f"Training time: {end - start} seconds")
    return np.mean(results)

def create_dataset(N: int, T: int, d: int, s: List[float], q: float, num_schedules: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create dataset for model training."""
    schedules = random_combination_with_replacement(T, N, num_schedules)
    neighbors_list = create_neighbors_list(schedules)

    objectives_schedule_1 = [calculate_objective(neighbor[0], s, d, q)[0] for neighbor in neighbors_list]
    objectives_schedule_2 = [calculate_objective(neighbor[1], s, d, q)[0] for neighbor in neighbors_list]
    objectives = [[obj, objectives_schedule_2[i]] for i, obj in enumerate(objectives_schedule_1)]
    rankings = [0 if obj[0] < obj[1] else 1 for obj in objectives]

    X = np.array([neighbors[0] + neighbors[1] for neighbors in neighbors_list])
    y = np.array(rankings)
    
    return X, y

def save_best_trial_params(study: optuna.Study, filepath: str) -> None:
    """Save the best trial parameters to a JSON file."""
    best_trial_params = study.best_trial.params
    with open(filepath, "w") as f:
        json.dump(best_trial_params, f)
    logger.info(f"Best trial parameters saved to {filepath}")

def load_best_trial_params(filepath: str) -> dict:
    """Load the best trial parameters from a JSON file."""
    with open(filepath, "r") as f:
        best_trial_params = json.load(f)
    return best_trial_params

def main():
    # Create training set
    N, T, d, s, q, num_schedules = 12, 18, 5, [0.0, 0.27, 0.28, 0.2, 0.15, 0.1], 0.20, 20000
    X, y = create_dataset(N, T, d, s, q, num_schedules)

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100)

    # Save the best trial parameters
    save_best_trial_params(study, "best_trial_params.json")

    # Print the best trial results
    logger.info("Best trial:")
    logger.info(f"  Value: {study.best_trial.value}")
    logger.info("  Params: ")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Load the best trial parameters
    best_trial_params = load_best_trial_params("best_trial_params.json")
    logger.info("Loaded best trial parameters:", best_trial_params)

    # Create an XGBoost classifier with the best parameters
    clf = xgb.XGBClassifier(**best_trial_params)
    
    # Split the data into train and test sets for further evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    logger.info(f"Train Score: {train_score}")
    logger.info(f"Test Score: {test_score}")

if __name__ == "__main__":
    main()
