from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def hyperparameter_search(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid,
    search_type="grid",
    n_iter=10,
    scoring="accuracy",
    cv=5,
    verbose=2,
):
    """
    Perform hyperparameter search using grid search or random search.

    Parameters:
    - model: The machine learning model to tune.
    - X_train: Training feature set.
    - y_train: Training target variable.
    - X_val: Validation feature set.
    - y_val: Validation target variable.
    - param_grid: Dictionary of hyperparameters to search over.
    - search_type: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
    - n_iter: Number of iterations for RandomizedSearchCV (ignored for GridSearchCV).
    - scoring: Scoring metric to use for evaluation.
    - cv: Number of cross-validation folds.
    - verbose: Verbosity level for the search.

    Returns:
    - best_model: The model with the best hyperparameters.
    - best_params: The best hyperparameters found during the search.
    - best_score: The best score achieved with the best hyperparameters.
    - all_results: DataFrame with hyperparameters and corresponding validation scores.
    """
    if search_type == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            return_train_score=True,
            verbose=verbose,
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            random_state=0,
            return_train_score=True,
            verbose=verbose,
        )
    else:
        raise ValueError("search_type must be either 'grid' or 'random'")

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    val_predictions = best_model.predict(X_val)
    val_score = accuracy_score(y_val, val_predictions)

    print(f"Validation Score with best hyperparameters: {val_score}")

    # Collect all results
    results = search.cv_results_
    all_results = pd.DataFrame(results)

    return best_model, best_params, best_score, all_results


def plot_hyperparameter_search_results(
    all_results, param_grid, score_metric="mean_test_score"
):
    fig, axes = plt.subplots(len(param_grid), 1, figsize=(10, 3 * len(param_grid)))
    for ax, (param, values) in zip(axes, param_grid.items()):
        means = all_results.groupby(f"param_{param}")[score_metric].mean()
        ax.hist(x=means.index, weights=means.values, bins=len(values), rwidth=0.8)
        ax.set_title(f"Effect of {param} on {score_metric}")
        ax.set_xlabel(param)
        ax.set_ylabel(score_metric)
    plt.tight_layout()


def log_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    # all columns but 'is_true', 'mutation', and 'Variant_Classification'
    features = df.drop(columns=["is_true", "mutation", "Variant_Classification"])
    # log-transform and normalize the features
    features = features.apply(lambda x: np.log(1 + x))
    features = (features - features.mean()) / features.std()
    # add back the non-numeric columns
    features = pd.concat(
        [features, df[["mutation", "Variant_Classification", "is_true"]]], axis=1
    )
    return features
