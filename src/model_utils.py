import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def run_nested_cv(X, y, groups, metadata_df, pipe, param_grid, n_jobs = -2):
    outer_cv = LeaveOneGroupOut()
    inner_cv = LeaveOneGroupOut()

    outer_results = {
            "test_location":[],
            "f1": [],
            "f1_macro":[],
            "accuracy": [],
            "best_n_estimators": [],
            "best_max_depth": [],
            "best_min_samples_leaf":[],
            "train_f1": [],
            "train_f1_macro": [],
            "train_accuracy": []
            }


    search = GridSearchCV(estimator = pipe, # the pipeline goes here, instead of model
                            param_grid = param_grid,
                            scoring = 'f1',
                            cv = inner_cv,
                            n_jobs= n_jobs)
    all_predictions = []

    for train_idx, test_idx in outer_cv.split(X, y, groups = groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        test_location = groups[test_idx][0]
        test_metadata = metadata_df.iloc[test_idx]
        
        
        search.fit(X_train, y_train, groups= groups_train) # note: dont assign search.fit, it modifies in place.

        best_model= search.best_estimator_

        # predict on validatation region

        y_pred = best_model.predict(X_test) # best model is now the pipeline and the transforms will be applied to the test data before (eg scaling)

        # want to also predict on the train regions so can asses overfitting
        train_pred = best_model.predict(X_train)
        train_f1 = f1_score(y_train, train_pred, pos_label= 1, average= "binary")
        train_f1_macro = f1_score(y_train, train_pred, average= "macro")
        train_accuracy = accuracy_score(y_train, train_pred)

        #------------------------------------------------------------
        # calculate and save the results of predicting to thetest region for this fold.``

        fold_f1 = f1_score(y_test, y_pred, pos_label= 1, average= "binary")
        fold_f1macro = f1_score(y_test, y_pred, average= "macro")
        fold_accuracy = accuracy_score(y_test, y_pred)
        fold_best_n_estimators = search.best_params_["model__n_estimators"]
        fold_best_max_depth = search.best_params_["model__max_depth"]
        fold_best_min_samples_leaf = search.best_params_["model__min_samples_leaf"]

        outer_results["test_location"].append(test_location)
        outer_results["f1"].append(fold_f1)
        outer_results["f1_macro"].append(fold_f1macro)
        outer_results["accuracy"].append(fold_accuracy)
        outer_results["best_n_estimators"].append(fold_best_n_estimators)
        outer_results["best_max_depth"].append(fold_best_max_depth)
        outer_results["best_min_samples_leaf"].append(fold_best_min_samples_leaf)
        outer_results["train_f1"].append(train_f1)
        outer_results["train_f1_macro"].append(train_f1_macro)
        outer_results["train_accuracy"].append(train_accuracy)
        
        #------------------------------------------------------------
        # create a dictionary of the arrays for y_hat, y_test, etc
        
        fold_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred,
        }, index = test_idx)

        fold_df = fold_df.join(test_metadata)
        all_predictions.append(fold_df)

    predictions_df= pd.concat(all_predictions)

    results_df = pd.DataFrame(outer_results)
    results_df["f1_gap"] = results_df["train_f1"] - results_df["f1"]
    results_df["f1_macro_gap"] = results_df["train_f1_macro"] - results_df["f1_macro"]
    results_df["accuracy_gap"] = results_df["train_accuracy"] - results_df["accuracy"]

    print(f"F1 Binary:  {np.mean(outer_results['f1']):.3f} +/- {np.std(outer_results['f1']):.3f}")
    print(f"F1 Macro:   {np.mean(outer_results['f1_macro']):.3f} +/- {np.std(outer_results['f1_macro']):.3f}")
    print(f"Accuracy:   {np.mean(outer_results['accuracy']):.3f} +/- {np.std(outer_results['accuracy']):.3f}")

    return results_df, predictions_df