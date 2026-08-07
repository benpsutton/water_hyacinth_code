import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import numpy as np
import torch
import kornia.augmentation as k
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def run_nested_cv(X, y, groups, metadata_df, pipe, param_grid, n_jobs = -2):
    outer_cv = LeaveOneGroupOut()
    inner_cv = LeaveOneGroupOut()

    outer_results = {
            "test_location":[],
            "test_f1": [],
            "test_f1_macro":[],
            "test_accuracy": [],
            "test_precision": [],
            "test_recall": [],
            "test_auroc": [],
            "best_n_estimators": [],
            "best_max_depth": [],
            "best_min_samples_leaf":[],
            "train_f1": [],
            "train_f1_macro": [],
            "train_accuracy": []
            }


    # multi-metric scoring: record several metrics for the inner folds, but still
    # select the best model on binary f1 via refit.
    scoring = {
        "f1": "f1",               # binary f1, pos_label=1
        "accuracy": "accuracy",
        "precision": "precision", # binary, pos_label=1
        "recall": "recall",       # binary, pos_label=1
        "auroc": "roc_auc",
    }

    search = GridSearchCV(estimator = pipe, # the pipeline goes here, instead of model
                            param_grid = param_grid,
                            scoring = scoring,
                            refit = "f1", # best model still chosen on binary f1 only
                            cv = inner_cv,
                            n_jobs= n_jobs)
    all_predictions = []
    inner_records = [] # per inner-fold metrics for the winning config, across outer folds

    for train_idx, test_idx in outer_cv.split(X, y, groups = groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        test_location = groups[test_idx][0]
        test_metadata = metadata_df.iloc[test_idx]


        search.fit(X_train, y_train, groups= groups_train) # note: dont assign search.fit, it modifies in place.

        best_model= search.best_estimator_

        #------------------------------------------------------------
        # capture the inner-fold metrics for the winning hyperparameter config.
        # best_index_ is the row in cv_results_ selected by refit="f1".
        best_i = search.best_index_
        cvr = search.cv_results_

        # reconstruct which val region belongs to each split, in the same order
        # GridSearchCV iterated them (rather than assuming sorted-unique order).
        inner_val_regions = [groups_train[val_idx][0]
                             for _, val_idx in inner_cv.split(X_train, y_train, groups_train)]

        for split_no, val_region in enumerate(inner_val_regions):
            inner_records.append({
                "outer_test_location": test_location,
                "inner_val_region": val_region,
                "f1": cvr[f"split{split_no}_test_f1"][best_i],
                "accuracy": cvr[f"split{split_no}_test_accuracy"][best_i],
                "precision": cvr[f"split{split_no}_test_precision"][best_i],
                "recall": cvr[f"split{split_no}_test_recall"][best_i],
                "auroc": cvr[f"split{split_no}_test_auroc"][best_i],
                "best_n_estimators": search.best_params_["model__n_estimators"],
                "best_max_depth": search.best_params_["model__max_depth"],
                "best_min_samples_leaf": search.best_params_["model__min_samples_leaf"],
            })

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
        fold_recall = recall_score(y_test, y_pred)
        fold_precision = precision_score(y_test, y_pred)
        fold_auroc = roc_auc_score(y_test, y_pred)
        fold_best_n_estimators = search.best_params_["model__n_estimators"]
        fold_best_max_depth = search.best_params_["model__max_depth"]
        fold_best_min_samples_leaf = search.best_params_["model__min_samples_leaf"]

        outer_results["test_location"].append(test_location)
        outer_results["test_f1"].append(fold_f1)
        outer_results["test_f1_macro"].append(fold_f1macro)
        outer_results["test_accuracy"].append(fold_accuracy)
        outer_results["test_recall"].append(fold_recall)
        outer_results["test_precision"].append(fold_precision)
        outer_results["test_auroc"].append(fold_auroc)
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
    results_df["f1_gap"] = results_df["train_f1"] - results_df["test_f1"]
    results_df["f1_macro_gap"] = results_df["train_f1_macro"] - results_df["test_f1_macro"]
    results_df["accuracy_gap"] = results_df["train_accuracy"] - results_df["test_accuracy"]

    inner_df = pd.DataFrame(inner_records)

    print(f"F1 Binary:  {np.mean(outer_results['test_f1']):.3f} +/- {np.std(outer_results['test_f1']):.3f}")
    print(f"F1 Macro:   {np.mean(outer_results['test_f1_macro']):.3f} +/- {np.std(outer_results['test_f1_macro']):.3f}")
    print(f"Accuracy:   {np.mean(outer_results['test_accuracy']):.3f} +/- {np.std(outer_results['test_accuracy']):.3f}")

    return results_df, predictions_df, inner_df

def load_patch_dict_as_tensor(Patch_dict, band_list = None):

    """
    Takes a supplied dictionary of pathes with their labels, converts patches and labels to a tensor, 
    extracts the class integer labels, locations and label_id to arrays
    """ 

    if not band_list:
        band_list = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    feature_list = Patch_dict["features"]

    patches = np.stack([
        np.stack([np.array(F["properties"][band]) for band in band_list]) # (N, C, H, W) which is what pytorch expects
        for F in feature_list  
    ])

    labels = torch.tensor([F["properties"]["lc"] for F in feature_list]).float().unsqueeze(1) # BCElogits loss will need to compare labels to float logits or probabilties
    # .unsqueeze(1) changes the shape from ([0,1,1,0]) to ([0],[1],[1],[0]) aka (B,) to (B, 1), need this for the criterion. 

    locations = np.array([F["properties"]["location"] for F in feature_list]) # array for easier boolean masking but no need for tensor

    class_int = np.array([F["properties"]["class_int"] for F in feature_list])
    
    label_id = np.array([F["properties"]["label_id"] for F in feature_list])

    patches = torch.from_numpy(patches).float()

    print(f"Returned patch tensor shape: {patches.shape}")
    print(f"Returned label tensor shape: {labels.shape}")
    print(f"Returned locations array: {locations.shape}")
    print(f"Returned class_int array: {class_int.shape}")
    print(f"Returned label_id array: {label_id.shape}")
    print(f"Bands used: {band_list}")

    return patches, labels, locations, class_int, label_id

def buildCNN_2xVGG(input_channels= 10, dropout = 0.3):

        myCNN = nn.Sequential(
        nn.Conv2d(in_channels=input_channels, 
                out_channels = 32, 
                kernel_size=3,
                stride= 1,
                padding=1  ),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, 
                out_channels = 32, 
                kernel_size=3,
                stride= 1,
                padding=1  ),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=32, 
                out_channels = 64, 
                kernel_size=3,
                stride= 1,
                padding=1  ),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, 
                out_channels = 64, 
                kernel_size=3,
                stride= 1,
                padding=1  ),
        nn.BatchNorm2d(64),
        
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(64,1))

        return myCNN

def build_CNN_dense(patch_size, input_channels = 10, dropout = 0.3):

    layers = [
         nn.Conv2d(in_channels=input_channels, 
                out_channels = 32, 
                kernel_size=3,
                stride= 1,
                padding=1  ),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, 
                out_channels = 32, 
                kernel_size=3,
                stride= 1,
                padding=1  ),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=32, 
                out_channels = 64, 
                kernel_size=3,
                stride= 1,
                padding=1  ),
        nn.BatchNorm2d(64),
        nn.ReLU()
    ]
    if patch_size == 31:
        layers += [
            nn.MaxPool2d(kernel_size=2)
            ]

    layers += [
        nn.Conv2d(in_channels=64, 
                out_channels = 64, 
                kernel_size=3,
                stride= 1,
                padding=1  ),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Dropout(p = dropout),
        nn.Linear(in_features= 576, out_features=64),
        nn.ReLU(),
        nn.Linear(64,1)
    ]    
    return nn.Sequential(*layers)

def train_for_one_epoch(model, train_loader, device, aug, optimizer, metrics, criterion):
    
    model.train()
    running_loss = 0.0
    metrics.reset()
    
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        batch_X = aug(batch_X) # Augment only in the train split

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        metrics.update(logits, batch_y) # accumulates the metrics per batch. Torchmetrics detects logits and applies sigmoid internally
        running_loss += loss.item()*batch_X.size(0)
    avg_loss = running_loss / len(train_loader.dataset) 

    results = {k: v.item() for k, v in metrics.compute().items()}   

    return avg_loss, results

    
def val_for_one_epoch(model, val_loader, metrics, device, criterion):
    
    model.eval()
    running_val_loss = 0.0
    metrics.reset() # clears the metrics from previous epochs
    

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)               # Outputting logits as useing BCEntropy with logit loss as criterion
            val_loss = criterion(logits, batch_y)
            
            #probs = torch.sigmoid(logits) # gives probabilities between 0 and 1
            #preds = (logits >= 0).int() #because sigmoid(0) = 0.5 and would use as probability threshold anyway
            metrics.update(logits, batch_y) # accumulates the metrics per batch. Torchmetrics detects logits and applies sigmoid internally

            running_val_loss += val_loss.item()*batch_X.size(0)
           

        avg_loss = running_val_loss / len(val_loader.dataset)
        results = {k: v.item() for k, v in metrics.compute().items()}
        
    return avg_loss, results

def predict_on_test_region(model, test_loader, metrics, device, criterion):
    
    # no per epoch as not training, testing the model once on the held out region
    model.eval()
    metrics.reset()
    running_test_loss = 0.0
    preds_tensor = torch.empty((0,1))
    logits_tensor = torch.empty((0,1))

    with torch.no_grad():
        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_X)
            logits_tensor = torch.cat([logits_tensor, logits.cpu()], dim = 0)

            test_loss = criterion(logits, batch_y)

            #probs = torch.sigmoid(logits) # gives probabilities between 0 and 1
            preds = (logits >= 0).cpu().int() #because sigmoid(0) = 0.5 and would use as probability threshold anyway
            preds_tensor = torch.cat([preds_tensor, preds], dim= 0) # concat the preds tensors for each batch

            metrics.update(logits, batch_y)
            running_test_loss += test_loss.item()*batch_X.size(0)
    
    average_loss = running_test_loss / len(test_loader.dataset)

    results = {k: v.item() for k, v in metrics.compute().items()}

    return average_loss, results, preds_tensor, logits_tensor

class PatchDataset(Dataset):
    """
    Standardises patches based on mean and std
    """

    def __init__(self, patches, labels, mean, std):
        
        mean = mean.detach().clone().view(1,-1,1,1) # changes from (C,) to (1,C,1,1)
        std = std.detach().clone().view(1,-1,1,1)
    
        self.patches = (patches - mean)/std 
        self.labels = labels 

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]
    

def load_dataset(patches, labels, mean, std, batch_size, shuffle = False):
        
    """ 
    Creates instance of PatchDataset class, returns loader with batched of specifed size. 
    Set shuffle= True for training patches"""

    dataset = PatchDataset(patches = patches, labels = labels, mean = mean, std = std)

    loader = DataLoader(
        dataset= dataset,
        shuffle= shuffle,
        batch_size= batch_size,
        num_workers= 0,
        pin_memory= True
        )
        
    return loader


        
def record_epoch(history_dict, run_ID, patch_size,  loop, epoch, test_region, train_metrics,train_loss, AWEI, best_epoch = None, val_region=None , smoothed_val_loss = None, k= None, val_loss= None,  val_metrics= None):
            
    """ Append the values and metrics for each epoch to a history_dict.
    History dicts for each run can then be conctenated inot a long format for plotting"""

    history_dict["run_ID"].append(run_ID)
    history_dict["patch_size"].append(patch_size)
    history_dict["test_region"].append(test_region)
    history_dict["AWEIp95"].append(AWEI)

    history_dict["loop"].append(loop)
    history_dict["epoch"].append(epoch+1)
    history_dict["train_loss"].append(train_loss)
    history_dict["train_accuracy"].append(train_metrics["accuracy"])
    history_dict["train_f1"].append(train_metrics["f1"])
    history_dict["train_precision"].append(train_metrics["precision"])
    history_dict["train_recall"].append(train_metrics["recall"])
    history_dict["train_auroc"].append(train_metrics["auroc"])
    if val_metrics is not None: # outer folds won't have a val_region
        history_dict["val_region"].append(val_region)
        history_dict["smoothed_val_loss"].append(smoothed_val_loss)
        history_dict["k"].append(k)
        history_dict["val_loss"].append(val_loss)
        history_dict["val_accuracy"].append(val_metrics["accuracy"])
        history_dict["val_f1"].append(val_metrics["f1"])
        history_dict["val_precision"].append(val_metrics["precision"])
        history_dict["val_recall"].append(val_metrics["recall"])
        history_dict["val_auroc"].append(val_metrics["auroc"])
        history_dict["best_epoch"].append(best_epoch)

def record_inner_loop_best_state(inner_best_state_dict, run_ID, patch_size, AWEI, best_epoch, test_region, val_region, train_loss, best_smoothed_val_loss, smoothing_window_k, best_val_loss, best_val_metrics, best_train_metrics, lr, dropout, weight_decay, batch_size): 
   
    """Save the metrics for the best epoch for each inner fold after early stopping"""

    inner_best_state_dict["run_ID"].append(run_ID)
    inner_best_state_dict["patch_size"].append(patch_size)
    inner_best_state_dict["epoch_stopped"].append(best_epoch)
    inner_best_state_dict["test_region"].append(test_region)
    inner_best_state_dict["val_region"].append(val_region)
    inner_best_state_dict["AWEIp95"].append(AWEI)
    inner_best_state_dict["train_loss"].append(train_loss)
    inner_best_state_dict["smoothed_val_loss"].append(best_smoothed_val_loss)
    inner_best_state_dict["k"].append(smoothing_window_k)
    inner_best_state_dict["val_loss"].append(best_val_loss)
    inner_best_state_dict["val_accuracy"].append(best_val_metrics["accuracy"])
    inner_best_state_dict["val_f1"].append(best_val_metrics["f1"])
    inner_best_state_dict["val_precision"].append(best_val_metrics["precision"])
    inner_best_state_dict["val_recall"].append(best_val_metrics["recall"])
    inner_best_state_dict["val_auroc"].append(best_val_metrics["auroc"])
    inner_best_state_dict["train_accuracy"].append(best_train_metrics["accuracy"])
    inner_best_state_dict["train_f1"].append(best_train_metrics["f1"])
    inner_best_state_dict["train_precision"].append(best_train_metrics["precision"])
    inner_best_state_dict["train_recall"].append(best_train_metrics["recall"])
    inner_best_state_dict["train_auroc"].append(best_train_metrics["auroc"])
    inner_best_state_dict["dropout"].append(dropout)
    inner_best_state_dict["lr"].append(lr)
    inner_best_state_dict["weight_decay"].append(weight_decay)
    inner_best_state_dict["batch_size"].append(batch_size)
    
def record_outer_loop_metrics(outer_metrics_dict, run_ID, patch_size, test_region, AWEI, median_epoch, smoothing_window_k, test_loss, train_loss, test_metrics, train_metrics,lr, dropout, weight_decay, batch_size):

    """ Save the metrics for each fold in the outer loop afer testing"""

    outer_metrics_dict["run_ID"].append(run_ID)
    outer_metrics_dict["patch_size"].append(patch_size)
    outer_metrics_dict["test_region"].append(test_region)
    outer_metrics_dict["AWEIp95"].append(AWEI)
    outer_metrics_dict["epochs_trained"].append(median_epoch)
    outer_metrics_dict["k"].append(smoothing_window_k)
    outer_metrics_dict["dropout"].append(dropout)
    outer_metrics_dict["lr"].append(lr)
    outer_metrics_dict["weight_decay"].append(weight_decay)
    outer_metrics_dict["batch_size"].append(batch_size)

    # Metrics for the prediction on held out test_region
    outer_metrics_dict["test_loss"].append(test_loss)
    outer_metrics_dict["test_accuracy"].append(test_metrics["accuracy"])
    outer_metrics_dict["test_f1"].append(test_metrics["f1"])
    outer_metrics_dict["test_precision"].append(test_metrics["precision"])
    outer_metrics_dict["test_recall"].append(test_metrics["recall"])
    outer_metrics_dict["test_auroc"].append(test_metrics["auroc"])

    # These are the train matrics from the final epoch
    outer_metrics_dict["train_loss"].append(train_loss)
    outer_metrics_dict["train_accuracy"].append(train_metrics["accuracy"])
    outer_metrics_dict["train_f1"].append(train_metrics["f1"])
    outer_metrics_dict["train_precision"].append(train_metrics["precision"])
    outer_metrics_dict["train_recall"].append(train_metrics["recall"])
    outer_metrics_dict["train_auroc"].append(train_metrics["auroc"])

