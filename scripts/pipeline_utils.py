#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, learning_curve
from tensorflow.keras.callbacks import EarlyStopping

metrics_scores = { 
    "accuracy": accuracy_score, 
    "f1": f1_score, 
    "precision": precision_score, 
    "recall": recall_score, 
    "roc_auc": roc_auc_score 
    } 

pipeline_metrics = ["accuracy", "roc_auc", "recall", "precision"]

def get_pipeline_results(pipeline, name, splits, X_train, y_train, X_test, y_test, fold_names, metrics = pipeline_metrics, pipeline_type = "logreg"):
    ''' 
    Get scores for pipeline: for each metric in metrics followng values are calculated: CV in folds, CV mean, CV std, 
    metric on the train set and metric on the test set
    '''
    results = {"name": name, "scores": {}} 
    results["scores in folds"] = {f"{m.capitalize()} CV in folds": {} for m in metrics}
    for i, (train_idx, val_idx) in enumerate(splits):
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx] 
        X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx] 

        if pipeline_type == "logreg":
            pipeline.fit(X_train_fold, y_train_fold)
        
        elif pipeline_type == "nn":
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            pipeline.fit(X_train_fold, y_train_fold,
                         nn__callbacks=[early_stopping],
                         nn__validation_data=(X_val_fold, y_val_fold))
            
        y_proba_val = pipeline.predict_proba(X_val_fold)[:, 1]
        y_pred_val = pipeline.predict(X_val_fold)

        #cv metric in folds 
        for m in metrics: 
            #fold_scores = [] 
            if m == "roc_auc":     
                fold_score = metrics_scores[m](y_val_fold, y_proba_val) 
            else:      
                fold_score = metrics_scores[m](y_val_fold, y_pred_val)
            #fold_scores.append(fold_score) 
            results["scores in folds"][f"{m.capitalize()} CV in folds"][fold_names[i]] = fold_score 

    if pipeline_type == "logreg":
        pipeline.fit(X_train, y_train)
        results["coef"] = {}
        results["coef"] = dict(zip(X_train.columns.tolist(), pipeline.named_steps["logreg"].coef_[0]))
    
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    y_proba_train = pipeline.predict_proba(X_train)[:, 1]
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    
    # scores = {"train":{}, "test":{}} # save scores for better ordering 
    # loop through metrics first time when the pipeline is fit to full train set
    for m in metrics:
        # cv metric mean and std 
        results["scores"][f"{m.capitalize()} CV mean"] = np.mean([v for k, v in results["scores in folds"][f"{m.capitalize()} CV in folds"].items()]) 
        results["scores"][f"{m.capitalize()} CV std"] = np.std([v for k, v in results["scores in folds"][f"{m.capitalize()} CV in folds"].items()])
        # metric on train and test 
        if m == "roc_auc": 
            results["scores"][f"{m.capitalize()} train"] = metrics_scores[m](y_train, y_proba_train) 
            results["scores"][f"{m.capitalize()} test"] = metrics_scores[m](y_test, y_proba_test) 
        else: 
            results["scores"][f"{m.capitalize()} train"] = metrics_scores[m](y_train, y_pred_train) 
            results["scores"][f"{m.capitalize()} test"] = metrics_scores[m](y_test, y_pred_test)
    
    return results
        
def show_pipeline_summary(results, title = "Pipeline summary"): 
    ''' 
    Shows summary of scores for a list of pipelines results without scores for individual folds 
    ''' 
    results_summary = [ 
        {"name":  res["name"], **res["scores"]}
        for res in results 
        ] 
    df = pd.DataFrame(results_summary) 
    display(df.style.set_caption(title).hide(axis = "index").format(precision = 3 )) 

def show_pipeline_folds(results, metrics = pipeline_metrics, plot = True, show_tables = True): 
    ''' 
    Plot scores and/or show them as a table for individual folds for a list of pipeline results for a list of metrics
    ''' 
    results_folds = []
    for m in metrics:
        results_folds_metric = [
            {"name": res["name"], **v}
            for res in results
            for k, v in res["scores in folds"].items() if m.capitalize() in k]
        results_folds.append(pd.DataFrame(results_folds_metric).set_index("name"))
    if show_tables:
        for m, df in zip(metrics, results_folds):
            display(df.T.style.set_caption(f"{m.capitalize()} in CV folds").format())
    if plot:
        for m, df in zip(metrics, results_folds):
            df.T.plot(figsize=(15, 5), title = f"{m.capitalize()} in CV folds", xlabel = "Folds", ylabel = m.capitalize()).legend(bbox_to_anchor=(1.0, 1.0), fontsize='small')

def show_pipeline_coef(results):
    '''
    Show feature coefficients for a list of pipelines and color by value
    '''
    results_coef = [
        {"name": res["name"], **res["coef"]}
        for res in results
    ]
    df = pd.DataFrame(results_coef).fillna(0).set_index("name")
    display(df.T.style
            .background_gradient(cmap="coolwarm",
                                 axis = 0,
                                 vmin = -1.5,
                                 vmax = 1.5)
            .set_caption("Coefficients for pipelines")
            .set_table_styles([{'selector':'th',
                            'props':[('word-wrap', ' break-word'),
                                     ('width','100px'),
                                     ( 'text-align', 'center')
                                    ]
                           }])
            .format(precision = 3 )) 

def plot_learning_curve(pipeline, X_train, y_train, stratification, scoring = "roc_auc", ylabel = "ROC AUC", title = "Learning Curve"):
    '''
    Plot learning curve for diagnostics. StratifiedKFold used for data splitting
    '''
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv = skf.split(X_train, stratification)

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline,
        X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,
        random_state=42
    )

    # Mean and std
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)

    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_mean, label="Training score", marker="o")
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
    plt.plot(train_sizes, val_mean, label="Validation score", marker="o")
    plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2)

    plt.xlabel("Training set size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
