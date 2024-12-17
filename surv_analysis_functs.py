import pandas as pd 
import sys 
import numpy as np
import datetime
import json

# For hyperparameter tuning 
from sklearn.model_selection import GridSearchCV

#CPH dependencies 
from sksurv.linear_model import CoxPHSurvivalAnalysis 

#SSVM dependencies 
from sksurv.svm import FastKernelSurvivalSVM

#RSF dependencies 
from sksurv.ensemble import RandomSurvivalForest

#Evaluation dependencies 
from sksurv.metrics import concordance_index_ipcw

def prep_Xy_sksurv(train_set, val_set, regex_filter, surv_headers, do_val=True):
    """
    Changes the survival & radiomic dataframes into X and y matrices with proper format for scikit-surv rsf
    
    Parameters
    -----------
    train_set: pandas.DataFrame
        A DataFrame containing all features, survival time, and the binary event censor for the training fold
    val_set: pandas.DataFrame
        A DataFrame containing all features, survival time, and the binary event censor for the validation fold
    regex_filter: regex string 
        Describes what to filter by to get only features and no survival, clinical, or protocol data
    surv_headers: array of str
        Describes the headers of the censoring and survival time information in the train and val set. 
        Assumes train and val set have consistent headers and that it is in the order Censor, Time
    do_val: bool 
        Whether or not to do the validation if you only want to pass in the training data

    Returns
    -----------
    X_train: pandas.DataFrame
        A DataFrame containing only features from the training set
    y_train: structured np array 
        A structured array compatible with scikit-surv rsf containing the binary event censor and the survival times for the training set
    X_val: pandas.DataFrame
        A DataFrame containing only features from the validation set
    y_val: structured np array
        A structured array compatible with scikit-surv rsf containing the binary event censor and the survival times for the validation set
    """

    #Split the data into X and y matrices
    X_train = train_set.filter(regex = regex_filter, axis = 1)

    #Create y structured arrays compatible with scikit-surv rsf data types
    y_train_arr = train_set[surv_headers].to_numpy()
    y_train_tuples = [(event, surv_time) for event, surv_time in y_train_arr]
    y_train= np.array(y_train_tuples, dtype = [("Event_Occ", "?"), ("Surv_Time", "<f8")])

    if do_val: 
        X_val = val_set.filter(regex = regex_filter, axis = 1)

        y_val_arr = val_set[surv_headers].to_numpy()
        y_val_tuples = [(event, surv_time) for event, surv_time in y_val_arr]
        y_val= np.array(y_val_tuples, dtype = [("Event_Occ", "?"), ("Surv_Time", "<f8")])

        return(X_train, y_train, X_val, y_val)
    
    return(X_train, y_train)

def hyperparam_tuning(X_train, y_train, parameter_grid, num_feats, model_type): 
    """
    Using a stratified k-fold approach, does a grid search of parameters given in the parameter grid and returns the grid search object.
    Uses GridSearchCV class in sklearn

    Parameters
    -----------
    X_train: pandas.DataFrame
        Feature dataframe (after normalization and feature selection)
    y_train: pandas.DataFrame
        Survival data, formatted for the sksurv survival object
    parameter_grid: dict or list of dict
        Has all parameters wanting to be tested, with the names as keys and the parameters to try as values. 
    num_feats: arr of int
        Denotes the number of features to put into the model for each grid search. 
    model_type: sksurv model 
        The type of model to use (e.g. CoxPHSurvivalAnalysis() or RandomSurvivalForest())

    Returns
    ----------
    grid_search_results: pandas.DataFrame
        Object which contains all of the settings and results of the grid search completed
    """
    X_train_copy = X_train.copy(deep = True)

    grid_search_results = pd.DataFrame(columns = ['Number of Features', 'Best Harrell C', 'Best Parameters'])
    for feat_num in num_feats: 
        X_train_curr = X_train_copy.iloc[:, :feat_num]
        grid_search = GridSearchCV(model_type, parameter_grid, cv = 5)
        grid_search.fit(X_train_curr, y_train)

        curr_results = pd.DataFrame({'Number of Features': feat_num, 'Best Harrell C': grid_search.best_score_, 'Best Parameters': [grid_search.best_params_]})
        grid_search_results = pd.concat([grid_search_results, curr_results], axis = 0, ignore_index = True)
        print("Parameter tuning complete for " + str(feat_num) + " features.")
    return(grid_search_results)

def fit_cph(X_train, y_train, a, tie_type, iter_num): 
    """
    Fits a classic CPH model. 
    
    Parameters
    -----------
    X_train: pandas.DataFrame
        Feature dataframe (after normalization and feature selection)
    y_train: pandas.DataFrame
        Survival data, formatted for the sksurv survival object
    a: float
        Regularization praameter for ridge regression penalty (0 = no penalization)
    tie_type: str
        How to handle tied event times. Can take 'breslow' or 'efron'
    iter_num: int
        Max number of iterations
    Returns
    -----------
    cph: sksurv.linear_model
        Cox proportional hazards model object fitted to the training data
    """
    cph = CoxPHSurvivalAnalysis(alpha = a, ties = tie_type, n_iter = iter_num)
    cph.fit(X_train, y_train)

    return(cph) 


def fit_svm_kernel(X_train, y_train, a, optimizer, max_iter, tol, random_state, rank_ratio, kernel_type): 
    """
    Fits a kernel survival support vector machine. 

    Parameters
    -----------
    X_train: pandas.DataFrame
        Feature dataframe (after normalization and feature selection)
    y_train: pandas.DataFrame
        Survival data, formatted for the sksurv survival object
    a: float 
        Alpha parameter that is the weight of penalizing the squared hinge loss in the objective function
    max_iter: int
        Max number of iteration used in Newton optimization 
    tol: float
        Tolerance for termination 
    random_state: int 
    rank_ratio: float 
        Bounded by 0 and 1. 0 --> only regression is performed in the objective function. 1 --> only ranking is performed in objective function.
    kernel_type:  sklearn.metrics.pairwise.pairwise_kernels
        Any pairwise kernel from sklearn is admissible here

    Returns
    ----------
    svm_ker: sksurv.svm
        A kernel SVM object fitted to the training data
    """
    svm_ker = FastKernelSurvivalSVM(alpha = a, kernel = kernel_type, optimizer = optimizer, max_iter = max_iter, tol = tol, random_state = random_state, rank_ratio = rank_ratio)
    svm_ker.fit(X_train, y_train)

    return(svm_ker)

def fit_rsf_model(X_train, y_train, n_est, max_depth, min_samp_split, min_samp_leaf, random_state): 
    """
    Create and train a Random Survival Forest (RSF) with the hyperparameters specified

    Parameters
    -----------
    X_train: pandas.DataFrame
        A DataFrame containing only features from the training set
    y_train: structured np array 
        A structured array compatible with scikit-surv rsf containing the binary event censor and the survival times for the training set
    n_est: int 
        The number of trees in the RSF
    max_depth: int or None
        Maximum depth of the tree. If None, nodes are expanded until leaves are pure or leaves contain less than min_samp_split samples
    min_samp_split: int or float
        Minimum number of samples required to split an internal node. If int, min_samp_split is the minimum number. If float, ciel(min_samp_split * n_samples) is the minimum number
    min_samp_leaf: int or float
        Minimum number of samples required to be at a leaf node. If int, min_samp_leaf is the minimum number. If float, ciel(min_samp_leaf * n_samples) is the minimum number
    random_state: int or None
        Controls the randomness of bootstrapping samples when building trees (if bootstrap = True) and sampling of features to consider when looking for best split at each node (if max_features < n_features)

    Returns
    ----------- 
    rsf: sksurv.ensemble.RandomSurvivalForest
        A RSF fit to the training data provided
    """
    rsf = RandomSurvivalForest(n_estimators = n_est, max_depth = max_depth, min_samples_split = min_samp_split, min_samples_leaf = min_samp_leaf, n_jobs = -1, random_state = random_state)
    rsf.fit(X_train, y_train)

    return(rsf)

def bootstrap_model(train_set, val_set, regex_filter, surv_headers, model, n_iter):
    """ 
    Bootstrap by taking a random sample of training and testing set (n = total sample size) 
    for C-index results (both Harrell's and Uno's C). Assumes that feature normalization
    and feature selection have already been completed.

    Parameters
    ------------
   train_set: pandas.DataFrame
        A DataFrame containing all features, survival time, and the binary event censor for the training fold
    val_set: pandas.DataFrame
        A DataFrame containing all features, survival time, and the binary event censor for the validation fold
    regex_filter: regex string 
        Describes what to filter by to get only features and no survival, clinical, or protocol data
    surv_headers: array of str
        Describes the headers of the censoring and survival time information in the train and val set. 
        Assumes train and val set have consistent headers
    model: sksurv model type 
        Model object fitted to X_train y_train that you'd like to evaluate
    n_iter: int
        The amount of iterations you'd like to run the bootstrapping for. 

    Returns 
    ------------
    bs_train_H: np.array 
        All Harrell's C indices from each sampled train set 
    bs_val_H: np.array
        All Harrell's C indices from each sampled validation set
    bs_train_U: np.array
        All Uno's C indices from each sampled train set 
    bs_val_H: np.array
        All Uno's C indices from each sampled validation set
    perf_stats: pandas.DataFrame 
        Contains the median and 95% confidence interval for both Uno's and Harrell's C
    """
    # Prep to store all C-indices 
    bs_train_H = []
    bs_val_H = []
    bs_train_U = []
    bs_val_U = []

    #Get y variables that the model was trained on to calculate Uno's C
    X_train, y_train, X_val, y_val = prep_Xy_sksurv(train_set, val_set, regex_filter, surv_headers)

    #Bootstrap Harrell's and Uno's C indices for training and testing 
    for x in range(n_iter): 
        train_samp = train_set.sample(train_set.shape[0], replace = True)
        val_samp = val_set.sample(val_set.shape[0], replace = True)

        X_train_samp, y_train_samp, X_val_samp, y_val_samp = prep_Xy_sksurv(train_samp, val_samp, regex_filter, surv_headers)
        
        #Harrell's C
        train_samp_H = model.score(X_train_samp, y_train_samp)
        val_samp_H = model.score(X_val_samp, y_val_samp)

        bs_train_H.append(train_samp_H)
        bs_val_H.append(val_samp_H)

        #Uno's C
        pred_Uno_train = pd.Series(model.predict(X_train_samp))
        Uno_train = concordance_index_ipcw(y_train, y_train_samp, pred_Uno_train)
        Uno_train = Uno_train[0] #Get only the C-index, remove the number of concordant and discordant pairs

        pred_Uno_val = pd.Series(model.predict(X_val_samp))
        Uno_val = concordance_index_ipcw(y_val, y_val_samp, pred_Uno_val)
        Uno_val = Uno_val[0]

        bs_train_U.append(Uno_train)
        bs_val_U.append(Uno_val)

    #Calculate median and confidence intervals for training and testing bootstrapping 
    conf_interval = 0.95 
    lower_percentile = ((1.0 - conf_interval) / 2.0) * 100
    upper_percentile = (conf_interval + ((1.0 - conf_interval) / 2.0)) * 100

    #Harrell's C
    lower_train_H = max(0.0, np.percentile(bs_train_H, lower_percentile)) #2.5th percentile
    upper_train_H = min(1.0, np.percentile(bs_train_H, upper_percentile)) #97.5th percentile
    median_train_H = np.percentile(bs_train_H, 50)

    lower_val_H = max(0.0, np.percentile(bs_val_H, lower_percentile)) #2.5th percentile
    upper_val_H = min(1.0, np.percentile(bs_val_H, upper_percentile)) #97.5th percentile
    median_val_H = np.percentile(bs_val_H, 50)

    #Uno's C 
    lower_train_U = max(0.0, np.percentile(bs_train_U, lower_percentile)) #2.5th percentile
    upper_train_U = min(1.0, np.percentile(bs_train_U, upper_percentile)) #97.5th percentile
    median_train_U = np.percentile(bs_train_U, 50)

    lower_val_U = max(0.0, np.percentile(bs_val_U, lower_percentile)) #2.5th percentile
    upper_val_U = min(1.0, np.percentile(bs_val_U, upper_percentile)) #97.5th percentile
    median_val_U = np.percentile(bs_val_U, 50)

    #Save stats outputs of train and test to dataframe 
    perf_stats_H = pd.DataFrame({"Metric": ["Harrell's C"], 
                                 "Train_2.5p": [lower_train_H], 
                                 "Train_97.5p": [upper_train_H],
                                 "Train_Median": [median_train_H],
                                 "Test_2.5p": [lower_val_H], 
                                 "Test_97.5p": [upper_val_H], 
                                 "Test_Median": [median_val_H]})
    
    perf_stats_U = pd.DataFrame({"Metric": ["Uno's C"], 
                                 "Train_2.5p": [lower_train_U], 
                                 "Train_97.5p": [upper_train_U],
                                 "Train_Median": [median_train_U],
                                 "Test_2.5p": [lower_val_U], 
                                 "Test_97.5p": [upper_val_U], 
                                 "Test_Median": [median_val_U]})
    
    perf_stats = pd.concat([perf_stats_H, perf_stats_U], axis = 0)

    return(bs_train_H, bs_val_H, bs_train_U, bs_val_U, perf_stats)  