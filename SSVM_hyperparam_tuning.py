import pandas as pd 
import sys 
import numpy as np

#CPH Hyperparameter tuning dependencies
from sksurv.svm import FastKernelSurvivalSVM

# Helper functions
from surv_analysis_functs import prep_Xy_sksurv, hyperparam_tuning

### Arguments 
args_dict = dict(arg.split("=") for arg in sys.argv[1:])
mRMR_path = args_dict['train_mRMR_dir']
surv_header = args_dict['surv_head'].split(',') #Assumes censor,time order
regex_filt = args_dict['feat_regex'].replace(',', '|') #Assumes different column values are separated by ","
save_name = args_dict['save_csv_name'] #Does not need .csv extention

### Script 
mRMR_feat_train = pd.read_csv('/global/home/hpc5116/npj_DM/Clin_Runs/Clinical_Data/' + mRMR_path)
if 'Unnamed: 0' in mRMR_feat_train.columns:
    mRMR_feat_train = mRMR_feat_train.drop(columns = ['Unnamed: 0'], axis = 1)

#Prep X and y dataset 
X_train, y_train = prep_Xy_sksurv(train_set = mRMR_feat_train, val_set = pd.DataFrame(), regex_filter = regex_filt, surv_headers = surv_header, do_val = False)

#Hyperparameter tuning
param_grid_ker = {'alpha': (1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8, 1e10), 
              'optimizer': ('avltree', 'rbtree'), 
              'kernel': ('linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine')
              }

num_feats = [5, 
             10, 
             15, 
             20, 
             25]

svm_ker = FastKernelSurvivalSVM()
gs_svm_ker = hyperparam_tuning(X_train, y_train, param_grid_ker, num_feats, model_type = svm_ker)
gs_svm_ker.to_csv(save_name + '.csv')