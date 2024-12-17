import pandas as pd 
import sys 
import numpy as np

# RSF Hyperparameter tuning dependencies
from sksurv.ensemble import RandomSurvivalForest

# Helper functions
from surv_analysis_functs import prep_Xy_sksurv, hyperparam_tuning

### Arguments 
args_dict = dict(arg.split("=") for arg in sys.argv[1:])
mRMR_path = args_dict['train_mRMR_dir']
surv_header = args_dict['surv_head'].split(',') #Assumes censor,time order
regex_filt = args_dict['feat_regex'].replace(',', '|') #Assumes different column values are separated by ","
save_name = args_dict['save_csv_name'] #Does not need .csv extention

### Script 
mRMR_feat_train = pd.read_csv(mRMR_path)
mRMR_feat_train = mRMR_feat_train.drop(columns = ['Unnamed: 0'], axis = 1)

#Prep X and y dataset 
X_train, y_train = prep_Xy_sksurv(train_set = mRMR_feat_train, val_set = pd.DataFrame(), regex_filter = regex_filt, surv_headers = surv_header, do_val = False)

#Hyperparameter tuning
param_grid = {'n_estimators': (250, 500, 750, 1000, 1250), 
              'max_depth': (5, 15, 25, 50), 
              'min_samples_split': (5, 15, 25, 50), 
              'min_samples_leaf': (5, 15, 25, 50)
              }
num_feats = [5,
             10, 
             15, 
             20,
             25
             ]
rsf = RandomSurvivalForest()
gs_rsf = hyperparam_tuning(X_train, y_train, param_grid, num_feats, model_type = rsf)

gs_rsf.to_csv(save_name + '_.csv')
