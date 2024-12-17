#General dependencies 
import pandas as pd
import numpy as np 
import sys 
import math
import datetime
import json

from surv_analysis_functs import prep_Xy_sksurv, fit_svm_kernel, bootstrap_model

# Arguments 
args_dict = dict(arg.split("=") for arg in sys.argv[1:])
train_mRMR_dir = args_dict['train_mRMR_dir']
val_dir = args_dict['val_dir'] #Assumed to be normalized using the same method and values as the train set
hyperparam_dir = args_dict['hyperparam_dir'] 
model_type = args_dict['model_type']
regex_filt = args_dict['regex_filter'].replace(',', '|')
surv_headers = args_dict['surv_headers'].split(',') #Assumes censor,time order
clin_end = args_dict['clin_end'] #Clinical endpoint, here is OS or HDFS
feat_type = args_dict['feat_type'] #Radiomic 'rad' or CNN 'cnn'
combat = args_dict['combat'] #Bool indicating whether ComBat was applied to features 

# Load in data 
train_mRMR_df = pd.read_csv(train_mRMR_dir) 
if 'Unnamed: 0' in train_mRMR_df.columns:
    train_mRMR_df = train_mRMR_df.drop(columns = ['Unnamed: 0'], axis = 1)
train_mRMR_df.columns = train_mRMR_df.columns.str.replace(r"[.]", "-", regex = True)

val_df = pd.read_csv(val_dir)
val_df.columns = val_df.columns.str.replace(r"[.]", "-", regex = True)
val_mRMR_df = val_df[train_mRMR_df.columns.values.tolist()] #Assumed file headers have same values as train set 

all_hyperparam_df = pd.read_csv(hyperparam_dir)
curr_hyperparam_df = all_hyperparam_df.loc[(all_hyperparam_df['model_type'] == model_type) & (all_hyperparam_df['clin_end'] == clin_end) & (all_hyperparam_df['feat_type'] == feat_type) & (all_hyperparam_df['combat'] == int(combat))]
curr_hyperparam_df = curr_hyperparam_df.reset_index(drop = True)
hyperparam_vals = json.loads(curr_hyperparam_df['opt_params'][0])

save_name = 'SSVM_Kernel_' + clin_end + '_' + feat_type + '_' + combat

#Prep dataset for training 
X_train, y_train, X_val, y_val = prep_Xy_sksurv(train_mRMR_df, val_mRMR_df, regex_filter = regex_filt, surv_headers = surv_headers)

#Train SSVM model
feature_number = curr_hyperparam_df['opt_feats'][0]
curr_train_data = train_mRMR_df.iloc[:, :feature_number+2] #Adding to to include the censor and survival time, assumes that the survival data comes in the first two columns of each after preprocessing
curr_val_data = val_mRMR_df.iloc[:, :feature_number+2]
curr_X_train = X_train.iloc[:, :feature_number]

svm_ker = fit_svm_kernel(X_train = curr_X_train, y_train = y_train, a = hyperparam_vals['alpha'], kernel_type = hyperparam_vals['kernel'], optimizer = hyperparam_vals['optimizer'], random_state = 42, tol = None, rank_ratio = 1, max_iter = 100)

#Bootstrap results
bs_train_Har, bs_val_Har, bs_train_Uno, bs_val_Uno, perf_stats_os = bootstrap_model(train_set = curr_train_data, 
                                                                        val_set = curr_val_data, 
                                                                        regex_filter = regex_filt, 
                                                                        surv_headers = surv_headers, 
                                                                        model = svm_ker, n_iter = 500)

#Save information 
np.save('Bootstrap_Results/' + datetime.date.today().strftime('%Y-%m-%d') + '_' + save_name + '_Train_HarC.npy', bs_train_Har)
np.save('Bootstrap_Results/' + datetime.date.today().strftime('%Y-%m-%d') + '_' + save_name + '_Val_HarC.npy', bs_val_Har)

np.save('Bootstrap_Results/' + datetime.date.today().strftime('%Y-%m-%d') + '_' + save_name + '_Train_UnoC.npy', bs_train_Uno)
np.save('Bootstrap_Results/' + datetime.date.today().strftime('%Y-%m-%d') + '_' + save_name + '_Val_UnoC.npy', bs_val_Uno)

perf_stats_os.to_csv('Bootstrap_Results/' + datetime.date.today().strftime('%Y-%m-%d') + '_' + save_name + '_C_indices.csv')




