#General dependencies 
import pandas as pd
import numpy as np 
import os
import json
import argparse

#Hyperparameter tuning dependencies 
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical 

#CPH dependencies 
from sksurv.linear_model import CoxPHSurvivalAnalysis 

#RSF dependencies 
from sksurv.ensemble import RandomSurvivalForest

#SSVM dependencies 
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM

#Evaluation dependencies 
from sksurv.metrics import concordance_index_ipcw

class SurvivalModel: 
    def __init__(self, root_path, model_type, surv_type, feature_type, combat, feat_filt): 
        self.root_path = root_path 
        self.model_type = model_type
        self.surv_type = surv_type
        self.combat = combat
        self.feature_type = feature_type
        self.feat_filt = feat_filt
        self.clin_train_path = (os.path.join(root_path, 'clin_train')).replace('\\', "/")
        self.clin_test_path = (os.path.join(root_path, 'clin_test')).replace("\\", "/")
        self.rad_train_path = (os.path.join(root_path, 'rad_mRMR_train')).replace("\\", "/")
        self.rad_test_path = (os.path.join(root_path, 'rad_test')).replace("\\", "/")
        self.ref_files_path = (os.path.join(root_path, 'ref_files')).replace("\\", "/") #To add patient ID back to rad training files
        self.hyperparam_path = (os.path.join(root_path, 'hyperparams')).replace("\\", "/") #Should either be empty to start or have only one file after hp tuning
        self.train_risks = None
        self.preprocess()
        self.get_train_test()

    def preprocess(self):
        # Check if there is no hyperparameter file, make a blank one with the correct column headers if necessary
        if len(os.listdir(self.hyperparam_path)) == 0: 
            empty_hp_df = pd.DataFrame(columns = ['clin_end', 'feat_type', 'combat', 'model_type', 'opt_params', 'best_Harrell_C'])
            self.hyperparams = empty_hp_df
        else: 
            self.hyperparams = pd.read_csv((os.path.join(self.hyperparam_path, os.listdir(self.hyperparam_path)[0])).replace("\\", "/"))

        # Load in clinical train and test sets, round survival times, and fix patient header
        self.clin_training = dict()
        self.clin_testing = dict()

        for file in os.listdir(self.clin_train_path):
            clin_train = pd.read_csv(os.path.join(self.clin_train_path, file))
            if 'OSTime' in clin_train.columns.values.tolist():
                clin_train['OSTime'] = clin_train['OSTime'].round(decimals = 0)
            elif 'HDFSTime' in clin_train.columns.values.tolist():
                clin_train['HDFSTime'] = clin_train['HDFSTime'].round(decimals = 0)
            clin_train = clin_train.rename(columns={'XNAT ID': 'Patient'})
            self.clin_training[file] = clin_train

        for file in os.listdir(self.clin_test_path):
            clin_test = pd.read_csv(os.path.join(self.clin_test_path, file))
            if 'OSTime' in clin_test.columns.values.tolist():
                clin_test['OSTime'] = clin_test['OSTime'].round(decimals = 0)
            elif 'HDFSTime' in clin_test.columns.values.tolist(): 
                clin_test['HDFSTime'] = clin_test['HDFSTime'].round(decimals = 0)
            clin_test = clin_test.rename(columns={'XNAT ID': 'Patient'})
            self.clin_testing[file] = clin_test
        
        # Load in radiomic train and test sets, round survival times, add patient ID (train only),
        # and get intersection of radiomic and clinical patients
        self.rad_training = dict()
        self.rad_testing = dict()
        self.os_ref = None
        self.hdfs_ref = None
        for ref_name in os.listdir(self.ref_files_path): 
            if 'OS' in ref_name: 
                self.os_ref = pd.read_csv(os.path.join(self.ref_files_path, ref_name))
            elif 'HDFS' in ref_name: 
                self.hdfs_ref = pd.read_csv(os.path.join(self.ref_files_path, ref_name))

        for file in os.listdir(self.rad_train_path):
            rad_train = pd.read_csv(os.path.join(self.rad_train_path, file))
            if 'OSTime' in rad_train.columns.values.tolist():
                clin_os_train = [value for key, value in self.clin_training.items() if 'OS' in key][0]
                rad_train['OSTime'] = rad_train['OSTime'].round(decimals = 0)
                rad_train['Patient'] = self.os_ref['Patient']
                rad_train = rad_train[rad_train['Patient'].isin(clin_os_train['Patient'])] #Assumes clinical patients subset of radiomic patients
                rad_train.columns = rad_train.columns.str.replace('[.]', '_', regex=True)
                self.rad_training[file] = rad_train.reset_index(drop = True)
            elif 'HDFSTime' in rad_train.columns.values.tolist(): 
                clin_hdfs_ref = [value for key, value in self.clin_training.items() if 'HDFS' in key][0]
                rad_train['HDFSTime'] = rad_train['HDFSTime'].round(decimals = 0)
                rad_train['Patient'] = self.hdfs_ref['Patient']
                rad_train = rad_train[rad_train['Patient'].isin(clin_hdfs_ref['Patient'])] #Same note as above
                rad_train.columns = rad_train.columns.str.replace('[.]', '_', regex=True)
                self.rad_training[file] = rad_train.reset_index(drop = True)
        
        for file in os.listdir(self.rad_test_path):
            rad_test = pd.read_csv(os.path.join(self.rad_test_path, file))
            if 'OSTime' in rad_test.columns.values.tolist():
                clin_os_test = [value for key, value in self.clin_testing.items() if 'OS' in key][0]
                rad_test['OSTime'] = rad_test['OSTime'].round(decimals = 0)
                rad_test = rad_test[rad_test['Patient'].isin(clin_os_test['Patient'])]
                rad_test.columns = rad_test.columns.str.replace('[.]', '_', regex = True)
                rad_test.columns = rad_test.columns.str.replace('[-]', '_', regex = True)
                self.rad_testing[file] = rad_test.reset_index(drop = True)
            elif 'HDFSTime' in rad_test.columns.values.tolist():
                clin_hdfs_test = [value for key, value in self.clin_testing.items() if 'HDFS' in key][0]
                rad_test['HDFSTime'] = rad_test['HDFSTime'].round(decimals = 0) 
                rad_test = rad_test[rad_test['Patient'].isin(clin_hdfs_test['Patient'])]
                rad_test.columns = rad_test.columns.str.replace('[.]', '_', regex=True)
                rad_test.columns = rad_test.columns.str.replace('[-]', '_', regex = True)
                self.rad_testing[file] = rad_test.reset_index(drop = True)
    
    def prep_Xy(self, data): 
        X = data.filter(regex = self.feat_filt, axis = 1)

        surv_headers = [self.surv_type + 'Cens', self.surv_type + 'Time']
        y_arr = data[surv_headers].to_numpy()
        y_tuples = [(event, surv_time) for event, surv_time in y_arr]
        y = np.array(y_tuples, dtype = [("Event_Occ", "?"), ("Surv_Time", "<f8")])

        return(X, y)
    
    def get_train_test(self):
        if self.feature_type == 'clin': 
            self.train_data = [value for key, value in self.clin_training.items() if self.surv_type in key][0]
            self.test_data = [value for key, value in self.clin_testing.items() if self.surv_type in key][0]

        elif self.feature_type == 'rad' or self.feature_type == 'cnn': 
            # Get data from radiomics dictionary that has all of the identified types in its filename
            self.train_data = [value for key, value in self.rad_training.items() if all(types in key for types in [self.surv_type, self.feature_type, self.combat])][0]
            self.test_data = [value for key, value in self.rad_testing.items() if all(types in key for types in [self.surv_type, self.feature_type, self.combat])][0]

        elif self.feature_type == 'risk': 
            # Use train and test data from clinical variables to get train and test splits 
            self.train_data_init = [value for key, value in self.clin_training.items() if self.surv_type in key][0]
            self.test_data_init = [value for key, value in self.clin_testing.items() if self.surv_type in key][0]
            self.train_data = None
            self.test_data = None

            return 0
        
        else: 
            self.train_data = None
            self.test_data = None
            
            return 0
        
        self.X_train, self.y_train = self.prep_Xy(self.train_data)
        self.X_test, self.y_test = self.prep_Xy(self.test_data)
        self.X_test = self.X_test[self.X_train.columns.values.tolist()]

    def base_risks(self): 
        self.train_preds = self.model.predict(self.X_train)
        self.train_risks = pd.DataFrame({self.model_type + '_' + self.feature_type + '_' + str(self.combat) + '_Pred': self.train_preds})
        self.train_risks[self.surv_type + 'Cens'] = self.train_data[self.surv_type + 'Cens']
        self.train_risks[self.surv_type + 'Time'] = self.train_data[self.surv_type + 'Time']
        self.train_risks['Patient'] = self.train_data['Patient']

        self.test_preds = self.model.predict(self.X_test)
        self.test_risks = pd.DataFrame({self.model_type + '_' + self.feature_type + '_' + str(self.combat) + '_Pred': self.test_preds})
        self.test_risks[self.surv_type + 'Cens'] = self.test_data[self.surv_type + 'Cens']
        self.test_risks[self.surv_type + 'Time'] = self.test_data[self.surv_type + 'Time']
        self.test_risks['Patient'] = self.test_data['Patient']

    def bootstrap(self, iter_num):
        self.train_samp_ids = []
        self.test_samp_ids = []
        self.bs_train_H = []
        self.bs_train_U = []
        self.bs_test_H = []
        self.bs_test_U = []

        for i in range(iter_num):
            train_samp = self.train_data.sample(self.train_data.shape[0], replace = True)
            test_samp = self.test_data.sample(self.test_data.shape[0], replace = True)

            self.train_samp_ids.append(train_samp['Patient'].to_numpy())
            self.test_samp_ids.append(test_samp['Patient'].to_numpy())

            X_train_samp, y_train_samp = self.prep_Xy(train_samp)
            X_test_samp, y_test_samp = self.prep_Xy(test_samp)
            X_test_samp = X_test_samp[X_train_samp.columns.values.tolist()]

            # Calculate Harrell's C 
            self.bs_train_H.append(self.model.score(X_train_samp, y_train_samp))
            self.bs_test_H.append(self.model.score(X_test_samp, y_test_samp))

            # Calculate Uno's C 
            pred_train_U = pd.Series(self.model.predict(X_train_samp))
            self.bs_train_U.append(concordance_index_ipcw(self.y_train, y_train_samp, pred_train_U)[0])
            
            pred_test_U = pd.Series(self.model.predict(X_test_samp))
            self.bs_test_U.append(concordance_index_ipcw(self.y_test, y_test_samp, pred_test_U)[0])

    def calc_conf(self, conf_interval): 
        # 0 < conf_interval < 1
        self.lower_perc = ((1.0 - conf_interval) / 2.0) * 100
        self.upper_perc = (conf_interval + ((1.0 - conf_interval) / 2.0)) * 100

        def do_calc(self, perf_info): 
            lower_conf = max(0.0, np.percentile(perf_info, self.lower_perc))
            upper_conf = min(1.0, np.percentile(perf_info, self.upper_perc))
            median = np.percentile(perf_info, 50)

            return(lower_conf, upper_conf, median)
        
        # Harrell's C Info 
        self.lower_train_H, self.upper_train_H, self.med_train_H = do_calc(self, self.bs_train_H)
        self.lower_test_H, self.upper_test_H, self.med_test_H = do_calc(self, self.bs_test_H)

        # Uno's C Info 
        self.lower_train_U, self.upper_train_U, self.med_train_U = do_calc(self, self.bs_train_U)
        self.lower_test_U, self.upper_test_U, self.med_test_U = do_calc(self, self.bs_test_U)

    def grid_hp_tuning(self): 
        def identify_model(self): 
            if self.model_type == 'cph': 
                curr_model = CoxPHSurvivalAnalysis()
                param_grid = {'alpha': (1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8)}

            elif self.model_type == 'ssvm_lin':
                curr_model = FastSurvivalSVM()
                param_grid = {'alpha': (1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8),
                              'optimizer': ('avltree', 'rbtree', 'simple')}
                
            elif self.model_type == 'ssvm_ker':
                curr_model = FastKernelSurvivalSVM()
                param_grid = {'alpha': (1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8), 
                              'optimizer': ('avltree', 'rbtree'), 
                              'kernel': ('linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine')}
                
            elif self.model_type == 'rsf':
                curr_model = RandomSurvivalForest()
                param_grid = {'n_estimators': (250, 500, 750, 1000), 
                              'max_depth': (5, 10, 20, 50), 
                              'min_samples_split': (5, 10, 20, 50), 
                              'min_samples_leaf': (5, 10, 20, 50)}
        
            return(curr_model, param_grid)
        model, parameters = identify_model(self)
        grid_search = GridSearchCV(model, parameters, cv = 5)
        self.gs_hp_info = grid_search.fit(self.X_train, self.y_train)
        print("Grid Search Complete for " + str(self.model_type))
        
    def bayes_hp_tuning(self): 
        def identify_model(self): 
            if self.model_type == 'cph': 
                curr_model = CoxPHSurvivalAnalysis()
                param_grid = {'alpha': Real(1e-10, 1e10, prior = 'uniform')}

            elif self.model_type == 'ssvm_lin':
                curr_model = FastSurvivalSVM()
                param_grid = {'alpha': Real(1e-10, 1e10, prior = 'uniform'),
                              'optimizer': Categorical(['avltree', 'rbtree', 'simple'])}
                
            elif self.model_type == 'ssvm_ker':
                curr_model = FastKernelSurvivalSVM()
                param_grid = {'alpha': Real(1e-10, 1e10, prior = 'uniform'), 
                              'optimizer': Categorical(['avltree', 'rbtree']), 
                              'kernel': Categorical(['linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'])}
                
            elif self.model_type == 'rsf':
                curr_model = RandomSurvivalForest()
                param_grid = {'n_estimators': Integer(100, 2000, prior = 'uniform'), 
                              'max_depth': Integer(5, 100, prior = 'uniform'), 
                              'min_samples_split': Integer(5, 100, prior = 'uniform'), 
                              'min_samples_leaf': Integer(5, 100, prior = 'uniform')}
        
            return(curr_model, param_grid)
        
        model, parameters = identify_model(self)
        bayes_opt = BayesSearchCV(
            estimator = model, 
            search_spaces = parameters,
            cv = 5, 
            random_state = 42,
            n_iter = 50
        )
        self.bayes_hp_info = bayes_opt.fit(self.X_train, self.y_train)
        
    def get_hps(self): 
        hp_df = self.hyperparams.loc[(self.hyperparams['model_type'] == self.model_type) & (self.hyperparams['clin_end'] == self.surv_type) & (self.hyperparams['feat_type'] == self.feature_type) & (self.hyperparams['combat'] == int(self.combat))]
        hp_df = hp_df.reset_index(drop = True)
        self.hp_vals = json.loads(hp_df['opt_params'][0])

    @staticmethod
    def create_model(model_type, root_path, surv_type, feature_type, combat, feat_filt): 
        if model_type == 'cph': 
            curr_model = CPHModel(root_path = root_path, 
                                surv_type = surv_type, 
                                feature_type = feature_type,
                                combat = combat, 
                                feat_filt = feat_filt)

        elif model_type == 'ssvm_lin':
            curr_model = LinearSSVMModel(root_path = root_path, 
                                    surv_type = surv_type, 
                                    feature_type = feature_type,
                                    combat = combat, 
                                    feat_filt = feat_filt)
            
        elif model_type == 'ssvm_ker':
            curr_model = KernelSSVMModel(root_path = root_path, 
                                    surv_type = surv_type, 
                                    feature_type = feature_type,
                                    combat = combat, 
                                    feat_filt = feat_filt)
            
        elif model_type == 'rsf':
            curr_model = RSFModel(root_path = root_path, 
                                    surv_type = surv_type, 
                                    feature_type = feature_type,
                                    combat = combat, 
                                    feat_filt = feat_filt)
        return(curr_model)
        
class CPHModel(SurvivalModel):
    def __init__(self, root_path, surv_type, feature_type, combat, feat_filt):
        super().__init__(root_path, 'cph', surv_type, feature_type, combat, feat_filt)

    def train(self): 
        self.get_hps()
        self.model = CoxPHSurvivalAnalysis(alpha = self.hp_vals['alpha'], 
                                           ties = 'breslow', 
                                           n_iter = 100)
        self.model.fit(self.X_train, self.y_train)    

    def late_train(self, params): 
        self.model = CoxPHSurvivalAnalysis(alpha = params['alpha'], 
                                           ties = 'breslow', 
                                           n_iter = 100)
        self.model.fit(self.X_train, self.y_train)      

class LinearSSVMModel(SurvivalModel): 
    def __init__(self, root_path, surv_type, feature_type, combat, feat_filt):
        super().__init__(root_path, 'ssvm_lin', surv_type, feature_type, combat, feat_filt)

    def train(self): 
        self.get_hps()
        self.model = FastSurvivalSVM(alpha = self.hp_vals['alpha'], 
                                     optimizer = self.hp_vals['optimizer'], 
                                     random_state = 42, 
                                     tol = None, 
                                     rank_ratio = 1, 
                                     max_iter = 100)
        self.model.fit(self.X_train, self.y_train)

    def late_train(self, params): 
        self.get_late_
        self.model = FastSurvivalSVM(alpha = params['alpha'], 
                                     optimizer = params['optimizer'], 
                                     random_state = 42, 
                                     tol = None, 
                                     rank_ratio = 1, 
                                     max_iter = 100)
        self.model.fit(self.X_train, self.y_train)
    
class KernelSSVMModel(SurvivalModel):
    def __init__(self, root_path, surv_type, feature_type, combat, feat_filt):
        super().__init__(root_path, 'ssvm_ker', surv_type, feature_type, combat, feat_filt)

    def train(self): 
        self.get_hps()
        self.model = FastKernelSurvivalSVM(alpha = self.hp_vals['alpha'], 
                                           kernel = self.hp_vals['kernel'], 
                                           optimizer = self.hp_vals['optimizer'], 
                                           random_state = 42, 
                                           tol = None, 
                                           rank_ratio = 1, 
                                           max_iter = 100)
        self.model.fit(self.X_train, self.y_train)
    
    def late_train(self, params): 
        self.model = FastKernelSurvivalSVM(alpha = params['alpha'], 
                                           kernel = params['kernel'], 
                                           optimizer = params['optimizer'], 
                                           random_state = 42, 
                                           tol = None, 
                                           rank_ratio = 1, 
                                           max_iter = 100)
        self.model.fit(self.X_train, self.y_train)

class RSFModel(SurvivalModel):
    def __init__(self, root_path, surv_type, feature_type, combat, feat_filt):
        super().__init__(root_path, 'rsf', surv_type, feature_type, combat, feat_filt)  

    def train(self): 
        self.get_hps()
        self.model = RandomSurvivalForest(n_estimators = self.hp_vals['n_estimators'], 
                                          max_depth = self.hp_vals['max_depth'], 
                                          min_samples_split = self.hp_vals['min_samples_split'], 
                                          min_samples_leaf = self.hp_vals['min_samples_leaf'], 
                                          random_state = 42)
        self.model.fit(self.X_train, self.y_train)

    def late_train(self, params): 
        self.model = RandomSurvivalForest(n_estimators = params['n_estimators'], 
                                          max_depth = params['max_depth'], 
                                          min_samples_split = params['min_samples_split'], 
                                          min_samples_leaf = params['min_samples_leaf'], 
                                          random_state = 42)
        self.model.fit(self.X_train, self.y_train)

class LateFusionModel(SurvivalModel): 
    def __init__(self, root_path, model_type, surv_type, combat):
        super().__init__(root_path, model_type, surv_type, 'risk', combat, '_Pred')
        self.hyperparams = pd.read_csv((os.path.join(self.hyperparam_path, os.listdir(self.hyperparam_path)[0])).replace("\\", "/")) #Assumes you have trained individual models first
        self.late_hyperparam_path = (os.path.join(root_path, 'late_hyperparams')).replace("\\", "/")
        if len(os.listdir(self.late_hyperparam_path)) == 0: 
            empty_hp_df = pd.DataFrame(columns = ['clin_end', 'feat_type', 'combat', 'model_type', 'opt_params', 'feat_param_grid', 'best_Harrell_C'])
            self.late_hyperparams = empty_hp_df
        else: 
            self.late_hyperparams = pd.read_csv((os.path.join(self.late_hyperparam_path, os.listdir(self.late_hyperparam_path)[0])).replace("\\", "/"))
        self.deep_risks_path = (os.path.join(root_path, 'deep_risks')).replace("\\", "/")
        for file in os.listdir(self.deep_risks_path): 
            if surv_type in file: 
                self.deep_risks = pd.read_csv(os.path.join(self.deep_risks_path, file).replace("\\", "/"))
        self.deep_risk_train = self.deep_risks[self.deep_risks['Patient'].isin(self.train_data_init['Patient'])]
        self.deep_risk_train = pd.merge(self.deep_risk_train, self.train_data_init[['Patient', self.surv_type + 'Time', self.surv_type + 'Cens']], on = 'Patient')
        self.deep_risk_test = self.deep_risks[self.deep_risks['Patient'].isin(self.test_data_init['Patient'])]
        self.deep_risk_test = pd.merge(self.deep_risk_test, self.test_data_init[['Patient', self.surv_type + 'Time', self.surv_type + 'Cens']], on = 'Patient')

    def set_first_layer(self, parameter_grid):
        self.first_layer_models = dict()

        for i in range(0, len(parameter_grid['model_type'])):
            if parameter_grid['model_type'][i] == 'deep': 
                self.first_layer_models[i] = 'deep_risks'
            else:
                curr_model = SurvivalModel.create_model(model_type = parameter_grid['model_type'][i],
                                                        root_path = self.root_path, 
                                                        surv_type = self.surv_type, 
                                                        feature_type = parameter_grid['feature_type'][i],
                                                        combat = parameter_grid['combat'][i], 
                                                        feat_filt = parameter_grid['feat_filt'][i])            
                curr_model.train()
                self.first_layer_models[i] = curr_model
    
    def get_risk_train_test(self):
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        for key, value in self.first_layer_models.items():
            
            if value == 'deep_risks': 
                try: 
                    train_risks = self.deep_risk_train.drop(columns = [self.surv_type + 'Cens', self.surv_type + 'Time']) 
                    test_risks = self.deep_risk_test.drop(columns = [self.surv_type + 'Cens', self.surv_type + 'Time'])
                    self.train_data = pd.merge(self.train_data, train_risks, how = 'left', on = 'Patient')
                    self.test_data = pd.merge(self.test_data, test_risks, how = 'left', on = 'Patient')
                except: 
                    self.train_data = self.deep_risk_train
                    self.test_data = self.deep_risk_test
            else: 
                value.base_risks()
                try: 
                    train_risks = value.train_risks.drop(columns = [self.surv_type + 'Cens', self.surv_type + 'Time'])
                    test_risks = value.test_risks.drop(columns = [self.surv_type + 'Cens', self.surv_type + 'Time'])
                    self.train_data = pd.merge(self.train_data, train_risks, how = 'left', on = 'Patient')
                    self.test_data = pd.merge(self.test_data, test_risks, how = 'left', on = 'Patient')
                except KeyError: 
                    self.train_data = value.train_risks
                    self.test_data = value.test_risks

    def get_late_hps(self, late_params):
        hp_df = self.late_hyperparams.loc[(self.late_hyperparams['model_type'] == self.model_type) & (self.late_hyperparams['clin_end'] == self.surv_type) & (self.late_hyperparams['feat_type'] == self.feature_type) & (self.late_hyperparams['combat'] == int(self.combat)) & (self.late_hyperparams['feat_param_grid'] == str(late_params).replace("'", '"'))]
        hp_df = hp_df.reset_index(drop = True)
        self.hp_vals = json.loads(hp_df['opt_params'][0])

    def late_grid_hp_tuning(self): 
        def identify_model(self): 
            if self.model_type == 'cph': 
                curr_model = CoxPHSurvivalAnalysis()
                param_grid = {'alpha': (1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8)}

            elif self.model_type == 'ssvm_lin':
                curr_model = FastSurvivalSVM()
                param_grid = {'alpha': (1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8),
                              'optimizer': ('avltree', 'rbtree', 'simple')}
                
            elif self.model_type == 'ssvm_ker':
                curr_model = FastKernelSurvivalSVM()
                param_grid = {'alpha': (1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8), 
                              'optimizer': ('avltree', 'rbtree'), 
                              'kernel': ('linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine')}
                
            elif self.model_type == 'rsf':
                curr_model = RandomSurvivalForest()
                param_grid = {'n_estimators': (250, 500, 750, 1000), 
                              'max_depth': (5, 10, 20, 50), 
                              'min_samples_split': (5, 10, 20, 50), 
                              'min_samples_leaf': (5, 10, 20, 50)}
        
            return(curr_model, param_grid)
        model, parameters = identify_model(self)
        grid_search = GridSearchCV(model, parameters, cv = 5)
        self.get_risk_train_test()
        self.X_train, self.y_train = self.prep_Xy(self.train_data)
        self.gs_hp_info = grid_search.fit(self.X_train, self.y_train)
        print("Grid Search Complete for " + str(self.model_type))

    def train(self, late_params): 
        self.get_late_hps(late_params = late_params)
        self.get_risk_train_test()
        model_obj = SurvivalModel.create_model(model_type = self.model_type, 
                                                root_path = self.root_path, 
                                                surv_type = self.surv_type,
                                                feature_type = self.feature_type, 
                                                combat = self.combat, 
                                                feat_filt = self.feat_filt)
        model_obj.X_train, model_obj.y_train = self.prep_Xy(self.train_data)
        model_obj.X_test, model_obj.y_test = self.prep_Xy(self.test_data)

        self.X_train = model_obj.X_train
        self.y_train = model_obj.y_train

        self.X_test = model_obj.X_test
        self.y_test = model_obj.y_test

        model_obj.late_train(params = self.hp_vals)

        self.model = model_obj.model

    def bootstrap(self, iter_num): 
        self.train_samp_ids = []
        self.test_samp_ids = []
        self.bs_train_H = []
        self.bs_train_U = []
        self.bs_test_H = []
        self.bs_test_U = []

        for i in range(iter_num):
            train_samp = self.train_data.sample(self.train_data.shape[0], replace = True)
            test_samp = self.test_data.sample(self.test_data.shape[0], replace = True)

            self.train_samp_ids.append(train_samp['Patient'].to_numpy())
            self.test_samp_ids.append(test_samp['Patient'].to_numpy())

            X_train_samp, y_train_samp = self.prep_Xy(train_samp)
            X_test_samp, y_test_samp = self.prep_Xy(test_samp)
            X_test_samp = X_test_samp[X_train_samp.columns.values.tolist()]

            # Calculate Harrell's C 
            self.bs_train_H.append(self.model.score(X_train_samp, y_train_samp))
            self.bs_test_H.append(self.model.score(X_test_samp, y_test_samp))

            # Calculate Uno's C 
            pred_train_U = pd.Series(self.model.predict(X_train_samp))
            self.bs_train_U.append(concordance_index_ipcw(self.y_train, y_train_samp, pred_train_U)[0])
            
            pred_test_U = pd.Series(self.model.predict(X_test_samp))
            self.bs_test_U.append(concordance_index_ipcw(self.y_test, y_test_samp, pred_test_U)[0])