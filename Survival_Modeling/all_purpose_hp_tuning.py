#Hyperparameter tuning of models using Bayesian optimization and/or grid search
import json
import pandas as pd


from survival_model import SurvivalModel, LateFusionModel
from joblib import Parallel, delayed

class HyperparameterTuner: 
    def __init__(self, root_path, model_types, surv_types, feature_types, combat_types): 
        self.root_path = root_path
        self.model_types = model_types
        self.surv_types = surv_types
        self.feature_types = feature_types
        self.combat_types = combat_types

    def generate_run_params(self):
        run_params = pd.DataFrame()
        for model in self.model_types: 
            for survival in self.surv_types:
                for feat_type in self.feature_types: 
                    for combat in self.combat_types:
                        if feat_type == 'rad': 
                            filt = 'Liver_|Tumor_'
                        elif feat_type == 'cnn':
                            filt = 'LivPar|LivWhole|Tumor'
                        elif feat_type == 'clin':
                            filt = 'Clin_'
                        elif feat_type == 'risk':
                            filt = '_Pred'
                        curr_params = pd.DataFrame({'Model': [model], 
                                                    'SurvType': [survival], 
                                                    'FeatType': [feat_type], 
                                                    'Combat': [combat], 
                                                    'FeatureFilter': [filt]})
                        run_params = pd.concat([run_params, curr_params], axis = 0).reset_index(drop = True)
        
        self.run_parameters = run_params
    
    def run_bayes_hp_tuning(self, param_index):
        curr_params = self.run_parameters
        curr_model = SurvivalModel(root_path = self.root_path, 
                                   model_type = curr_params['Model'][param_index], 
                                   surv_type = curr_params['SurvType'][param_index],
                                   feature_type = curr_params['FeatType'][param_index],
                                   combat = curr_params['Combat'][param_index],
                                   feat_filt = curr_params['FeatureFilter'][param_index])
        curr_model.bayes_hp_tuning()
        hp_perf_df = pd.DataFrame({'clin_end': curr_params['SurvType'][param_index],
                                   'feat_type': curr_params['FeatType'][param_index],
                                   'combat': curr_params['Combat'][param_index],
                                   'model_type': curr_params['Model'][param_index],
                                   'opt_params': json.dumps(curr_model.bayes_hp_info.best_params_),
                                   'best_Harrell_C': curr_model.bayes_hp_info.best_score_},
                                   index = [param_index])
        return(curr_model, hp_perf_df)
    
    def run_grid_hp_tuning(self, param_index): 
        curr_params = self.run_parameters
        curr_model = SurvivalModel(root_path = self.root_path, 
                                   model_type = curr_params['Model'][param_index], 
                                   surv_type = curr_params['SurvType'][param_index],
                                   feature_type = curr_params['FeatType'][param_index],
                                   combat = curr_params['Combat'][param_index],
                                   feat_filt = curr_params['FeatureFilter'][param_index])
        curr_model.grid_hp_tuning()
        hp_perf_df = pd.DataFrame({'clin_end': curr_params['SurvType'][param_index],
                                   'feat_type': curr_params['FeatType'][param_index],
                                   'combat': curr_params['Combat'][param_index],
                                   'model_type': curr_params['Model'][param_index],
                                   'opt_params': json.dumps(curr_model.gs_hp_info.best_params_),
                                   'best_Harrell_C': curr_model.gs_hp_info.best_score_},
                                   index = [param_index])
        return(curr_model, hp_perf_df)
    
    def run_late_grid_hp_tuning(self, param_index, late_param_grid): 
        curr_params = self.run_parameters
        hp_perf_df = pd.DataFrame()
        for idx, row in late_param_grid.iterrows():
            parameter_grid = json.loads(row['param_grid'])
            curr_model = LateFusionModel(root_path = self.root_path, 
                                         model_type = curr_params['Model'][param_index], 
                                         surv_type = curr_params['SurvType'][param_index], 
                                         combat = curr_params['Combat'][param_index])
            curr_model.set_first_layer(parameter_grid = parameter_grid)
            curr_model.late_grid_hp_tuning()
            curr_perf_df = pd.DataFrame({'clin_end': curr_params['SurvType'][param_index],
                                        'feat_type': curr_params['FeatType'][param_index],
                                        'combat': curr_params['Combat'][param_index],
                                        'model_type': curr_params['Model'][param_index],
                                        'feat_param_grid': str(parameter_grid),
                                        'opt_params': json.dumps(curr_model.gs_hp_info.best_params_),
                                        'best_Harrell_C': curr_model.gs_hp_info.best_score_},
                                        index = [param_index])
            hp_perf_df = pd.concat([hp_perf_df, curr_perf_df], axis = 0)
        
        return(hp_perf_df)
    
    def run_parallel_late_grid_tune(self, param_index, late_param_grid, cores, total_cores):
        def grid_search(self, curr_params, param_index, row): 
            parameter_grid = json.loads(row['param_grid'])
            print(parameter_grid)
            curr_model = LateFusionModel(root_path = self.root_path, 
                                         model_type = curr_params['Model'][param_index], 
                                         surv_type = curr_params['SurvType'][param_index], 
                                         combat = curr_params['Combat'][param_index])
            curr_model.set_first_layer(parameter_grid = parameter_grid)
            curr_model.late_grid_hp_tuning()
            curr_perf_df = pd.DataFrame({'clin_end': curr_params['SurvType'][param_index],
                                        'feat_type': curr_params['FeatType'][param_index],
                                        'combat': curr_params['Combat'][param_index],
                                        'model_type': curr_params['Model'][param_index],
                                        'feat_param_grid': str(parameter_grid),
                                        'opt_params': json.dumps(curr_model.gs_hp_info.best_params_),
                                        'best_Harrell_C': curr_model.gs_hp_info.best_score_},
                                        index = [param_index])            
            return(curr_perf_df)
        
        curr_params = self.run_parameters
        hp_perf_df = pd.DataFrame()
        results = Parallel(n_jobs = cores//total_cores, verbose = 10)(delayed(grid_search)(self, curr_params, param_index, row) for idx, row in late_param_grid.iterrows())
        
        for result in results: 
            hp_perf_df = pd.concat([hp_perf_df, result], axis = 0)
        
        return(hp_perf_df)
    


