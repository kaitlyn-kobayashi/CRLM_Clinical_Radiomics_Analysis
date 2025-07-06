from sklearn.model_selection import train_test_split
import pandas as pd 

#Directories and variables 
clin_data_df = pd.read_csv("path/to/clinical_data_with_patient_list.csv")
stratify_vars_OS = ["Censor 7: OS"]
stratify_vars_HDFS = ["Censor 4: HDFS"]
rand_state = 70

train_OS, test_OS = train_test_split(clin_data_df, test_size = 0.20, random_state = rand_state, stratify = clin_data_df[stratify_vars_OS])
train_HDFS, test_HDFS = train_test_split(clin_data_df, test_size = 0.20, random_state = rand_state, stratify = clin_data_df[stratify_vars_HDFS])

# Add to clinical data as columns 
train_OS['OSTrain'] = 1 
test_OS['OSTrain'] = 0

clin_data_OS = pd.concat([train_OS, test_OS], axis = 0)

train_HDFS['HDFSTrain'] = 1
test_HDFS['HDFSTrain'] = 0

clin_data_HDFS = pd.concat([train_HDFS, test_HDFS], axis = 0)

clin_data_end = pd.merge(clin_data_OS, clin_data_HDFS[['XNAT ID', 'HDFSTrain']], on = 'XNAT ID')

# Export as dataframe 
clin_data_end.to_csv('npj_Digital_Medicine_Clinical_Data_FINAL_SPLIT.csv')
