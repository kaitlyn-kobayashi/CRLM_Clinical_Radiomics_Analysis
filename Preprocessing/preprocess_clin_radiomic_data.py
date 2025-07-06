import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.preprocessing import StandardScaler

def calc_dfi(row):
    '''
    Calculates disease-free interval, defined as time from primary 
    resection to diagnosis of colorectal liver metastases. 

    Parameters
    -----------
    row: pandas.DataFrame row
        Each row represents one patient's data. Must have 
        CRC Resection Date and Synchronous Disease headers.

    Returns
    -----------
    dfi: int 
        Disease-free interval in days.
    '''
    if (pd.isnull(row['CRC Resection Date']) == True) and (row['Synchronous Disease?'] == 1):
        dfi = 0
    elif (pd.isnull(row['CRC Resection Date']) == True) and (row['Synchronous Disease?'] == 0):
        dfi = None
    else: 
        dfi = datetime.strptime(row['CRLM Diagnosis Date'], '%Y-%m-%d') - datetime.strptime(row['CRC Resection Date'], '%Y-%m-%d')
        dfi = dfi.days
    return (dfi)

def bin_dfi(row): 
    '''
    Binarizes DFI based on a 12 month threshold. Requires DFI
    column to exist.

    Parameters
    -----------
    row: pandas.DataFrame row
        Each row represents one patient's data. Must have 
        CRC Resection Date and Synchronous Disease headers.

    Returns
    -----------
    dfi_less_12: int
        0 if DFI > 12 months, 1 if DFI < 12 months
    '''
    if pd.isnull(row['DFI']):
        dfi_less_12 = None
    else: 
        dfi_less_12 = 1 if row['DFI'] < 365 else 0
    return (dfi_less_12)

def binarize_resec_site(clin_df): 
    '''
    Separates primary resection site variable into distinct 
    binary columns per resection site for downstream analysis. 

    Parameters
    -----------
    clin_df: pandas.DataFrame
        DataFrame with at least the CRC Resection Site column. 
    
    Returns
    -----------
    new_clin_df: pandas.DataFrame 
        clin_df with the added binarized primary resection site
        columns appended to it. 
    '''    
    new_clin_df = clin_df.copy(deep = True)
    for site in range(1,4): 
        new_clin_df['CRC Resection Site ' + str(site)] = new_clin_df['CRC Resection Site'].apply(lambda x: 1 if x == site else None if np.isnan(x) else 0)

    new_clin_df = new_clin_df.rename(columns = {'CRC Resection Site 1': 'Right CRC Resection',
                                      'CRC Resection Site 2': 'Left CRC Resection',
                                      'CRC Resection Site 3': 'Transverse CRC Resection'})
    return (new_clin_df)

def uniqueness_and_normalize(train_set, test_set, filt):
    '''
    Removes radiomic features whose values are identical 50% or 
    more of the time. Then normalizes data based on z-score 
    normalization. All calculations done on training set only. 

    Parameters
    -----------
    train_set: pandas.DataFrame
        Only contains radiomic features of patients in the 
        training set.
    test_set: pandas.DataFrame 
        Only contains radiomic features of patients in the test
        set.
    filt: str
        Used to filter columns by a common substring if further
        processing is required. Regex term. 

    Returns 
    -----------
    train_norm: pandas.DataFrame
        Preprocessed radiomic features from the training set. 
    test_norm: pandas.DataFrame 
        Preprocessed radiomic features from the test set. 
    '''
    train_copy = train_set.copy(deep = True) 
    test_copy = test_set.copy(deep = True) 
    train_copy = train_copy.filter(regex = filt, axis = 1)
    test_copy = test_copy.filter(regex = filt, axis = 1)

    #Remove features that have the same value for 50% or more of the dataset
    for feature in train_copy.columns.values.tolist():
        feat_val_count = train_copy[feature].value_counts()
        for count in feat_val_count: 
            if count > (train_copy.shape[0]*0.5):
                train_copy = train_copy.drop(columns = [feature])
                test_copy = test_copy.drop(columns = [feature])

    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_copy.to_numpy())
    train_norm = pd.DataFrame(train_norm, columns = train_copy.columns.values.tolist())
    test_norm = scaler.transform(test_copy.to_numpy())
    test_norm = pd.DataFrame(test_norm, columns = test_copy.columns.values.tolist())

    return(train_norm, test_norm)

def clamp_rad_values(train_data, test_data, filter, quant_lower, quant_upper): 
    '''
    TO BE USED AFTER NORMALIZATION WITH SURVIVAL DATA
    Removes outliers by clamping values to specified quantile 
    range. 

    Parameters
    -----------
    train_data: pandas.DataFrame 
        Contains normalized data from the training set.
    test_data: pandas.DataFrame 
        Contains normalized data from the testing set. 
    filter: str
        Filters columns to include by a regex term.
    quant_lower: float 
        Lower quantile to clamp values to. Must be between 0 and 1.
    quant_upper: float
        Upper quantile to clamp values to. Must be between 0 and 1.

    Returns
    ----------
    feat_only_train: pandas.DataFrame
        Clamped training data with only radiomic features. 
    feat_only_test: pandas.DataFrame 
        Clamped testing data with only radiomic features. 
    '''
    feat_only_train = train_data.filter(regex = filter) 
    feat_only_test = test_data.filter(regex = filter)
    
    lower_quant_vals = feat_only_train.quantile(quant_lower)
    upper_quant_vals = feat_only_train.quantile(quant_upper)
    
    for col in feat_only_train.columns.values.tolist(): 
        curr_lower = lower_quant_vals[col]
        curr_upper = upper_quant_vals[col]
        feat_only_train[col].loc[feat_only_train[col] <= curr_lower] = curr_lower  
        feat_only_train[col].loc[feat_only_train[col] >= curr_upper] = curr_upper

        feat_only_test[col].loc[feat_only_test[col] <= curr_lower] = curr_lower
        feat_only_test[col].loc[feat_only_test[col] >= curr_upper] = curr_upper
    
    return(feat_only_train, feat_only_test)

if __name__ == '__main__': 
    #Prepare clinical data for imputation and further analysis
    clin_data = pd.read_csv(('path/to/clinical_data.csv'))
    cols_to_keep = ['Age at Liver Surgery', 
                    'Sex', 
                    'Height (m)', 
                    'Weight (kg)',
                    'Patient', 
                    'DFI Less Than 12 Months',
                    'CRC Resection Site', 
                    'Node Positive?', 
                    'Synchronous Disease?',
                    'CEA Levels Before Resection',
                    'Number of CRLM', 
                    'Diameter of Largest CRLM',
                    'Extrahepatic Disease?',
                    'Bilateral Metastases?',
                    'PV Embolization?',
                    'Surv Time Censor 4', 
                    'Surv Time Censor 7', 
                    'Censor 4: HDFS', 
                    'Censor 7: OS',
                    'OSTrain',
                    'HDFSTrain'
                    ]

    clin_data['DFI'] = clin_data.apply(calc_dfi, axis = 1)
    clin_data['DFI Less Than 12 Months'] = clin_data.apply(bin_dfi, axis = 1)

    # Clean dataframe for missing values and rename survival columns 
    preop_df = clin_data[cols_to_keep]
    preop_df['CRC Resection Site'] = preop_df['CRC Resection Site'].replace(5, np.NaN)
    preop_df = preop_df.replace('unknown', np.NaN)
    preop_df = preop_df.rename(columns = {'Surv Time Censor 4': 'HDFSTime',
                                        'Surv Time Censor 7': 'OSTime', 
                                        'Censor 4: HDFS': 'HDFSCens', 
                                        'Censor 7: OS': 'OSCens'})
    final_preop_df = binarize_resec_site(preop_df)
    final_preop_df.to_csv('Clinical_Data_For_Imputation_and_Survival_2025-02-27.csv')

    ## Preprocess radiomic features ##
    #Hand-crafted radiomic features  
    rad_features = pd.read_csv('path/to/radiomic_features.csv')
    rad_surv_data = pd.merge(final_preop_df[['OSTime', 'OSCens', 'OSTrain', 'HDFSTime', 'HDFSCens', 'HDFSTrain', 'Patient']], rad_features, how = 'left', on = 'Patient')

    rad_surv_os_train = rad_surv_data[rad_surv_data['OSTrain'] == 1].filter(regex = 'Liver_|Tumor_|Patient|OSTime|OSCens').reset_index(drop = True)
    rad_surv_os_test = rad_surv_data[rad_surv_data['OSTrain'] == 0].filter(regex = 'Liver_|Tumor_|Patient|OSTime|OSCens').reset_index(drop = True)

    rad_surv_hdfs_train = rad_surv_data[rad_surv_data['HDFSTrain'] == 1].filter(regex = 'Liver_|Tumor_|Patient|HDFSTime|HDFSCens').reset_index(drop = True)
    rad_surv_hdfs_test = rad_surv_data[rad_surv_data['HDFSTrain'] == 0].filter(regex = 'Liver_|Tumor_|Patient|HDFSTime|HDFSCens').reset_index(drop = True)

    norm_os_train, norm_os_test = uniqueness_and_normalize(rad_surv_os_train, rad_surv_os_test, filt = 'Liver_|Tumor_')
    norm_hdfs_train, norm_hdfs_test = uniqueness_and_normalize(rad_surv_hdfs_train, rad_surv_hdfs_test, filt = 'Liver_|Tumor_')

    clamped_os_train, clamped_os_test = clamp_rad_values(norm_os_train, norm_os_test, filter = 'Liver_|Tumor_', quant_lower = 0.01, quant_upper = 0.99)
    clamped_hdfs_train, clamped_hdfs_test = clamp_rad_values(norm_hdfs_train, norm_hdfs_test, filter = 'Liver_|Tumor_', quant_lower = 0.01, quant_upper = 0.99)

    clamped_os_train = pd.concat([clamped_os_train, rad_surv_os_train[['Patient', 'OSTime', 'OSCens']]], axis = 1).reset_index(drop = True)
    clamped_os_test = pd.concat([clamped_os_test, rad_surv_os_test[['Patient', 'OSTime', 'OSCens']]], axis = 1).reset_index(drop = True)

    clamped_hdfs_train = pd.concat([clamped_hdfs_train, rad_surv_hdfs_train[['Patient', 'HDFSTime', 'HDFSCens']]], axis = 1).reset_index(drop = True)
    clamped_hdfs_test = pd.concat([clamped_hdfs_test, rad_surv_hdfs_test[['Patient', 'HDFSTime', 'HDFSCens']]], axis = 1).reset_index(drop = True)

    clamped_os_train.to_csv("path/to/ref_files/OS_rad_train_data.csv")
    clamped_os_test.to_csv("path/to/rad_test/OS_rad_0_val_data.csv")

    clamped_hdfs_train.to_csv("path/to/ref_files/HDFS_rad_train_data.csv")
    clamped_hdfs_test.to_csv("path/to/rad_test/HDFS_rad_0_val_data.csv")

    #Deep learning radiomic features 
    deep_livwhole_os = pd.read_csv("path/to/OS_livwhole_DeepFeatures.csv")
    deep_livwhole_hdfs = pd.read_csv("path/to/HDFS_livhwole_DeepFeatures.csv")
    deep_livpar_os = pd.read_csv("path/to/OS_livpar_DeepFeatures.csv")
    deep_livpar_hdfs = pd.read_csv("path/to/HDFS_livpar_DeepFeatures.csv")
    deep_tumor_os = pd.read_csv("path/to/OS_tumor_DeepFeatures.csv")
    deep_tumor_hdfs = pd.read_csv("path/to/HDFS_tumor_DeepFeatures.csv")

    deep_surv_livwhole_os = pd.merge(final_preop_df[['OSTime', 'OSCens', 'OSTrain', 'HDFSTime', 'HDFSCens', 'HDFSTrain', 'Patient']], deep_livwhole_os, how = 'left', on = 'Patient')
    deep_surv_livwhole_hdfs = pd.merge(final_preop_df[['OSTime', 'OSCens', 'OSTrain', 'HDFSTime', 'HDFSCens', 'HDFSTrain', 'Patient']], deep_livwhole_hdfs, how = 'left', on = 'Patient')
    deep_surv_livpar_os = pd.merge(final_preop_df[['OSTime', 'OSCens', 'OSTrain', 'HDFSTime', 'HDFSCens', 'HDFSTrain', 'Patient']], deep_livpar_os, how = 'left', on = 'Patient')
    deep_surv_livpar_hdfs = pd.merge(final_preop_df[['OSTime', 'OSCens', 'OSTrain', 'HDFSTime', 'HDFSCens', 'HDFSTrain', 'Patient']], deep_livpar_hdfs, how = 'left', on = 'Patient')
    deep_surv_tumor_os = pd.merge(final_preop_df[['OSTime', 'OSCens', 'OSTrain', 'HDFSTime', 'HDFSCens', 'HDFSTrain', 'Patient']], deep_tumor_os, how = 'left', on = 'Patient')
    deep_surv_tumor_hdfs = pd.merge(final_preop_df[['OSTime', 'OSCens', 'OSTrain', 'HDFSTime', 'HDFSCens', 'HDFSTrain', 'Patient']], deep_tumor_hdfs, how = 'left', on = 'Patient')

    deep_surv_livwhole_os_train = deep_surv_livwhole_os[deep_surv_livwhole_os['OSTrain'] == 1].filter(regex = 'CNN_|Patient|OSTime|OSCens').reset_index(drop = True)
    deep_surv_livwhole_os_test = deep_surv_livwhole_os[deep_surv_livwhole_os['OSTrain'] == 0].filter(regex = 'CNN_|Patient|OSTime|OSCens').reset_index(drop = True)
    deep_surv_livpar_os_train = deep_surv_livpar_os[deep_surv_livpar_os['OSTrain'] == 1].filter(regex = 'CNN_|Patient|OSTime|OSCens').reset_index(drop = True) 
    deep_surv_livpar_os_test = deep_surv_livpar_os[deep_surv_livpar_os['OSTrain'] == 0].filter(regex = 'CNN_|Patient|OSTime|OSCens').reset_index(drop = True)
    deep_surv_tumor_os_train = deep_surv_tumor_os[deep_surv_tumor_os['OSTrain'] == 1].filter(regex = 'CNN_|Patient|OSTime|OSCens').reset_index(drop = True) 
    deep_surv_tumor_os_test = deep_surv_tumor_os[deep_surv_tumor_os['OSTrain'] == 0].filter(regex = 'CNN_|Patient|OSTime|OSCens').reset_index(drop = True) 

    deep_surv_livwhole_hdfs_train = deep_surv_livwhole_hdfs[deep_surv_livwhole_hdfs['HDFSTrain'] == 1].filter(regex = 'CNN_|Patient|HDFSTime|HDFSCens').reset_index(drop = True) 
    deep_surv_livwhole_hdfs_test = deep_surv_livwhole_hdfs[deep_surv_livwhole_hdfs['HDFSTrain'] == 0].filter(regex = 'CNN_|Patient|HDFSTime|HDFSCens').reset_index(drop = True) 
    deep_surv_livpar_hdfs_train = deep_surv_livpar_hdfs[deep_surv_livpar_hdfs['HDFSTrain'] == 1].filter(regex = 'CNN_|Patient|HDFSTime|HDFSCens').reset_index(drop = True) 
    deep_surv_livpar_hdfs_test = deep_surv_livpar_hdfs[deep_surv_livpar_hdfs['HDFSTrain'] == 0].filter(regex = 'CNN_|Patient|HDFSTime|HDFSCens').reset_index(drop = True) 
    deep_surv_tumor_hdfs_train = deep_surv_tumor_hdfs[deep_surv_tumor_hdfs['HDFSTrain'] == 1].filter(regex = 'CNN_|Patient|HDFSTime|HDFSCens').reset_index(drop = True) 
    deep_surv_tumor_hdfs_test = deep_surv_tumor_hdfs[deep_surv_tumor_hdfs['HDFSTrain'] == 0].filter(regex = 'CNN_|Patient|HDFSTime|HDFSCens').reset_index(drop = True) 

    norm_livwhole_os_train, norm_livwhole_os_test = uniqueness_and_normalize(deep_surv_livwhole_os_train, deep_surv_livwhole_os_test, filt = 'CNN_')
    norm_livwhole_hdfs_train, norm_livwhole_hdfs_test = uniqueness_and_normalize(deep_surv_livwhole_hdfs_train, deep_surv_livwhole_hdfs_test, filt = 'CNN_')
    norm_livpar_os_train, norm_livpar_os_test = uniqueness_and_normalize(deep_surv_livpar_os_train, deep_surv_livpar_os_test, filt = 'CNN_')
    norm_livpar_hdfs_train, norm_livpar_hdfs_test = uniqueness_and_normalize(deep_surv_livpar_hdfs_train, deep_surv_livpar_hdfs_test, filt = 'CNN_')
    norm_tumor_os_train, norm_tumor_os_test = uniqueness_and_normalize(deep_surv_tumor_os_train, deep_surv_tumor_os_test, filt = 'CNN_')
    norm_tumor_hdfs_train, norm_tumor_hdfs_test = uniqueness_and_normalize(deep_surv_tumor_hdfs_train, deep_surv_tumor_hdfs_test, filt = 'CNN_')

    cl_livwhole_os_train, cl_livwhole_os_test = clamp_rad_values(norm_livwhole_os_train, norm_livwhole_os_test, filter = 'LivWhole', quant_lower = 0.01, quant_upper = 0.99)
    cl_livwhole_hdfs_train, cl_livwhole_hdfs_test = clamp_rad_values(norm_livwhole_hdfs_train, norm_livwhole_hdfs_test, filter = 'LivWhole', quant_lower = 0.01, quant_upper = 0.99)
    cl_livpar_os_train, cl_livpar_os_test = clamp_rad_values(norm_livpar_os_train, norm_livpar_os_test, filter = 'LivPar', quant_lower = 0.01, quant_upper = 0.99)
    cl_livpar_hdfs_train, cl_livpar_hdfs_test = clamp_rad_values(norm_livpar_hdfs_train, norm_livpar_hdfs_test, filter = 'LivPar', quant_lower = 0.01, quant_upper = 0.99) 
    cl_tumor_os_train, cl_tumor_os_test = clamp_rad_values(norm_tumor_os_train, norm_tumor_os_test, filter = 'Tumor_', quant_lower = 0.01, quant_upper = 0.99)
    cl_tumor_hdfs_train, cl_tumor_hdfs_test = clamp_rad_values(norm_tumor_hdfs_train, norm_tumor_hdfs_test, filter = 'Tumor_', quant_lower = 0.01, quant_upper = 0.99)

    cl_livwhole_os_train = pd.concat([cl_livwhole_os_train, deep_surv_livwhole_os_train[['Patient', 'OSTime', 'OSCens']]], axis = 1).reset_index(drop = True) 
    cl_livwhole_os_test = pd.concat([cl_livwhole_os_test, deep_surv_livwhole_os_test[['Patient', 'OSTime', 'OSCens']]], axis = 1).reset_index(drop = True) 
    cl_livwhole_hdfs_train = pd.concat([cl_livwhole_hdfs_train, deep_surv_livwhole_hdfs_train[['Patient', 'HDFSTime', 'HDFSCens']]], axis = 1).reset_index(drop = True) 
    cl_livwhole_hdfs_test = pd.concat([cl_livwhole_hdfs_test, deep_surv_livwhole_hdfs_test[['Patient', 'HDFSTime', 'HDFSCens']]], axis = 1). reset_index(drop = True) 

    cl_livpar_os_train = pd.concat([cl_livpar_os_train, deep_surv_livpar_os_train[['Patient', 'OSTime', 'OSCens']]], axis = 1).reset_index(drop = True) 
    cl_livpar_os_test = pd.concat([cl_livpar_os_test, deep_surv_livpar_os_test[['Patient', 'OSTime', 'OSCens']]], axis = 1).reset_index(drop = True) 
    cl_livpar_hdfs_train = pd.concat([cl_livpar_hdfs_train, deep_surv_livpar_hdfs_train[['Patient', 'HDFSTime', 'HDFSCens']]], axis = 1).reset_index(drop = True) 
    cl_livpar_hdfs_test = pd.concat([cl_livpar_hdfs_test, deep_surv_livpar_hdfs_test[['Patient', 'HDFSTime', 'HDFSCens']]], axis = 1).reset_index(drop = True) 

    cl_tumor_os_train = pd.concat([cl_tumor_os_train, deep_surv_tumor_os_train[['Patient', 'OSTime', 'OSCens']]], axis = 1).reset_index(drop = True) 
    cl_tumor_os_test = pd.concat([cl_tumor_os_test, deep_surv_tumor_os_test[['Patient', 'OSTime', 'OSCens']]], axis = 1).reset_index(drop = True) 
    cl_tumor_hdfs_train = pd.concat([cl_tumor_hdfs_train, deep_surv_tumor_hdfs_train[['Patient', 'HDFSTime', 'HDFSCens']]], axis = 1).reset_index(drop = True) 
    cl_tumor_hdfs_test = pd.concat([cl_tumor_hdfs_test, deep_surv_tumor_hdfs_test[['Patient', 'HDFSTime', 'HDFSCens']]], axis = 1).reset_index(drop = True) 
    
    #Export to appropriate locations
    cl_livwhole_os_train.to_csv("path/to/ref_files/OS_livwhole_cnn_train.csv")
    cl_livwhole_os_test.to_csv("path/to/OS_livwhole_cnn_0_val_data.csv")

    cl_livwhole_hdfs_train.to_csv("path/to/ref_files/HDFS_livwhole_cnn_train.csv")
    cl_livwhole_hdfs_test.to_csv("path/to/HDFS_livwhole_cnn_0_val_data.csv")

    cl_livpar_os_train.to_csv("path/to/ref_files/OS_cnn_train_livpar.csv")
    cl_livpar_os_test.to_csv("path/to/rad_test/OS_cnn_0_val_data_livpar.csv")

    cl_livpar_hdfs_train.to_csv("path/to/ref_files/HDFS_cnn_train_livpar.csv")
    cl_livpar_hdfs_test.to_csv("path/to/rad_test/HDFS_cnn_0_val_data_livpar.csv")

    cl_tumor_os_train.to_csv("path/to/ref_files/OS_cnn_train_tumor.csv")
    cl_tumor_os_test.to_csv("path/to/rad_test/OS_cnn_0_val_data_tumor.csv")

    cl_tumor_hdfs_train.to_csv("path/to/ref_files/HDFS_cnn_train_tumor.csv")
    cl_tumor_hdfs_test.to_csv("path/to/rad_test/HDFS_cnn_0_val_data_tumor.csv")

