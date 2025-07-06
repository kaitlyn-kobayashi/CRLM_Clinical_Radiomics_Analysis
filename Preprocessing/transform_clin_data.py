import pandas as pd

def rename_columns(formatted_df, unformatted_df, ignore_list): 
    '''
    Replaces column names based on a template formatted DataFrame 
    to be more readable/accessible. 

    Parameters
    -----------
    formatted_df: pandas.DataFrame 
        Any DataFrame which has the column headers in the exact
        order wanted for replacement.
    unformatted_df: pandas.DataFrame 
        The DataFrame to have its column headers replaced. 
    ignore_list: list 
        The names of columns which you would like to drop from the 
        final DataFrame.

    Returns
    -----------
    new_cols_df: pandas.DataFrame
        A version of the unformatted_df with the columns names 
        replaced accordingly and the unwanted columns dropped. 
    '''
    new_colnames = list(formatted_df.columns) 
    old_colnames = list(unformatted_df.columns) 
    dropped_df = unformatted_df.drop(labels = ignore_list, 
                                     axis = 1) 
    
    idx = 0 
    for new_col in new_colnames: 
        new_cols_df = dropped_df.rename(columns = {old_colnames[idx]: new_col})
        idx += 1
    
    return(new_cols_df)

def recode_bin_vars(df, bin_var_list): 
    '''
    Change binary variables which are coded with 2 = false to be 
    0 = false. 

    Parameters 
    -----------
    df: pandas.DataFrame 
        DataFrame which includes the binary variable columns to be 
        modified.
    bin_var_list: list
        Contains the column headers of the binary variables. 
    
    Returns 
    -----------
    recoded_df: pandas.DataFrame
        df with the chosen binary variables recoded to 0 = false, 
        1 = true. 
    '''

    for bin_var in bin_var_list: 
        recoded_df = df.replace({bin_var: 2}, 0)
    
    return(recoded_df)

def add_events(clin_df, time_unit): 
    '''
    Adds seven event and survival time definitions:
        1. Initial Hepatic Disease-Free Time (Initial HDF)
           Interval from hepatic resection until initial hepatic
           recurrence. Patients who are lost to follow-up, have died, 
           or had an initial recurrence at a separate site are 
           censored. 
        2. Initial Hepatic Disease-Free Survival (Initial HDFS)
           Interval from hepatic resection until initial hepatic
           recurrence or death. Patients who are lost to follow-up
           or had an initial recurrence at a separate site are 
           censored. 
        3. Hepatic Disease-Free Time (HDF) 
           Interval from hepatic resection until hepatic recurrence
           at any time. Patients who are lost to follow-up, have died,
           or experienced a recurrence at a separate site are 
           censored.
        4. Hepatic Disease-Free Survival (HDFS) 
           Interval from hepatic resection until hepatic recurrence
           at any time or death. Patients who are lost to follow-up 
           or experienced a recurrence at a separate site are 
           censored. 
        5. Disease-Free Time (DF) 
           Interval from hepatic resection until recurrence at 
           any site. Patients who are lost to follow-up or have 
           died are censored. 
        6. Disease-Free Survival (DFS) 
           Interval from hepatic resection until recurrence at 
           any site or death. Patients who are lost to follow-up
           are censored. 
        7. Overall Survival (OS) 
           Interval from hepatic resection until death from any 
           cause. Patients who are lost to follow-up are
           censored. 

    Parameters 
    -----------
    clin_df: pandas.DataFrame 
        DataFrame containing clinical variables necessary to 
        calculate survival time and generate event/censor labels
    time_unit: str 
        Will convert days to months if "Months" is entered. If not,
        time will be calculated in days. 

    Returns 
    ----------
    surv_clin_df: pandas.DataFrame 
        clin_df with added columns for the event/censor indicators
        and survival times. 
    '''
        #Initialize new columns 
    clin_df["Resection Start Date"] = None
    clin_df["Surv Time Censor 1"] = None
    clin_df["Surv Time Censor 2"] = None 
    clin_df["Surv Time Censor 3"] = None 
    clin_df["Surv Time Censor 4"] = None
    clin_df["Surv Time Censor 5"] = None
    clin_df["Surv Time Censor 6"] = None
    clin_df["Surv Time Censor 7"] = None

    #Censor 1: First recurrence is in liver (excluding death as an event)
    clin_df["Censor 1: Initial HDF"] = clin_df["Liver Recurrence"]   

    for idx in range(clin_df.shape[0]):
        if clin_df["Two-Stage Resection?"][idx] == 0: 
            clin_df["Resection Start Date"][idx] = clin_df["Single Stage Resection Date"][idx]
        elif clin_df["Two-Stage Resection?"][idx] == 1:
            clin_df["Resection Start Date"][idx] = clin_df["First Resection Date"][idx]

        if clin_df["Censor 1: Initial HDF"][idx] == 0:
            surv_date = clin_df["Follow-Up Date"][idx]
        elif clin_df["Censor 1: Initial HDF"][idx] == 1:
            surv_date = clin_df["Recurrence Date"][idx] 
        surv_time = surv_date - clin_df["Resection Start Date"][idx]
        if time_unit == "Months":
            surv_time = float(surv_time.days)/30.4167   
        clin_df["Surv Time Censor 1"][idx] = int(surv_time.days)   #Floors float  

    #Censor 2: First recurrence is in liver (including death as an event)(AKA Hepatic Disease-Free Survival)
    clin_df["Censor 2: Initial HDFS"] = clin_df["Dead Status"] + clin_df["Liver Recurrence"]

    clin_df = clin_df.replace({"Censor 2: Initial HDFS": 2}, 1) #If sum is 2, then patient is dead and had a recurrence, should be coded as an event
    clin_df = clin_df.replace({"Censor 2: Initial HDFS": 3}, 0) #If sum is 3, then patient was lost to follow up and did not have recurrence, should be coded as censored
    clin_df = clin_df.replace({"Censor 2: Initial HDFS": 4}, 1) #If sum is 4, then patient was lost to follow up but did have a recurrence, should be coded as an event

    #If patient is dead, follow up date is death date
    #If patient had no recurrence and is alive, the follow up date is their survival time
    #If patient is dead and had a recurrence, their recurrence date should still be the survival time
    #Therefore nothing changes between censor 1 and censor 2 survival times
    clin_df["Surv Time Censor 2"] = clin_df["Surv Time Censor 1"] 

    #Censor 3: Liver Recurrence at Anytime (excluding death as an event)
    clin_df["Censor 3: HDF"] = clin_df["Liver Recurrence Date (not first)"]
    clin_df = clin_df.replace({"Censor 3: HDF": pd.NaT}, 0)
    clin_df.loc[clin_df["Censor 3: HDF"] != 0, "Censor 3: HDF"] = 1

    for idx in range(clin_df.shape[0]):
        if clin_df["Censor 3: HDF"][idx] == 0 and clin_df["Censor 1: Initial HDF"][idx] == 0:
            surv_date = clin_df["Follow-Up Date"][idx]
        elif clin_df["Censor 3: HDF"][idx] == 1:
            surv_date = clin_df["Liver Recurrence Date (not first)"][idx]
        elif clin_df["Censor 1: Initial HDF"][idx] == 1:
            surv_date = clin_df["Recurrence Date"][idx]

        surv_time = surv_date - clin_df["Resection Start Date"][idx]
        if time_unit == "Months":
            surv_time = float(surv_time.days)/30.4167            
        clin_df["Surv Time Censor 3"][idx] = int(surv_time.days)

    clin_df["Censor 3: HDF"] = clin_df["Censor 3: HDF"] + clin_df["Censor 1: Initial HDF"] 
    clin_df = clin_df.replace({"Censor 3: HDF": 2}, 1) #If sum is 2, then patient had initial recurrence be in the liver, but also had an additional liver recurrence after that first recurrence. Should be coded as an event
    
    #Censor 4: Liver Recurrence at Anytime (including death)
    clin_df["Censor 4: HDFS"] = clin_df["Dead Status"] + clin_df["Censor 3: HDF"]

    clin_df["Surv Time Censor 4"] = clin_df["Surv Time Censor 3"] #See notes for censor 2 for explanation of why there would be equal

    clin_df = clin_df.replace({"Censor 4: HDFS": 2}, 1) #See notes for censor 2 for all of the explanations for these
    clin_df = clin_df.replace({"Censor 4: HDFS": 3}, 0) 
    clin_df = clin_df.replace({"Censor 4: HDFS": 4}, 1)

    #Censor 5: Any recurrence (excluding death as an event)
    clin_df["Censor 5: DF"] = clin_df["Recurrence?"]

    for idx in range(clin_df.shape[0]): 
        if clin_df["Censor 5: DF"][idx] == 0: 
            surv_date = clin_df["Follow-Up Date"][idx]
        else:
            surv_date = clin_df["Recurrence Date"][idx]

        surv_time = surv_date - clin_df["Resection Start Date"][idx]
        if time_unit == "Months":
            surv_time = float(surv_time.days)/30.4167            
        clin_df["Surv Time Censor 5"][idx] = int(surv_time.days)

    #Censor 6: Any recurrence (including death)
    clin_df["Censor 6: DFS"] = clin_df["Dead Status"] + clin_df["Censor 5: DF"]
    clin_df = clin_df.replace({"Censor 6: DFS": 2}, 1) #See notes for censor 2 for all of the explanations for these
    clin_df = clin_df.replace({"Censor 6: DFS": 3}, 0) 
    clin_df = clin_df.replace({"Censor 6: DFS": 4}, 1)

    clin_df["Surv Time Censor 6"] = clin_df["Surv Time Censor 5"]

    #Censor 7: Overall Survival 
    clin_df["Censor 7: OS"] = clin_df["Dead Status"]
    clin_df = clin_df.replace({"Censor 7: OS": 3}, 0) #If patient is lost to follow-up, code as censored

    for idx in range(clin_df.shape[0]):
        surv_date = clin_df["Follow-Up Date"][idx]
        surv_time = surv_date - clin_df["Resection Start Date"][idx]
        if time_unit == "Months":
            surv_time = float(surv_time.days)/30.4167
        clin_df["Surv Time Censor 7"][idx] = int(surv_time.days)

    return(clin_df)

def complete_clin_transform(old_format, new_format, ignore_list, bin_var_list, time_unit): 
    '''
    Combines the transform functions into one for easier use. 
    Parameters are the same as described in the base functions. 

    Returns
    --------
    surv_time_clin_df: pandas.DataFrame
        DataFrame with all transforms applied. 
    '''

    renamed_col_df = rename_columns(old_format, new_format, ignore_list)
    recoded_bin_df = recode_bin_vars(renamed_col_df, bin_var_list)
    surv_time_clin_df = add_events(recoded_bin_df, time_unit)

    return(surv_time_clin_df)

if __name__ == '__main__': 
    old_format_dir = "path/to/new_format_dir.xlsx"
    old_format_sheet =  "Old_Format"
    
    new_format_dir = "path/to/clin_data.xlsx"
    new_format_sheet = "New_Format"

    old_format_df = pd.read_excel(old_format_dir, 
                                  sheet_name = old_format_sheet)
    new_format_df = pd.read_excel(new_format_dir, 
                                  sheet_name = new_format_sheet)
    
    cols_to_drop = ["record_id", "mda_de_id"]
    bin_vars = ["Comorbidities?", 
                "Node Positive?", 
                "Synchronous Disease?", 
                "Neoadjuvant Chemo?", 
                "Neoadjuvant Chemo: Oxaliplatin", 
                "Neoadjuvant Chemo: Irinotecan", 
                "Neoadjuvant Chemo: 5-FU", 
                "Neoadjuvant Chemo: Biological", 
                "Neoadjuvant Chemo: Xeloda", 
                "Neoadjuvant Chemo: FUDR",
                "Neoadjuvant Chemo Pump?", 
                "Bilateral Metastases?", 
                "PV Embolization?", 
                "Dead Status", 
                "Recurrence?", 
                "Adrenal Recurrence", 
                "Bone Recurrence", 
                "Diaphragm Recurrence", 
                "Local Colon Recurrence", 
                "Lung Recurrence", 
                "Lymph Nodes Recurrence", 
                "Multiorgan Recurrence", 
                "Other Recurrence", 
                "Ovary Recurrence", 
                "Peritoneal Recurrence", 
                "Liver Recurrence"]
    
    reformatted_clin_data = complete_clin_transform(old_format_df, 
                                                    new_format_df, 
                                                    cols_to_drop, 
                                                    bin_vars,
                                                    time_unit = "Days")
    
    reformatted_clin_data.to_csv("Transformed_Clinical_Data.csv")