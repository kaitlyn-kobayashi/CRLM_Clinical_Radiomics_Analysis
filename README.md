# Improving Preoperative Risk Stratification in Colorectal Liver Metastases: A Multi-Institutional Evaluation of Multimodal Prediction Models

This repository contains a pipeline for preprocessing clinical and radiomic data, training unimodal and multimodal models, and evaluating models for the purposes of predicting hepatic disease-free survival (HDFS) and overall survival (OS) following curative intent resection. 

## Installation and Requirements
```bash
git clone https://github.com/<user>/CRLM_Clinical_Radiomics_Analysis.git
pip install -r requirements.txt
```
R packages which are required to are listed at the top of each of the R scripts. 

## CRLM Clinicoradiomic Analysis

### Feature Extraction 
Please refer to this [README.md](https://github.com/kaitlyn-kobayashi/CRLM_Clinical_Radiomics_Analysis/blob/main/Feature_Extraction/README.md) within the Feature_Extraction folder for more information.

### Preprocessing 
Below details the preprocessing steps used to prepare the clinical and radiomic feature data for further modelling. **Note that these steps were for the unimodal statistical and machine learning models and the multimodal late-fusion models. Different preprocessing was used for the deep learning models based on the architecture and computational limitations. For more information, please refer to the scripts within the [Deep_Learning](https://github.com/kaitlyn-kobayashi/CRLM_Clinical_Radiomics_Analysis/tree/main/Deep_Learning) folder**

Preprocessing includes: 
* *clin_MICE.R* --> Prepares clinical data for and executes Multivariate Imputation by Chained Equations (MICE) to handle clinical features with missing values. 
* *clin_train_test_bias.R* --> Utilizes 2-sample t-tests and chi-squared tests to see if there are significant differences in clinical variables in the training and testing split. 
* *preprocess_clin_radiomic_data* --> Calculates additional clinical features from base data, preprocesses radiomic features (checking for uniqueness, normalizing features, clamping features based on quantile range) from all regions of interest, and exports the relevant train and test information for downstream modelling.
* *train_test_splitting.py* --> Creates a train-test split stratified by the binary survival censor.
* *transform_clin_data.py* --> A script created specific to our dataset which calculates survival times and defines survival censors, defines variable names to be more readable, and recodes binary variables. 

### Feature Selection
All feature selection methods were used on the training splits only.
* *clin_univariate_analysis.R* --> For each of the outcomes of interest, performs univariate Cox regression for the clinical variables.
* *rad_MRMR.R* --> Performs feature selection for radiomic features using minimum redundancy maximum relevance (mRMR)

### Survival Modelling - Deep Learning
Automated 3-D deep-learning pipeline for **liver-only colorectal cancer liver metastasis (CRLM)** survival prediction.

![GPU](https://img.shields.io/badge/GPU-RTX%20A6000-77B900?logo=nvidia&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.4-1482C5?logo=nvidia)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-E34F26?logo=pytorch&logoColor=white)

The repository provides six runnable scripts—one per region-of-interest (ROI) × endpoint pair:

| ROI (3-D mask)   | Endpoint | Script               |
|------------------|---------|----------------------|
| **Whole Liver**  | Overall Survival (OS) | `Whole_Liver_OS.py` |
|                  | Hepatic DFS (HDFS)   | `Whole_Liver_HDFS.py` |
| **Liver Parenchyma** | OS    | `Liver_OS.py` |
|                      | HDFS  | `Liver_HDFS.py` |
| **Tumor Volume** | OS    | `Tumor_OS.py` |
|                  | HDFS  | `Tumor_HDFS.py` |

---

#### Technical Specs ⚙️

* 🖥️ Single **NVIDIA RTX A6000** (driver `550.90.07`, CUDA `12.4`)  
* 🔥 **PyTorch 2.1** with Automatic Mixed Precision (AMP)

All experiments were executed on this hardware configuration.

---

#### Installation
```bash
git clone https://github.com/<user>/CRLM_Clinical_Radiomics_Analysis.git
cd CRLM_Clinical_Radiomics_Analysis/Deep_Learning

# Conda example
conda create -n crlm_dl python=3.10
conda activate crlm_dl
pip install -r requirements.txt
```
#### Software Requirements
- `torch==2.1`
- `torchvision==0.16`
- `scikit-learn>=1.4`
- `scikit-survival>=0.22`
- `numpy>=1.26`
- `pandas>=2.2`
- `nibabel>=5.2`
- `scipy>=1.13`
- `scikit-image>=0.23`
- `rich>=13.7`

### Survival Modelling - Statistical, Machine Learning, and Late Fusion
The scripts for training and evaluating these models require a specific file organization for configuration:
```
.
└── survival_late_fusion_data/
    ├── clin_train
    ├── clin_test
    ├── hyperparams
    ├── Late_Fusion_Param_Grids
    ├── late_hyperparams
    ├── rad_mRMR_train
    ├── rad_test
    └── ref_files
```

The folders under `survival_late_fusion_data` must have the same names as above for the scripts to find the correct data. Below are descriptions of data to put within each of these folders (all in csv format): 
* *clin_train* --> Training data for clinical-only models. Contains clinical variables after preprocessing and feature selection, survival times, and survival binary censor. 
* *clin_test* --> Testing data for clinical-only models. Contains clinical variables after preprocessing (before or after feature selection), survival times, and survival binary censor. 
* *hyperparams* --> Results of hyperparameter tuning for unimodal models. 
* *Late_Fusion_Param_Grids* --> Parameter grids for combinations of models and features to run in late fusion hyperparameter tuning. File contains only one column called `param_grid`, with each row containing a json-formatted parameter grid. 
* *late_hyperparams* --> Results of late fusion hyperparameter tuning. 
* *rad_mRMR_train* --> Training data for radiomic-only models. Contains radiomic variables after preprocessing and feature selection, survival times, and survival binary censor. 
* *rad_test* --> Testing data for radiomic-only models. Contains radiomic variables after preprocessing (before or after feature selection), survival times, and survival binary censor. 
* *ref_files* --> Contains training data before feature selection. Used to add back Patient ID after preprocessing and feature selection only. Assumes Patient ID order stays the same throughout preprocessing and feature selection. 

Both the hyperparameter tuning and survival modelling scripts make use of specifying substrings within the column headers of data and file names. This is used to ensure the proper variables and clinical endpoints are selected. For example, files containing specific survival endpoints (e.g. OS) should put `OS` in the names of the survival time and censor variables (e.g. `OSTime`) and in the name of the file (e.g. `name_of_file_OS.csv`). Similarly, radiomic variables and clinical variables should have their own unique identifying substring within their file names (e.g. `rad_OS_train.csv`). These substrings also correspond to how the features get filtered in runs in case multiple types of features exist in the same file. An example of this is in the `generate_run_params` function in `all_purpose_hp_tuning.py`.

An example setup of how to run hyperparameter tuning and model training and evaluation are within the `survmodel_example_setup.ipynb` notebook [here](https://github.com/kaitlyn-kobayashi/CRLM_Clinical_Radiomics_Analysis/blob/main/Survival_Modeling/survmodel_example_setup.ipynb).

For any questions or concerns regarding these specific scripts, please reach out to kaitlyn.kobayashi@queensu.ca. 
