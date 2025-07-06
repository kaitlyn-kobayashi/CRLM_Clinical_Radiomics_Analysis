install.packages("mice")
install.packages("naniar")
install.packages("readxl")
install.packages("survival")
install.packages("gtsummary")
install.packages("dplyr")

library(mice)
library(naniar)
library(readxl)
library(survival)
library(gtsummary)
library(dplyr)

# Load in clinical data 
clin_data <- read.csv("path/to/Clinical_Data_For_Imputation_And_Survival.csv")

# Clean up data for easier analysis in R (same as univariate analysis)
clin_data <- clin_data %>%
  replace_with_na(replace = list('Comorbidities?' = 999))

clin_data <- clin_data %>%
  replace_with_na(replace = list('CRC Resection Site' = 5))

clin_data <- clin_data %>%
  replace_with_na_at(.vars = c('KRAS Mutation?', 
                               'Microsatellite Instability?', 
                               'Steatosis?', 
                               'Steatohepatitis?', 
                               'Sinus Injury?', 
                               '% Necrosis',
                               '% Fibrosis', 
                               '% Acellular Mucin', 
                               '% Viability'), 
                     condition = ~.x == -999)

clin_data <- clin_data %>%
  replace_with_na_at(.vars= c('Microsatellite Instability?', 
                              'Steatosis?', 
                              'Steatohepatitis', 
                              'Sinus Injury?'),
                     condition = ~.x == 3)

na_strings <- c('unknown')
clin_data <- clin_data %>%
  replace_with_na_all(condition = ~.x %in% na_strings)

clin_data <- clin_data %>%
  janitor::clean_names(case = "all_caps") %>% # uppercase col names
  mutate(across(where(lubridate::is.POSIXt), lubridate::as_date))

# Calculate BMI and CEA > 200
clin_data <- clin_data %>% mutate(
  BMI=WEIGHT_KG/(HEIGHT_M)^2,
  cea.200=ifelse(is.na(CEA_LEVELS_BEFORE_RESECTION),NA,
                 ifelse(CEA_LEVELS_BEFORE_RESECTION>200,'1','0'))
)

# Subset data
clin_data_to_use <- clin_data[c('PATIENT',
                                'AGE_AT_LIVER_SURGERY', 
                                'SEX', 
                                'BMI',
                                'NODE_POSITIVE',
                                'SYNCHRONOUS_DISEASE',
                                'DFI_LESS_THAN_12_MONTHS',
                                'RIGHT_CRC_RESECTION',
                                'LEFT_CRC_RESECTION',
                                'TRANSVERSE_CRC_RESECTION',
                                'cea.200',
                                'CEA_LEVELS_BEFORE_RESECTION',
                                'NUMBER_OF_CRLM',
                                'DIAMETER_OF_LARGEST_CRLM',
                                'EXTRAHEPATIC_DISEASE',
                                'BILATERAL_METASTASES',
                                'PV_EMBOLIZATION',
                                'HDFS_CENS',
                                'HDFS_TIME',
                                'OS_CENS',
                                'OS_TIME',
                                'OS_TRAIN',
                                'HDFS_TRAIN'
)] 

vars_to_investigate <- c('AGE_AT_LIVER_SURGERY', 
                         'SEX', 
                         'BMI',
                         'NODE_POSITIVE',
                         'SYNCHRONOUS_DISEASE',
                         'DFI_LESS_THAN_12_MONTHS',
                         'RIGHT_CRC_RESECTION',
                         'LEFT_CRC_RESECTION',
                         'TRANSVERSE_CRC_RESECTION',
                         'cea.200',
                         'CEA_LEVELS_BEFORE_RESECTION',
                         'NUMBER_OF_CRLM',
                         'DIAMETER_OF_LARGEST_CRLM',
                         'EXTRAHEPATIC_DISEASE',
                         'BILATERAL_METASTASES',
                         'PV_EMBOLIZATION'
)
clin_data_to_use$NODE_POSITIVE <- factor(clin_data_to_use$NODE_POSITIVE,
                                         levels = c(0,1),
                                         labels= c(0, 1))
clin_data_to_use$cea.200 <- factor(clin_data_to_use$cea.200,
                                   levels = c(0,1),
                                   labels = c(0, 1))
clin_data_to_use$BILATERAL_METASTASES <- factor(clin_data_to_use$BILATERAL_METASTASES,
                                                levels = c(0, 1),
                                                labels = c(0, 1))

# MICE 
impute_clin_OS <- mice(clin_data_to_use[c(vars_to_investigate, 'OS_TEST')], 
                       ignore = clin_data_to_use$OS_TEST, 
                       method = "pmm", 
                       seed = 42)
imputed_OS <- complete(impute_clin_OS)
imputed_OS <- imputed_OS[c('AGE_AT_LIVER_SURGERY',
                           'NODE_POSITIVE',
                           'NUMBER_OF_CRLM',
                           'DIAMETER_OF_LARGEST_CRLM',
                           'DFI_LESS_THAN_12_MONTHS',
                           'BILATERAL_METASTASES',
                           'EXTRAHEPATIC_DISEASE',
                           'OS_TEST')] #Subsetting by what was significant in UV analysis

colnames(imputed_OS) <- paste("Clin", colnames(imputed_OS), sep = "_")
imputed_OS['Patient'] = clin_data_to_use['PATIENT']
imputed_OS['OSTime'] = clin_data_to_use['OS_TIME']
imputed_OS['OSCens'] = clin_data_to_use['OS_CENS']
imputed_OS_train <- subset(imputed_OS, Clin_OS_TEST == FALSE)
imputed_OS_train <- subset(imputed_OS_train, select = -c(Clin_OS_TEST))
imputed_OS_test <- subset(imputed_OS, Clin_OS_TEST == TRUE)
imputed_OS_test <- subset(imputed_OS_test, select = -c(Clin_OS_TEST))
write.csv(imputed_OS_train, 'path/to/clin_train/clin_OS_train_MICE_R.csv')
write.csv(imputed_OS_test, 'path/to/clin_test/clin_OS_test_MICE_R.csv')

impute_clin_HDFS <- mice(clin_data_to_use[c(vars_to_investigate, 'HDFS_TEST')], 
                         ignore = clin_data_to_use$HDFS_TEST,
                         method = 'pmm',
                         seed = 42)
imputed_HDFS <- complete(impute_clin_HDFS)
imputed_HDFS <- imputed_HDFS[c('NODE_POSITIVE',
                               'SYNCHRONOUS_DISEASE',
                               'NUMBER_OF_CRLM',
                               'DIAMETER_OF_LARGEST_CRLM',
                               'DFI_LESS_THAN_12_MONTHS',
                               'BILATERAL_METASTASES',
                               'EXTRAHEPATIC_DISEASE',
                               'HDFS_TEST')] #Subsetting by what is significant in UV analysis
colnames(imputed_HDFS) <- paste("Clin", colnames(imputed_HDFS), sep = "_")
imputed_HDFS['Patient'] = clin_data_to_use['PATIENT']
imputed_HDFS['HDFSTime'] = clin_data_to_use['HDFS_TIME']
imputed_HDFS['HDFSCens'] = clin_data_to_use['HDFS_CENS']
imputed_HDFS_train <- subset(imputed_HDFS, Clin_HDFS_TEST == FALSE)
imputed_HDFS_train <- subset(imputed_HDFS_train, select = -c(Clin_HDFS_TEST))
imputed_HDFS_test <- subset(imputed_HDFS, Clin_HDFS_TEST == TRUE)
imputed_HDFS_test <- subset(imputed_HDFS_test, select = -c(Clin_HDFS_TEST))
write.csv(imputed_HDFS_train, 'path/to/clin_train/clin_HDFS_train_MICE_R.csv')
write.csv(imputed_HDFS_test, 'path/to/clin_test/clin_HDFS_test_MICE_R.csv')