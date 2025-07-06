# univariate analysis for clinical variables

install.packages("naniar")
install.packages("readxl")
install.packages("survival")
install.packages("gtsummary")
install.packages("dplyr")

library(naniar)
library(readxl)
library(survival)
library(gtsummary)
library(dplyr)

# Load in clinical data 
clin_data <- read.csv("path/to/Clinical_Data_For_Imputation_And_Survival.csv")

# Clean up data for easier analysis in R
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

# Generate summaries of variables of interest for OS and HDFS
table_OS<-clin_data_to_use %>% filter(OS_TRAIN==1 & OS_CENS %in% c(0,1)) %>% 
  select(AGE_AT_LIVER_SURGERY,
        SEX,
        BMI,
        NODE_POSITIVE,
        SYNCHRONOUS_DISEASE,
        cea.200,                                                                           
        DFI_LESS_THAN_12_MONTHS,
        BILATERAL_METASTASES,
        EXTRAHEPATIC_DISEASE,
        PV_EMBOLIZATION,
        NUMBER_OF_CRLM,
        RIGHT_CRC_RESECTION,
        LEFT_CRC_RESECTION,
        TRANSVERSE_CRC_RESECTION,                                                                         
        DIAMETER_OF_LARGEST_CRLM
)

tbl_summary(table_OS)

table_HDFS<-clin_data_to_use%>% filter(HDFS_TRAIN==1 & HDFS_CENS %in% c(0,1)) %>% 
  select(AGE_AT_LIVER_SURGERY,
         SEX,
         BMI,
         NODE_POSITIVE,
         SYNCHRONOUS_DISEASE,
         cea.200,                                                                           
         DFI_LESS_THAN_12_MONTHS,
         BILATERAL_METASTASES,
         EXTRAHEPATIC_DISEASE,
         PV_EMBOLIZATION,
         NUMBER_OF_CRLM,
         RIGHT_CRC_RESECTION,
         LEFT_CRC_RESECTION,
         TRANSVERSE_CRC_RESECTION,                                                                          
         DIAMETER_OF_LARGEST_CRLM
)
tbl_summary(table_HDFS)

# Univariate analysis 
df.os.train <- clin_data_to_use %>% filter(OS_TRAIN==1 & OS_CENS %in% c(0,1))
OS <-
  tbl_uvregression(
    df.os.train[c('AGE_AT_LIVER_SURGERY',
                  'SEX',
                  'BMI',
                  'NODE_POSITIVE',
                  'RIGHT_CRC_RESECTION',
                  'LEFT_CRC_RESECTION',
                  'TRANSVERSE_CRC_RESECTION',
                  'SYNCHRONOUS_DISEASE',
                  'NUMBER_OF_CRLM',
                  'DIAMETER_OF_LARGEST_CRLM',
                  'cea.200',
                  'CEA_LEVELS_BEFORE_RESECTION',
                  'DFI_LESS_THAN_12_MONTHS',
                  'BILATERAL_METASTASES',
                  'EXTRAHEPATIC_DISEASE',
                  'PV_EMBOLIZATION',
                  'OS_TIME',
                  'OS_CENS'
    )],
    method = coxph,
    y = Surv(OS_TIME, OS_CENS),
    exponentiate = TRUE,
    pvalue_fun = function(x) style_pvalue(x, digits = 2)
  ) 
OS

df.hdfs.train <- clin_data_to_use %>% filter(HDFS_TRAIN == 1 & HDFS_CENS %in% c(0,1))

HDFS <- tbl_uvregression(
  df.hdfs.train[c('AGE_AT_LIVER_SURGERY',
                  'SEX',
                  'BMI',
                  'NODE_POSITIVE',
                  'RIGHT_CRC_RESECTION',
                  'LEFT_CRC_RESECTION',
                  'TRANSVERSE_CRC_RESECTION',
                  'SYNCHRONOUS_DISEASE',
                  'NUMBER_OF_CRLM',
                  'DIAMETER_OF_LARGEST_CRLM',
                  'cea.200',
                  'CEA_LEVELS_BEFORE_RESECTION',
                  'DFI_LESS_THAN_12_MONTHS',
                  'BILATERAL_METASTASES',
                  'EXTRAHEPATIC_DISEASE',
                  'PV_EMBOLIZATION',
                  'HDFS_TIME',
                  'HDFS_CENS'
  )],
  method = coxph,
  y = Surv(HDFS_TIME, HDFS_CENS),
  exponentiate = TRUE,
  pvalue_fun = function(x) style_pvalue(x, digits = 2)
) 
HDFS