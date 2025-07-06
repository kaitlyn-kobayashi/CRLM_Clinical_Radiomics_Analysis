# Calculating whether there are significant differences between train and test 
# splits for each of the clinical variables to assess bias

install.packages("table1")
install.packages("dplyr")

library(table1)
library(dplyr)

#Calculate p-value 
pvalue <- function(x, ...) {
  # Construct vectors of data y, and groups (strata) g
  y <- unlist(x)
  g <- factor(rep(1:length(x), times=sapply(x, length)))
  if (is.numeric(y)) {
    # For numeric variables, perform a standard 2-sample t-test
    p <- t.test(y ~ g)$p.value
  } else {
    # For categorical variables, perform a chi-squared test of independence
    p <- chisq.test(table(y, g))$p.value
  }
  # Format the p-value, using an HTML entity for the less-than sign.
  # The initial empty string places the output on the line below the variable label.
  c("", sub("<", "&lt;", format.pval(p, digits=3, eps=0.001)))
}

clin_data <- read.csv("path/to/npj_Digital_Medicine_Clinical_Data_FINAL_SPLIT.csv")

# Clean variables 
clin_data <- clin_data %>%
  janitor::clean_names(case = "all_caps") %>% # uppercase col names and get rid of spaces
  mutate(across(where(lubridate::is.POSIXt), lubridate::as_date))

# Calculate BMI, DFI and CEA
clin_data <- clin_data %>% mutate(
  BMI=WEIGHT_KG/(HEIGHT_M)^2,
  cea.200=ifelse(is.na(CEA_LEVELS_BEFORE_RESECTION),NA,
                 ifelse(CEA_LEVELS_BEFORE_RESECTION>200,'1','0'))
)

# Refactor all relevant categorical/binary variables 
clin_data$SEX <- factor(clin_data$SEX, levels = c(1, 2), labels = c("Male", "Female"))
clin_data$INSTITUTION <- factor(clin_data$INSTITUTION, levels = c(1,2), labels = c("MSK", "MDA"))
clin_data$NODE_POSITIVE <- factor(clin_data$NODE_POSITIVE, levels = c(1,0), labels = c("Yes", "No"))
clin_data$SYNCHRONOUS_DISEASE <- factor(clin_data$SYNCHRONOUS_DISEASE, levels = c(1,0), labels = c("Yes", "No"))
clin_data$NEOADJUVANT_CHEMO <- factor(clin_data$NEOADJUVANT_CHEMO, levels = c(1,0), labels = c("Yes", "No"))
clin_data$NEOADJUVANT_CHEMO_OXALIPLATIN <- factor(clin_data$NEOADJUVANT_CHEMO_OXALIPLATIN, levels = c(1,0), labels = c("Yes", "No"))
clin_data$NEOADJUVANT_CHEMO_IRINOTECAN <- factor(clin_data$NEOADJUVANT_CHEMO_IRINOTECAN, levels = c(1,0), labels = c("Yes", "No"))
clin_data$NEOADJUVANT_CHEMO_5_FU <- factor(clin_data$NEOADJUVANT_CHEMO_5_FU, levels = c(1,0), labels = c("Yes", "No"))
clin_data$NEOADJUVANT_CHEMO_BIOLOGICAL <- factor(clin_data$NEOADJUVANT_CHEMO_BIOLOGICAL, levels = c(1,0), labels = c("Yes", "No"))
clin_data$NEOADJUVANT_CHEMO_XELODA <- factor(clin_data$NEOADJUVANT_CHEMO_XELODA, levels = c(1,0), labels = c("Yes", "No"))
clin_data$NEOADJUVANT_CHEMO_FUDR <- factor(clin_data$NEOADJUVANT_CHEMO_FUDR, levels = c(1,0), labels = c("Yes", "No"))
clin_data$EXTRAHEPATIC_DISEASE <- factor(clin_data$EXTRAHEPATIC_DISEASE, levels = c(1,0), labels = c("Yes", "No"))
clin_data$BILATERAL_METASTASES <- factor(clin_data$BILATERAL_METASTASES, levels = c(1,0), labels = c("Yes", "No"))
clin_data$PV_EMBOLIZATION <- factor(clin_data$PV_EMBOLIZATION, levels = c(1,0), labels = c("Yes", "No"))
clin_data$CRC_RESECTION_SITE <- factor(clin_data$CRC_RESECTION_SITE, 
                                       levels = c(1,2,3,4,5), 
                                       labels = c("Right Colon", "Left Colon", "Transverse Colon", "Rectum", "Unknown"))
clin_data$CENSOR_7_OS <- factor(clin_data$CENSOR_7_OS, levels = c(1,0), labels = c("Yes", "No"))
clin_data$CENSOR_4_HDFS <- factor(clin_data$CENSOR_4_HDFS, levels = c(1,0), labels = c("Yes", "No"))
clin_data$OS_TRAIN <- factor(clin_data$OS_TRAIN, levels = c(1, 0), labels = c("OSTrain", "OSTest"))
clin_data$HDFS_TRAIN <- factor(clin_data$HDFS_TRAIN, levels = c(1, 0), labels = c("HDFSTrain", "HDFSTest"))

# Add labels 
label(clin_data$AGE_AT_LIVER_SURGERY) <- "Age at Liver Surgery"
label(clin_data$SEX) <- "Sex"
label(clin_data$BMI) <- "BMI"
label(clin_data$INSTITUTION) <- "Institution"
label(clin_data$NODE_POSITIVE) <- "Node Positive Primary"
label(clin_data$SYNCHRONOUS_DISEASE) <- "Synchronous Disease"
label(clin_data$NEOADJUVANT_CHEMO) <- "Neoadjuvant Chemotherapy"
label(clin_data$NEOADJUVANT_CHEMO_OXALIPLATIN) <- "Oxaliplatin"
label(clin_data$NEOADJUVANT_CHEMO_IRINOTECAN) <- "Irinotecan"
label(clin_data$NEOADJUVANT_CHEMO_5_FU) <- "5-FU"
label(clin_data$NEOADJUVANT_CHEMO_BIOLOGICAL) <- "Biological"
label(clin_data$NEOADJUVANT_CHEMO_XELODA) <- "Xeloda"
label(clin_data$NEOADJUVANT_CHEMO_FUDR) <- "FUDR"
label(clin_data$DIAMETER_OF_LARGEST_CRLM) <- "Diameter of Largest CRLM" 
label(clin_data$NUMBER_OF_CRLM) <- "Number of CRLM" 

units(clin_data$AGE_AT_LIVER_SURGERY) <- "years"
units(clin_data$BMI) <- "kg/m^2"
units(clin_data$DIAMETER_OF_LARGEST_CRLM) <- 'mm'

## QQ Plots for Continuous Variables (Test for normal distribution)
qqnorm(clin_data$AGE_AT_LIVER_SURGERY, pch = 1, frame = FALSE)
qqline(clin_data$AGE_AT_LIVER_SURGERY, col = 'green', lwd = 2)

qqnorm(clin_data$BMI, pch = 1, frame = FALSE)
qqline(clin_data$BMI, col = 'green', lwd = 2)

qqnorm(clin_data$DIAMETER_OF_LARGEST_CRLM, pch = 1, frame = FALSE)
qqline(clin_data$DIAMETER_OF_LARGEST_CRLM, col = 'green', lwd = 2)

#Axial diameter doesn't pass test for normality, performing Mann-Whitney U
wilcox.test(clin_data$DIAMETER_OF_LARGEST_CRLM ~ clin_data$OS_TRAIN, data = clin_data, exact = FALSE)

wilcox.test(clin_data$DIAMETER_OF_LARGEST_CRLM ~ clin_data$HDFS_TRAIN, data = clin_data, exact = FALSE)

#For OVERALL SURVIVAL 
table1(~AGE_AT_LIVER_SURGERY 
       + INSTITUTION + 
         SEX + 
         BMI + 
         NODE_POSITIVE + 
         SYNCHRONOUS_DISEASE + 
         NEOADJUVANT_CHEMO + 
         NEOADJUVANT_CHEMO_OXALIPLATIN +  
         NEOADJUVANT_CHEMO_IRINOTECAN + 
         NEOADJUVANT_CHEMO_5_FU + 
         NEOADJUVANT_CHEMO_BIOLOGICAL + 
         NEOADJUVANT_CHEMO_XELODA + 
         NEOADJUVANT_CHEMO_FUDR + 
         EXTRAHEPATIC_DISEASE + 
         BILATERAL_METASTASES + 
         DIAMETER_OF_LARGEST_CRLM + 
         NUMBER_OF_CRLM + 
         CENSOR_7_OS + 
         CENSOR_4_HDFS|
         OS_TRAIN, 
       data = clin_data, 
       overall = F, 
       extra.col = list('P-value' = pvalue))

# For HEPATIC-DISEASE FREE SURVIVAL
table1(~AGE_AT_LIVER_SURGERY 
       + INSTITUTION + 
         SEX + 
         BMI + 
         NODE_POSITIVE + 
         SYNCHRONOUS_DISEASE + 
         NEOADJUVANT_CHEMO + 
         NEOADJUVANT_CHEMO_OXALIPLATIN +  
         NEOADJUVANT_CHEMO_IRINOTECAN + 
         NEOADJUVANT_CHEMO_5_FU + 
         NEOADJUVANT_CHEMO_BIOLOGICAL + 
         NEOADJUVANT_CHEMO_XELODA + 
         NEOADJUVANT_CHEMO_FUDR + 
         EXTRAHEPATIC_DISEASE + 
         BILATERAL_METASTASES + 
         DIAMETER_OF_LARGEST_CRLM + 
         NUMBER_OF_CRLM + 
         CENSOR_7_OS + 
         CENSOR_4_HDFS|
         HDFS_TRAIN,
       data = clin_data, 
       overall = F, 
       extra.col = list('P-value' = pvalue))