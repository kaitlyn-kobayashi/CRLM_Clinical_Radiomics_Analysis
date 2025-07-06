
install.packages("mRMRe")
install.packages("survival")

library(mRMRe)
library(survival)

### Directories ###
os_rad_uni <- "path/to/OS_rad_train_data.csv"
hdfs_rad_uni <- "path/to/HDFS_rad_train_data.csv"

os_cnn_uni <- "path/to/OS_cnn_train_data.csv"
hdfs_cnn_uni <- "path/to/HDFS_cnn_train_data.csv"

rad_train_out <- "path/to/rad_mRMR_train"

### List of filters to use ### 
rad_filt <- 'Liver_|Tumor_' # For radiomic features
cnn_filt <- 'LivWhole_' # For CNN extracted features, remember to change 
                        #output file name in below code when switching ROIs
#cnn_filt can also be LivPar_ or Tumor_ depending on the ROI used

### Running feature selection 
type <- 'rad' #change to be 'cnn' if you 
if(type == 'rad'){
  os_train <- read.csv(os_rad_uni)
  
  hdfs_train <- read.csv(hdfs_rad_uni)
  
  #For both tumour and liver features combined
  os_train_rad <- os_train[, grepl(rad_filt, colnames(os_train))]
  hdfs_train_rad <- hdfs_train[, grepl(rad_filt, colnames(hdfs_train))]
  
  os_surv <- Surv(event = os_train$OSCens, time = os_train$OSTime)
  hdfs_surv <- Surv(event = hdfs_train$HDFSCens, time = hdfs_train$HDFSTime)
  
  os_train_surv <- cbind(os_surv, os_train_rad)
  hdfs_train_surv <- cbind(hdfs_surv, hdfs_train_rad)
  
  ## Apply mRMR ##
  #Classic type
  feature_data_os = mRMR.data(data = os_train_surv)
  filter_os = mRMR.classic("mRMRe.Filter", data = feature_data_os, target_indices = c(1), feature_count = 25)
  col_indices_os <- c()
  for (num in solutions(filter_os)$"1"){
    col_indices_os <- c(col_indices_os, num)
  }
  cols_to_keep_os <- c("Patient", "OSTime", "OSCens", colnames(os_train_surv[, col_indices_os]))
  mRMR_train_os <- os_train[cols_to_keep_os]
  write.csv(mRMR_train_os, paste0(rad_train_out, "/mRMR_OS_rad_0.csv"))
  
  feature_data_hdfs = mRMR.data(data = hdfs_train_surv)
  filter_hdfs = mRMR.classic("mRMRe.Filter", data = feature_data_hdfs, target_indices = c(1), feature_count = 25)
  col_indices_hdfs <- c()
  for (num in solutions(filter_hdfs)$"1"){
    col_indices_hdfs <- c(col_indices_hdfs, num)
  }
  cols_to_keep_hdfs <- c("Patient", "HDFSTime", "HDFSCens", colnames(hdfs_train_surv[, col_indices_hdfs]))
  mRMR_train_hdfs <- hdfs_train[cols_to_keep_hdfs]
  write.csv(mRMR_train_hdfs, paste0(rad_train_out, "/mRMR_HDFS_rad_0.csv"))
} else if(type == 'cnn'){
  os_train <- read.csv(os_cnn_uni)
  hdfs_train <- read.csv(hdfs_cnn_uni)
  
  os_train_cnn <- os_train[, grepl(cnn_filt, colnames(os_train))]
  hdfs_train_cnn <- hdfs_train[, grepl(cnn_filt, colnames(hdfs_train))]
  
  os_surv <- Surv(event = os_train$OSCens, time = os_train$OSTime)
  hdfs_surv <- Surv(event = hdfs_train$HDFSCens, time = hdfs_train$HDFSTime)
  
  os_train_surv <- cbind(os_surv, os_train_cnn)
  hdfs_train_surv <- cbind(hdfs_surv, hdfs_train_cnn)
  
  ## Apply mRMR ##
  #Classic type
  feature_data_os = mRMR.data(data = os_train_surv)
  filter_os = mRMR.classic("mRMRe.Filter", data = feature_data_os, target_indices = c(1), feature_count = 25)
  col_indices_os <- c()
  for (num in solutions(filter_os)$"1"){
    col_indices_os <- c(col_indices_os, num)
  }
  cols_to_keep_os <- c("Patient", "OSTime", "OSCens", colnames(os_train_surv[, col_indices_os]))
  mRMR_train_os <- os_train[cols_to_keep_os]
  write.csv(mRMR_train_os, paste0(rad_train_out, "/mRMR_OS_cnnlivwhole_0.csv"))
  
  feature_data_hdfs = mRMR.data(data = hdfs_train_surv)
  filter_hdfs = mRMR.classic("mRMRe.Filter", data = feature_data_hdfs, target_indices = c(1), feature_count = 25)
  col_indices_hdfs <- c()
  for (num in solutions(filter_hdfs)$"1"){
    col_indices_hdfs <- c(col_indices_hdfs, num)
  }
  cols_to_keep_hdfs <- c("Patient", "HDFSTime", "HDFSCens", colnames(hdfs_train_surv[, col_indices_hdfs]))
  mRMR_train_hdfs <- hdfs_train[cols_to_keep_hdfs]
  write.csv(mRMR_train_hdfs, paste0(rad_train_out, "/mRMR_HDFS_cnnlivwhole_0.csv"))
}