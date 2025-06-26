library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2)
FILES <- list(
  unimodal = "data/All_Bootstrap_Results_2025-03-20.csv",
  multimodal = "data/Late_Fusion_All_Bootstrap_Results_2025-03-17.xlsx"
)

str_replace_many <- function (str, patterns) {
  for (k_v in patterns) {
    k <- k_v[0]
    v <- k_v[1]
    str <- str_replace(str, k, v)
  }
  return (str)
}

load_unimodal <- function () {
  read.csv(FILES$unimodal) %>%
    filter(ComBat==0) %>%
    mutate(feature_spec = plyr::revalue(factor(str_c(Feature_Type, ROI, sep="_"), 
                                         levels = c("clin_", "rad_", "cnn_Tumor", "cnn_LivPar", "cnn_LivWhole")),
                                  c("rad_"="Radiomics", "clin_"="Clinical",
                                    "cnn_LivPar"="CNN-LP",
                                    "cnn_LivWhole"="CNN-WL", 
                                    "cnn_Tumor" = "CNN-T")),
           Model_Type=plyr::revalue(factor(Model_Type, levels=c("cph", "ssvm_ker", "rsf", "deep_learning")), 
                              c("cph"="CPH", "ssvm_ker"="SSVM", "rsf"="RSF", "deep_learning"="DL")))
}

load_multimodal <- function () {
  multi <- readxl::read_excel(FILES$multimodal) %>%
    filter(ComBat=="FALSE")
  
  param_read = jsonlite::fromJSON(paste("[", str_flatten(multi$Parameters, collaps=", "), "]")) %>%
    unnest_wider(model_type, names_sep="_") %>%
    mutate(Imaging_Model_Type=plyr::revalue(factor(model_type_1, levels=c("cph", "ssvm_ker", "rsf", "deep")), 
                                      c("cph"="CPH", "ssvm_ker"="SSVM", "rsf"="RSF", "deep"="DL")),
           Clinical_Model_Type=plyr::revalue(factor(model_type_2, levels=c("cph", "ssvm_ker", "rsf", "deep")), 
                                       c("cph"="CPH", "ssvm_ker"="SSVM", "rsf"="RSF", "deep"="DL")),
           .keep = "none")
  cbind(multi, param_read) %>%
           mutate(Imaging_Feature_Type=plyr::revalue(factor(str_c(Imaging_Type, ROI, sep="_"), 
                                               levels = c("rad_LiverTumor", 
                                                          "cnn_Tumor", 
                                                          "cnn_LivPar", 
                                                          "cnn_LivWhole",
                                                          "deep_risk_Tumor", 
                                                          "deep_risk_LivPar", 
                                                          "deep_risk_LivWhole")),
                                        c("rad_LiverTumor"="Radiomics",
                                          "cnn_LivPar"="CNN-LP",
                                          "cnn_LivWhole"="CNN-WL", 
                                          "cnn_Tumor" = "CNN-T",
                                          "deep_risk_Tumor" = "CNN-T",
                                          "deep_risk_LivPar" = "CNN-LP",
                                          "deep_risk_LivWhole" = "CNN-WL"
                                          )),
                  Late_Model_Type=plyr::revalue(factor(Late_Model_Type, levels=c("cph", "ssvm_ker", "rsf", "deep")), 
                                                                   c("cph"="CPH", "ssvm_ker"="SSVM", "rsf"="RSF", "deep"="DL")),
                  Joint_Model_Type=factor(str_c(Imaging_Model_Type, Clinical_Model_Type, sep=", "),
                                          levels = c(
                                            "CPH, CPH",
                                            "CPH, SSVM", 
                                            "CPH, RSF",
                                            "SSVM, CPH",
                                            "SSVM, SSVM", 
                                            "SSVM, RSF",
                                            "RSF, CPH",
                                            "RSF, SSVM", 
                                            "RSF, RSF",
                                            "DL, CPH",
                                            "DL, SSVM", 
                                            "DL, RSF"
                                          ))
    )
}

library(viridis)
## OPTION 1
#VIRIDIS_MAP <- "viridis"
#RANGE_END <- 0.87
#RANGE_BEGIN <- 0
#DIRECTION <- 1
#COLOR_THRESH <- 0.59

## Option 2
#VIRIDIS_MAP <- "turbo"
#RANGE_END <- 0.45
#RANGE_BEGIN <- 1
#DIRECTION <- 1
#COLOR_THRESH <- 0.55

## Option 3
VIRIDIS_MAP <- "turbo"
RANGE_END <- 0.5
RANGE_BEGIN <- 0
DIRECTION <- 1
COLOR_THRESH <- 0.56
my_heat_map <- list(
  geom_tile(color = "white", lwd = 1.5, linetype = 1),
  scale_fill_viridis(limits=c(0.5, 0.64), option=VIRIDIS_MAP, begin=RANGE_BEGIN, end=RANGE_END, direction=DIRECTION),
  geom_text(aes(label = round(Test_Median, 2), color=ifelse(Test_Median < COLOR_THRESH, "white", "black")), 
            size = 3),
  scale_color_manual(values=c("white"="white", "black"="black"), guide="none"),
  theme_classic()
  #theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))
)

my_tiny_heat_map <- list(
  geom_tile(color = "white", lwd = 1.5, linetype = 1),
  scale_fill_viridis(limits=c(0.5, 0.64), option=VIRIDIS_MAP, begin=RANGE_BEGIN, end=RANGE_END, direction=DIRECTION),
  geom_text(aes(label = round(Test_Median, 2), color=ifelse(Test_Median < COLOR_THRESH, "white", "black")), 
            size = 3),
  scale_color_manual(values=c("white"="white", "black"="black"), guide="none"),
  theme_classic()#,
  #theme(axis.text.x = element_text(angle=90, vjust=0.5, hjust=1))
)