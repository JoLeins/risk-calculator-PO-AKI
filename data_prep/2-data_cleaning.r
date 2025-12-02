# ==============================================================================
# Data analysis of the PO-AKI Dataset
# ==============================================================================


# MIT License
# Copyright (c) 2025 Jonas Leins and Hendrik Meyer
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


rm( list = ls())

# ----
# functions


# code to install needed packages

count_unavailable_packages <- function(vector_package_names){
  #'@title count packages that are unavailable
  #'@param vector_package_names is a vector containing the names of needed packages as string variables.
  #'@return the lenght of a vector containing the names of all needed packages that are not within the list of installed packages as integer value.
  return (length(setdiff(vector_package_names, rownames(installed.packages()))))
}

install_packages_if_unavailable <- function(vector_package_names){
  #'@title A function to install all packages that are not yet installed.
  #'@description The function calls the function count_unavailable_packages and pastes the
  #'input parameter into the install.packages() function if the output of the former isnt equal to 0.
  #'If after the call of install.packages the output of the function count_unavailable_packages still isnt
  #'equal to zero, this function stops and returns an error message containing not installed packages.
  #'@param vector_package_names is a vector containing the names of needed packages as string variables.
  if (count_unavailable_packages(vector_package_names) > 0) {
    install.packages(vector_package_names)
  }
  if (count_unavailable_packages(vector_package_names) > 0) {
    stop(paste0("Folgende Pakete konnten nicht installiert werden: ",
                setdiff(vector_package_names, rownames(installed.packages()))))
  }
}

load_packages <- function(vector_package_names){
  #'@title a function to load all required packages.
  #'@description The function calls the function install_packages_if_unavailable() to make
  #'sure that all required packages  (param) are installed, and then proceeds to loop through
  #'the list of packages, calling the library() function to load each element in the input vector.
  #'@param vector_package_names is a vector containing the names of needed packages as string variables.
  install_packages_if_unavailable(vector_package_names)
  for(pack in vector_package_names){
    library(pack,character.only = TRUE)
  }
}

summary_tab <- function(table, filename){
  column_names <- colnames(table)
  column_means <- c()
  column_sd <- c()
  column_min <- c()
  column_max <- c()
  column_content <- colSums(unlist(table != 0))/length(table[,1]) #create vector containing percentage of nonzero values for each column
  for( name in colnames(table)){
    #print(cont_data[, name])
    column_means <- append(column_means, mean(unlist(table[, name])))# insert mean for every column in empty mean vector
    column_sd <- append(column_sd, sd(unlist(table[, name])))# insert sd for every column in empty sd vector
    column_min <- append(column_min, min(unlist(table[, name])))# insert sd for every column in empty sd vector
    column_max <- append(column_max, max(unlist(table[, name])))# insert sd for every column in empty sd vector
  }
  
  cont_summary_table <- data.frame(column_names, column_means, column_sd, column_min, column_max, column_content)
  colnames(cont_summary_table) <- c("Feature", "Mean", "Sd", "Min", "Max", "Non-zero")
  rownames(cont_summary_table)<- NULL
  
  write.csv(cont_summary_table, filename , row.names = FALSE)
}

qq_and_hist <- function(table, filename){
  
  pdf(file = filename)
  par(mfrow = c(2, 2))
  
  for (name in colnames(table)) {
    # Plot histogram
    hist(table[, name], main = paste("Histogram of", name), xlab = name, col = "lightblue")
    # Plot QQ plot
    qqnorm(table[, name], main = paste("QQ Plot of", name))
    qqline(table[, name])
  }
  
  dev.off()
  
}

# Create a function to scale data using stored mean and standard deviation

scale_with_params <- function(data, mean_vals, sd_vals) {
  scaled_data <- scale(data, center = mean_vals, scale = sd_vals)
  return(scaled_data)
}





# ----
# setup

# load packages

load_packages( c("tidyverse","dplyr", "ggpubr", "ggplot2", "caret", "gtsummary", "writexl"))
setwd("/Your/Path/")


set.seed(3010)









# ----
# main

# ==============================================================================
# Data analysis of the original Dataset
# ==============================================================================

# load data [old specifies the main data frame, new is simply a preprocessed df containing
# the feature set we want. It is not nessecary to load such a dataframe, manual 
# creation and downstream insertion of a feature vector is sufficient]
data_old <- read.csv("/your/path/full_data_frame.csv") # read data from csv
data_new <- read.csv("/your/path/Dataset_PO-AKI_preprocessed.csv") # only for feature set
MAP <- read.csv("/your/path/MAP_Features.csv")# Output of "1-


# binarize medication variables
data_new$medEinlKateBolus_bin <- ifelse(data_new$medEinlKateBolus > 0, 1, 0)
data_new$medOpAtroBolus_bin <- ifelse(data_new$medOpAtroBolus > 0, 1, 0)
data_new$medOpKateBolus_bin <- ifelse(data_new$medOpKateBolus > 0, 1, 0)

# create binary variables of most important clinics

data_new$klinik_kat_ACH <- ifelse(data_new$klinik == "ACH", 1, 0)
data_new$klinik_kat_GCH <- ifelse(data_new$klinik == "GCH", 1, 0)
data_new$klinik_kat_HCH <- ifelse(data_new$klinik == "HCH", 1, 0)
data_new$klinik_kat_HNO <- ifelse(data_new$klinik == "HNO", 1, 0)
data_new$klinik_kat_NCH <- ifelse(data_new$klinik == "NCH", 1, 0)
data_new$klinik_kat_PWC <- ifelse(data_new$klinik == "PWC", 1, 0)
data_new$klinik_kat_TCH <- ifelse(data_new$klinik == "TCH", 1, 0)
data_new$klinik_kat_UCH <- ifelse(data_new$klinik == "UCH", 1, 0)
data_new$klinik_kat_URO <- ifelse(data_new$klinik == "URO", 1, 0)
data_new$klinik_kat_Andere <- ifelse(!data_new$klinik %in% c("ACH", "GCH", "HCH", "HNO", "NCH", "PWC", "TCH", "UCH", "URO"), 1, 0)

#check authenticity
clinics <- data_new[, 828:837]
sum(rowSums(clinics)==1)


# remove superfluos variables [For this you can also create a name vector 
# yourself or use predictor names from the models provided on gitHub]
MAP_filtered <- MAP %>% select(intersect(names(MAP), names(data_old)))
data_new_filtered <- data_new %>% select(intersect(names(data_new), names(data_old)))

# merge by ID variable X

data <- merge(data_new_filtered, MAP_filtered, by="X")

data <- data %>% select(-c("einlDauer.y","opDauer.y", "narkoseDauer.y"))
names(data)[names(data) == "einlDauer.x"] <- "einlDauer"
names(data)[names(data) == "opDauer.x"] <- "opDauer"
names(data)[names(data) == "narkoseDauer.x"] <- "narkoseDauer"



# binarize dosis variables and outcome

data$AKI_bin <- ifelse(data$AKI > 0, 1, 0)



dim(data)

#remove patients with bmi over 100
data <- data[data$bmi <= 100,]
dim(data) #4 samples removed

#remove patients with size below 100 cm
data <- data[data$groesse>= 100,]
dim(data) #2 patients removed

#remove patients with weight over 205 kg
data <- data[data$gewicht <= 205,]
dim(data) # 5 patients removed

# remove patients who are previously diagnosed with renal insufficiency (as well as the variable)
data <- data[data$nierenInsuff == 0 ,]
data <- data %>% select(-nierenInsuff)
dim(data) # 1595 patients and 1 variable removed 

# ====
# properly rename variables for english readers

new_names <- c("ID",
               "inductDurat",
               "surgDurat",
               "anestDurat",
               "age",
               "weight",
               "height",
               "bmi",
               "sex",
               "rcri",
               "urgency",
               "decrease",
               "asa",
               "nyha",
               "chf",
               "cad",
               "cvd",
               "pad",
               "diab",
               "diab_bin",
               "liverCirrh",
               "ekg",
               "ltmACE",
               "ltmSartan",
               "ltmBB",
               "ltmCCB",
               "ltmBig",
               "ltmIns",
               "ltmDiu",
               "ltmStat",
               "iuc", 
               "stomTube",
               "doseInduEtomidateBolus",
               "doseInduPropoBolus",
               "doseInduPropoPerfu",# perfusor zu syringe?
               "doseInduRemifPerfu",
               "doseInduSteroInf",
               "doseInduSufenBolus",
               "doseInduSufenPerfu",
               "doseInduThiopBolus",
               "doseSurgGelafInf",
               "doseSurgSteroInf",
               "maxSevoExp",
               "medSevoExp",
               "eGFRPreSurg",
               "AKI",
               "medInduKateBolus_bin",
               "medSurgAtroBolus_bin",
               "medSurgKateBolus_bin",
               "clinic_cat_ACH",
               "clinic_cat_GCH",
               "clinic_cat_HCH",
               "clinic_cat_HNO",
               "clinic_cat_NCH",
               "clinic_cat_PWC",
               "clinic_cat_TCH",
               "clinic_cat_UCH",
               "clinic_cat_URO",
               "clinic_cat_Other",
               "minMAPcumu1MinAnes",
               "minMAPcumu5MinAnes",
               "minMAPcumu1MinIndu",
               "minMAPcumu5MinIndu",
               "minMAPcumu1MinInduplus30",
               "minMAPcumu5MinInduplus30",
               "minMAPcumu1MinInduplus60",
               "minMAPcumu5MinInduplus60",
               "minMAPcumu1MinInduplus90",
               "minMAPcumu5MinInduplus90",
               "minMAPcumu1MinInduplus120",
               "minMAPcumu5MinInduplus120",
               "minMAPcumu1MinInduplus150",
               "minMAPcumu5MinInduplus150",
               "minMAPcumu1MinSurg",
               "minMAPcumu5MinSurg", 
               "aucMAPunder65Anes",
               "aucMAPunder65Indu",
               "aucMAPunder65Induplus30",
               "aucMAPunder65Induplus60",
               "aucMAPunder65Induplus90",
               "aucMAPunder65Induplus120",
               "aucMAPunder65Induplus150",
               "aucMAPunder65Surg",
               "twaMAPunder65Anes",
               "twaMAPunder65Indu",
               "twaMAPunder65Induplus30",
               "twaMAPunder65Induplus60",
               "twaMAPunder65Induplus90",
               "twaMAPunder65Induplus120",
               "twaMAPunder65Induplus150",
               "twaMAPunder65Surg",
               "meanAnes",
               "stdAnes",
               "entropyAnes",
               "trendAnes",
               "kurtosisAnes",
               "skewnessAnes",
               "meanIndu",
               "stdIndu",
               "meanInduplus30",
               "stdInduplus30",
               "meanInduplus60",
               "stdInduplus60",
               "meanInduplus90",
               "stdInduplus90",
               "meanInduplus120",
               "stdInduplus120",
               "meanInduplus150",
               "stdInduplus150",
               "mean_Surg",
               "std_Surg",
               "baselineMAP",
               "twaMAPdecrBL10Anes",
               "twaMAPdecrBL10Indu",
               "twaMAPdecrBL10Induplus30",
               "twaMAPdecrBL10Induplus60",
               "twaMAPdecrBL10Induplus90",
               "twaMAPdecrBL10Induplus120",
               "twaMAPdecrBL10Induplus150",
               "twaMAPdecrBL10Surg",
               "AKI_bin")

colnames(data) <- new_names


# ==============================================================================
# split and export
# ==============================================================================

# split dataset 50:50 into train and validation cohort
trainIndex <- createDataPartition(data$AKI, p = 2/3,
                                  list = FALSE,
                                  times = 1)
saveRDS(trainIndex, "/Your/Path/trainIndex.rds")

train <- data[ trainIndex,] # 10324 x 122
valid <- data[-trainIndex,] # 5161 x 122


# save split cohorts

saveRDS(data, "/Your/Path/full_dataset.rds")
saveRDS(train, "/Your/Path/training_cohort.rds")
saveRDS(valid, "/Your/Path/validation_cohort.rds")

write.csv2(data, "/Your/Path/full_dataset.csv")
write.csv2(train, "/Your/Path/training_cohort.csv")
write.csv2(valid, "/Your/Path/validation_cohort.csv")



# remove duplicate or irrelevant variables 

data <- subset(data, select = -c(ID, 
                                 #aucMAPunder65Anes,
                                 aucMAPunder65Indu,
                                 aucMAPunder65Induplus30,
                                 aucMAPunder65Induplus60,
                                 aucMAPunder65Induplus90,
                                 aucMAPunder65Induplus120,
                                 aucMAPunder65Induplus150,
                                 aucMAPunder65Surg,
                                 #twaMAPunder65Anes,
                                 twaMAPunder65Indu,
                                 twaMAPunder65Induplus30,
                                 twaMAPunder65Induplus60,
                                 twaMAPunder65Induplus90,
                                 twaMAPunder65Induplus120,
                                 twaMAPunder65Induplus150,
                                 twaMAPunder65Surg,
                                 #minMAPcumu1MinAnes,
                                 #minMAPcumu5MinAnes,
                                 minMAPcumu1MinIndu,
                                 minMAPcumu5MinIndu,
                                 minMAPcumu1MinInduplus30,
                                 minMAPcumu5MinInduplus30,
                                 minMAPcumu1MinInduplus60,
                                 minMAPcumu5MinInduplus60,
                                 minMAPcumu1MinInduplus90,
                                 minMAPcumu5MinInduplus90,
                                 minMAPcumu1MinInduplus120,
                                 minMAPcumu5MinInduplus120,
                                 minMAPcumu1MinInduplus150,
                                 minMAPcumu5MinInduplus150,
                                 minMAPcumu1MinSurg,
                                 minMAPcumu5MinSurg,
                                 #meanAnes,
                                 #stdAnes,
                                 meanIndu,
                                 stdIndu,
                                 meanInduplus30,
                                 stdInduplus30,
                                 meanInduplus60,
                                 stdInduplus60,
                                 meanInduplus90,
                                 stdInduplus90,
                                 meanInduplus120,
                                 stdInduplus120,
                                 meanInduplus150,
                                 stdInduplus150,
                                 mean_Surg,
                                 std_Surg,
                                 #baselineMAP,
                                 twaMAPdecrBL10Anes,
                                 twaMAPdecrBL10Indu,
                                 twaMAPdecrBL10Induplus30,
                                 twaMAPdecrBL10Induplus60,
                                 twaMAPdecrBL10Induplus90,
                                 twaMAPdecrBL10Induplus120,
                                 twaMAPdecrBL10Induplus150,
                                 twaMAPdecrBL10Surg,
                                 AKI
                                 ))


# ==============================================================================
# split and export
# ==============================================================================


train <- data[ trainIndex,] # 9260 x 69
valid <- data[-trainIndex,] # 4630 x 69


# save split cohorts

saveRDS(data, "/Your/Path/reduced_full_dataset.rds")
saveRDS(train, "/Your/Path/reduced_training_cohort.rds")
saveRDS(valid, "/Your/Path/reduced_validation_cohort.rds")

write.csv2(data, "/Your/Path/reduced_full_dataset.csv")
write.csv2(train, "/Your/Path/reduced_training_cohort.csv")
write.csv2(valid, "/Your/Path/reduced_validation_cohort.csv")
# ----
#end