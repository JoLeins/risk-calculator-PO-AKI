# ==============================================================================
# Training Extreme Gradient boosting Models on PO-AKI Dataset using different seeds for 
# variance estimation and preprocessing selection [[Part 1](Split for parallel submission)]
# ==============================================================================

# MIT License
# Copyright (c) 2025 Jonas Leins
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
#functions


# ----
#functions


# code to install needed packages

count_unavailable_packages <- function(vector_package_names){ 
  
  #'@title count packages that are unavailable
  #'@param vector_package_names is a vector containing the names of needed packages as string variables.
  #'@return the lenght of a vector containing the names of all needed packages that are not within the list of installed packages as integer value.
  
  return (length(setdiff(vector_package_names, rownames(installed.packages(lib.loc = "/your/path/")))))  
}


install_packages_if_unavailable <- function(vector_package_names){
  
  #'@title A function to install all packages that are not yet installed.
  #'@description The function calls the function count_unavailable_packages and pastes the
  #'input parameter into the install.packages() function if the output of the former isnt equal to 0.
  #'If after the call of install.packages the output of the function count_unavailable_packages still isnt 
  #'equal to zero, this function stops and returns an error message containing not installed packages.
  #'@param vector_package_names is a vector containing the names of needed packages as string variables.
  
  if (count_unavailable_packages(vector_package_names) > 0) {
    install.packages(vector_package_names, lib = "/your/path/", repos="https://ftp.gwdg.de/pub/misc/cran/")
  }
  if (count_unavailable_packages(vector_package_names) > 0) {
    
    stop(paste0("Folgende Pakete konnten nicht installiert werden: ",
                setdiff(vector_package_names, rownames(installed.packages(lib.loc = "/your/path/")))))
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
    library(pack,character.only = TRUE, lib.loc = "/your/path/")
  }
}

seeded_xGBM_split <- function(X, y, vector_of_seeds, control, treeparams, linparams){
  
  list_models <- list()
  for(i in vector_of_seeds){
    name <- as.character(i)
    
    # split the data into train and test
    set.seed(i)
    trainIndex <- createDataPartition(cbind(y,X)[,1], p = .8,
                                      list = FALSE,
                                      times = 1)
    
    X_train <- X[trainIndex,]
    y_train <- y[trainIndex]
    
    
    list_models[[name]] <- list()
    
    # store train indices
    list_models[[name]]$train_ids <- trainIndex 
    
    # train models and save to list
    list_models[[name]]$linear <- train(X_train, 
                                        as.factor(y_train),
                                        trControl = control,
                                        tuneGrid = linparams,
                                        method = "xgbLinear",
                                        nthread = 16
    )
    
    list_models[[name]]$tree <- train(X_train, 
                                      as.factor(y_train),
                                      trControl = control,
                                      tuneGrid = treeparams,
                                      method = "xgbTree",
                                      nthread = 16
    )
    
  }
  return(list_models)
}








# ----
# setup

#load packages

load_packages( c("tidyverse","dplyr", "mltools","xgboost", "data.table", "caret", "tidymodels" ))




set.seed(3010)
vector_of_seeds <- sample(12501:25000, 25, replace = FALSE)








# ----
# main












# ==============================================================================
# loading and preparing dataset
# ==============================================================================



# ==========
# load data
devds_unbal <- readRDS("/your/path/devds_original_prep.rds")
devds_bal <- readRDS("/your/path/devds_balanced_prep.rds")


print("Preparation started")

# ==========
# separate predictors (X) and outcome (y)
X_unbal <- devds_unbal %>% select(-c(AKI_bin))
y_unbal <- devds_unbal$AKI_bin
X_bal <- devds_bal %>% select(-c(AKI_bin))
y_bal <- devds_bal$AKI_bin




# ==========
# separate continuous and discrete predictors

continuousNames <- c("baselineMAP",
                     "inductDurat",
                     "anestDurat",
                     "age",
                     "height",
                     "bmi",
                     "doseInduEtomidateBolus",
                     "doseInduPropoBolus",
                     "doseInduPropoPerfu",
                     "doseInduRemifPerfu",
                     "doseInduSufenBolus",
                     "doseInduSufenPerfu",
                     "doseInduThiopBolus",
                     "doseSurgGelafInf",
                     "doseSurgSteroInf",
                     "maxSevoExp",
                     "eGFRPreSurg",
                     "minMAPcumu1MinAnes",
                     "minMAPcumu5MinAnes",
                     "aucMAPunder65Anes",
                     "meanAnes",              
                     "stdAnes",                
                     "entropyAnes",            
                     "trendAnes",             
                     "kurtosisAnes",           
                     "skewnessAnes") # vector containing the names of all continuous features

cont_unbal <- X_unbal %>% select(all_of(continuousNames))
disc_unbal <- X_unbal %>% select(- c(all_of(continuousNames)))
cont_bal <- X_bal %>% select(all_of(continuousNames))
disc_bal <- X_bal %>% select(- c(all_of(continuousNames)))


rm(continuousNames)



# ==========
# encoding discrete data in dummy variables
dummies <- fastDummies::dummy_cols(disc_unbal[, 2:5], remove_first_dummy = TRUE)
disc_unbal <- disc_unbal[,-c(2:5)] # remove multi-level features
disc_unbal <- cbind(disc_unbal, dummies[, 5:18])
dummies <- fastDummies::dummy_cols(disc_bal[, 2:5], remove_first_dummy = TRUE)
disc_bal <- disc_bal[,-c(2:5)] # remove multi-level features
disc_bal <- cbind(disc_bal, dummies[, 5:18])



# ==========
# convert predictor dataframes (X_train, X_test) to matrix
X_unbal <- as.matrix(cbind(disc_unbal, cont_unbal))
X_bal <- as.matrix(cbind(disc_bal, cont_bal))




# convert Matrix from character to numeric
X_unbal <- apply(X_unbal, 2 ,as.numeric)
X_bal <- apply(X_bal, 2 ,as.numeric)




rm(cont_unbal, disc_unbal, cont_bal, disc_bal)

print("Preparation finished")









################################################################################

#                           HP tuning and Training

################################################################################ 




# specify cross validation methods
xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE
)


# specify hyperparametergrids
xgbTREEGrid <- expand.grid(nrounds = c(50, 100, 200, 300),  # number of boosting rounds
                       max_depth = c(10, 15, 25), # maximum depth of a tree # 15, 20
                       colsample_bytree = seq(0.4, 0.95, length.out = 9), # subsample ratio of columns when constructing each tree
                       eta = c(0.1, 0.01), # learning rate
                       gamma= 0, # minimum loss reduction  !!!!!!!!!!!START WITH 0 , IF PERFORMANCE ISNT GOOD, TRY ANOTHER! 
                       min_child_weight = 1, # minimum sum of instance weight (hessian) needed ina child
                       subsample = 1 # subsample ratio of the training instances
)


xgbLINGrid <- expand.grid(nrounds = c(50, 100, 200, 300),  # number of boosting rounds
                          lambda = seq(0.01, 1, length.out = 6), # L2 regularization
                          alpha = seq(0.01, 1, length.out = 6), # L1 regularization 
                          eta = c(0.1, 0.01) # learning rate
)






# ==============================================================================



# ==========
# train models


# Unbalanced
print("Unbalanced SVM started")
start <- Sys.time()
XGBM_unbalanced <- seeded_xGBM_split(X_unbal, y_unbal, vector_of_seeds, xgb_trcontrol, xgbTREEGrid, xgbLINGrid)
end <- Sys.time()

end - start

saveRDS(XGBM_unbalanced, "/your/path/list_XGBM_unbal_2.rds")

print("Unbalanced SVM finished")

# Balanced
print("Balanced SVM started")
start <- Sys.time()
XGBM_balanced <- seeded_xGBM_split(X_bal, y_bal, vector_of_seeds, xgb_trcontrol, xgbTREEGrid, xgbLINGrid)
end <- Sys.time()
atb <- end- start
print(paste("The training process took ", end-start, "time units"))

saveRDS(XGBM_balanced, "/your/path/list_XGBM_bal_2.rds")

print("Balanced SVM finished")


# ----
# end