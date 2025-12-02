# ==============================================================================
# training LASSO and Elastic Net Regression Models on PO-AKI Dataset using 
# different seeds for variance estimation and preprocessing selection
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


# code to install needed packages

# ----
#functions


# code to install needed packages

count_unavailable_packages <- function(vector_package_names){ 
  
  #'@title count packages that are unavailable
  #'@param vector_package_names is a vector containing the names of needed packages as string variables.
  #'@return the lenght of a vector containing the names of all needed packages that are not within the list of installed packages as integer value.
  
  return (length(setdiff(vector_package_names, rownames(installed.packages(lib.loc = "/project/ag-zacharias/software2/r-libs/4.2.2")))))  
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





seeded_grLASSO <- function(X, y, vector_of_seeds, group_vec){
  
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
    y_train <- ifelse(y_train == 1, 1, -1)# recode response to be numeric -1 and 1
    
    list_models[[name]] <- list()
    
    # store train indices
    list_models[[name]]$train_ids <- trainIndex 
    
    # train model and save to list
    list_models[[name]]$model <- cv.gglasso(x = X_train, 
                                      y = y_train, # cv.gglasso requires response variable to be numeric -1 and 1
                                      group = group_vec, 
                                      loss = "logit", 
                                      pred.loss = "misclass", # classification error set as loss function for cv error 
                                      intercept = TRUE, 
                                      nfolds = 5)
     
    
  }
  return(list_models)
}



seeded_ngrLASSO <- function(X, y, vector_of_seeds){
  
  list_models <- list()
  for(i in vector_of_seeds){
  
    name <- as.character(i)
    
    # split the data into train and test
    set.seed(i)
    trainIndex <- createDataPartition(cbind(y,X)[,1], p = .8,
                                      list = FALSE,
                                      times = 1)
    
    X_train <- X[ trainIndex,]
    y_train <- y[trainIndex]
    
list_models[[name]] <- list()
    
    # store train indices
    list_models[[name]]$train_ids <- trainIndex 
    
    # train model and save to list
      list_models[[name]]$model <- cv.glmnet(X_train, 
                                       as.factor(y_train),
                                       type.measure = "class",
                                       alpha = 1, 
                                       family= "binomial",
                                       nfolds = 5)
  }
  return(list_models)
}



seeded_Ridge <- function(X, y, vector_of_seeds){
  
  list_models <- list()
  for(i in vector_of_seeds){
    
    name <- as.character(i)
    
    # split the data into train and test
    set.seed(i)
    trainIndex <- createDataPartition(cbind(y,X)[,1], p = .8,
                                      list = FALSE,
                                      times = 1)
    
    X_train <- X[ trainIndex,]
    y_train <- y[trainIndex]
    
    list_models[[name]] <- list()
    
    # store train indices
    list_models[[name]]$train_ids <- trainIndex 
    
    # train model and save to list
    list_models[[name]]$model <- cv.glmnet(X_train, 
                                           as.factor(y_train),
                                           type.measure = "class",
                                           alpha = 0, 
                                           family= "binomial",
                                           nfolds = 5) 
  }
  return(list_models)
}



seeded_EN <- function(X, y, vector_of_seeds){
  
  list_models <- list()
  for(i in vector_of_seeds){
    
    name <- as.character(i)
    
    # split the data into train and test
    set.seed(i)
    trainIndex <- createDataPartition(cbind(y,X)[,1], p = .8,
                                      list = FALSE,
                                      times = 1)
    
    X_train <- X[ trainIndex,]
    y_train <- y[trainIndex]
    
    list_models[[name]] <- list()
    
    # store train indices
    list_models[[name]]$train_ids <- trainIndex 
    
    # train model and save to list
    list_models[[name]]$model <- cv.glmnet(X_train, 
                                           as.factor(y_train),
                                           type.measure = "class",
                                           alpha = 0.5, 
                                           family= "binomial",
                                           nfolds = 5) 
  }
  return(list_models)
}



# ----
# setup

#load packages

load_packages( c("tidyverse", "dplyr", "glmnet", "gglasso",  "fastDummies", "caret"))


#setwd("~/../Desktop/_masterthesis/data/")

set.seed(3010) #set initial seed for script
vector_of_seeds <- sample(1:50000, 100, replace = FALSE) # create vector of 100 random seeds for seeded model training










# ----
# main



# ==============================================================================
# loading datasets
# ==============================================================================



# ==========
# load data

devds_unbal <- readRDS("/your/path/devds_original_prep.rds")
devds_bal <- readRDS("/your/path/devds_balanced_prep.rds")






################################################################################

# ==============================================================================

#                             Unbalanced data set

# ==============================================================================

start_entire <- Sys.time() # Track start time

#_______________________________________________________________________________

# ------------------------------------------------------------------------------

#                               Preparation


print("Preparation started")

# ==========
# separate predictors (X) and outcome (y)
X <- devds_unbal %>% select(-c(AKI_bin))
y <- devds_unbal %>% select(c(AKI_bin))



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

cont <- X %>% select(all_of(continuousNames))
disc <- X %>% select(- all_of(continuousNames))

rm(continuousNames)



# ==========
# encoding discrete data in dummy variables

disc <- disc %>%
  mutate(across(where(is.numeric), as.factor)) # convert all numeric columns to factor
dummies <- fastDummies::dummy_cols(disc[, 2:5], remove_first_dummy = TRUE)
disc <- disc[,-c(2:5)] # remove multi-level features
disc <- cbind(disc, dummies[, 5:18])



# ==========
# combine and convert predictor dataframes (X_train, X_test) to matrix
X <- as.matrix(cbind(disc, cont))



# convert Matrix from character to numeric
X <- apply(X, 2 ,as.numeric)

str(X)
typeof(X)

rm(cont, disc, dummies)



# making sure response variable is vector format
y <- as.factor(y$AKI)

print("Preparation finished")

# ------------------------------------------------------------------------------

#                                 Training


# ==============================================================================
# group regression (LASSO)
# ==============================================================================

print("Unbalanced Group LASSO started")

# ==========
#create group index for X variables
v.group <- c(1:21, 
             22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 
             23, 23, 23, 
             24, 24, 24,
             25, 25, 25, 25, 
             26, 26, 26, 26, 
             27:52)
length(X[1 ,])==length(v.group)



# ==========
# Run 10-fold cross-validation & select lambda 100 times with different seeds

start_time <- Sys.time() # Track start time

#_______________________________________________________________________________

list_gr_lasso_unbalanced <- seeded_grLASSO(X, y, vector_of_seeds, v.group)

#__________________________  _____________________________________________________
end_time <- Sys.time() # Track end time
time_taken <- end_time - start_time # Calculate the difference
print(time_taken)# Print the time taken


#save results
saveRDS(list_gr_lasso_unbalanced, "/your/path/list_grpLASSO_unbal.rds")

print("Unbalanced Group LASSO finished")




# ==============================================================================
# non-group LASSO regression
# ==============================================================================


# ==========
#  10-fold cross validation and model training with different seeds

print("Unbalanced Non-Group LASSO started")
start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

list_ngr_lasso_unbalanced <- seeded_ngrLASSO(X, y, vector_of_seeds)

#_______________________________________________________________________________
end_time <- Sys.time() # Track end time
time_taken <- end_time - start_time # Calculate the difference
print(time_taken)# Print the time taken

# run duration with 10 seeds: 2.578623 mins



# save LASSO model
saveRDS(list_ngr_lasso_unbalanced, "/your/path/list_ngrpLASSO_unbal.rds")
print("Unbalanced Non-Group LASSO finished")


# ==============================================================================
# non group Ridge Regression
# ==============================================================================


print("Unbalanced Non-Group Ridge started")
start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

list_ngr_Ridge_unbalanced <- seeded_Ridge(X, y, vector_of_seeds)

#_______________________________________________________________________________
end_time <- Sys.time() # Track end time
time_taken <- end_time - start_time # Calculate the difference
print(time_taken)# Print the time taken





# save LASSO model
saveRDS(list_ngr_Ridge_unbalanced, "/your/path/list_ngrpRidge_unbal.rds")
print("Unbalanced Non-Group Ridge started")



# ==============================================================================
# non group Elastic Net Regression
# ==============================================================================



print("Unbalanced Non-Group Elastic Net started")
start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

list_ngr_EN_unbalanced <- seeded_EN(X, y, vector_of_seeds)

#_______________________________________________________________________________
end_time <- Sys.time() # Track end time
time_taken <- end_time - start_time # Calculate the difference
print(time_taken)# Print the time taken





# save LASSO model
saveRDS(list_ngr_EN_unbalanced, "/your/path/list_ngrpEN_unbal.rds")

print("Unbalanced Non-Group Elastic Net finished")














################################################################################

# ==============================================================================

#                         All transformed balanced

# ==============================================================================

print("Balanced:")

# ------------------------------------------------------------------------------

#                               Preparation

print("Preparation started")

# ==========
# separate predictors (X) and outcome (y)
X <- devds_bal %>% select(-c(AKI_bin))
y <- devds_bal %>% select(c(AKI_bin))



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

cont <- X %>% select(all_of(continuousNames))
disc <- X %>% select(- all_of(continuousNames))

rm(continuousNames)



# ==========
# encoding discrete data in dummy variables

disc <- disc %>%
  mutate(across(where(is.numeric), as.factor)) # convert all numeric columns to factor
dummies <- fastDummies::dummy_cols(disc[, 2:5], remove_first_dummy = TRUE)
disc <- disc[,-c(2:5)] # remove multi-level features
disc <- cbind(disc, dummies[, 5:18])



# ==========
# combine and convert predictor dataframes (X_train, X_test) to matrix
X <- as.matrix(cbind(disc, cont))



# convert Matrix from character to numeric
X <- apply(X, 2 ,as.numeric)

str(X)
typeof(X)

rm(cont, disc, dummies)



# converting response variable to vector format
y <- as.factor(y$AKI)

print("Preparation finished")

# ------------------------------------------------------------------------------

#                                 Training


# ==============================================================================
# group regression (LASSO)
# ==============================================================================

print("Balanced Group LASSO started")

# ==========
#create group index for X variables
length(X[1 ,])==length(v.group)


# ==========
# Run 10-fold cross-validation & select lambda 100 times with different seeds

start_time <- Sys.time() # Track start time

#_______________________________________________________________________________

list_gr_lasso_balanced <- seeded_grLASSO(X, y, vector_of_seeds, v.group)

#__________________________  _____________________________________________________
end_time <- Sys.time() # Track end time
time_taken <- end_time - start_time # Calculate the difference
print(time_taken)# Print the time taken


#save results
saveRDS(list_gr_lasso_balanced, "/your/path/list_grpLASSO_bal.rds")

print("Balanced Group LASSO finished")




# ==============================================================================
# non-group LASSO regression
# ==============================================================================


# ==========
#  10-fold cross validation and model training with different seeds

print("Balanced Non-Group LASSO started")
start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

list_ngr_lasso_balanced <- seeded_ngrLASSO(X, y, vector_of_seeds)

#_______________________________________________________________________________
end_time <- Sys.time() # Track end time
time_taken <- end_time - start_time # Calculate the difference
print(time_taken)# Print the time taken

# run duration with 10 seeds: 2.578623 mins



# save LASSO model
saveRDS(list_ngr_lasso_balanced, "/your/path/list_ngrpLASSO_bal.rds")
print("Balanced Non-Group LASSO finished")


# ==============================================================================
# non group Ridge Regression
# ==============================================================================


print("Balanced Non-Group Ridge started")
start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

list_ngr_Ridge_balanced <- seeded_Ridge(X, y, vector_of_seeds)

#_______________________________________________________________________________
end_time <- Sys.time() # Track end time
time_taken <- end_time - start_time # Calculate the difference
print(time_taken)# Print the time taken





# save LASSO model
saveRDS(list_ngr_Ridge_balanced, "/your/path/list_ngrpRidge_bal.rds")
print("Balanced Non-Group Ridge started")



# ==============================================================================
# non group Elastic Net Regression
# ==============================================================================



print("Balanced Non-Group Elastic Net started")
start_time <- Sys.time() # Track start time
#_______________________________________________________________________________

list_ngr_EN_balanced <- seeded_EN(X, y, vector_of_seeds)

#_______________________________________________________________________________
end_time <- Sys.time() # Track end time
time_taken <- end_time - start_time # Calculate the difference
print(time_taken)# Print the time taken





# save LASSO model
saveRDS(list_ngr_EN_balanced, "/your/path/list_ngrpEN_bal.rds")
print("Balanced Non-Group Elastic Net finished")








# ----
# end

#2.5  days
