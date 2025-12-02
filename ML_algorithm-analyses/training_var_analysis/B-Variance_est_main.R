# ==============================================================================
# Estimation of Input based training variance in PO-AKI Prediction Models based 
# on 100-fold nested cross-validation on the development dataset.
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


rm(list = ls())

# functions
# ----



# ==========
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

calculate_metrics <- function(y_pred, y_pred_class, y_test, pos) {
  # NOTE THAT PRED AND y_test NEED TO HAVE THE SAME LEVELS
  # Calculate AUC
  yscore <- data.frame(y_pred)
  rdb <- cbind(as.factor(y_test),yscore)
  colnames(rdb) = c('y','yscore')
  auc <- roc_auc(rdb, y, yscore, event_level = "second")
  auc <- auc$.estimate
  
  # Calculate confusion matrix
  confusion <- confusionMatrix(table(y_pred_class, y_test), positive = pos)
  cm_perf <- as.numeric(confusion$byClass)
  Perf <- append(auc, cm_perf)
  
  # Return metrics
  return(Perf)
}





predict_classes_from_list_gr <- function(list_of_models, X){
  predicted_probs_list <- list()
  
  # Loop through each model and predict probabilities
  for (model_name in names(list_of_models)) {
    
    # get test set based on seed specific train ids
    trainIndex <- list_of_models[[model_name]]$train_ids
    X_test <- X[-trainIndex,]
    
    # Extract the model
    model <- list_of_models[[model_name]]$model$gglasso.fit
    lambda <- list_of_models[[model_name]]$model$lambda.min
    
    # Predict probabilities using the model
    y_pred <- predict(model, 
                      s = lambda,
                      newx = X_test, 
                      type = "link") # predict to obtain class probabiities
    
    y_pred_class <- predict(model,
                            s = lambda,
                            newx = X_test, 
                            type = "class") # re-predicting in order to obtain predicted classes
    
    
    # Store predicted probabilities in the list
    predicted_probs_list[[model_name]]$y_pred <- y_pred
    predicted_probs_list[[model_name]]$y_pred_class <- y_pred_class
  }
  return(predicted_probs_list)
}

predict_classes_from_list_ngr <- function(list_of_models, X){
  predicted_probs_list <- list()
  
  # Loop through each model and predict probabilities
  for (model_name in names(list_of_models)) {
    
    # get test set based on seed specific train ids
    trainIndex <- list_of_models[[model_name]]$train_ids
    X_test <- X[-trainIndex,]
    
    # Extract the model
    model <- list_of_models[[model_name]]$model$glmnet.fit
    lambda <- list_of_models[[model_name]]$model$lambda.min
    
    # Predict probabilities using the model
    y_pred <- predict(model, 
                      s = lambda,
                      newx = X_test, 
                      type = "response") # predict to obtain class probabilities
    
    y_pred_class <- predict(model,
                            s = lambda,
                            newx = X_test, 
                            type = "class") # re-predicting in order to obtain predicted classes
    
    
    # Store predicted probabilities in the list
    predicted_probs_list[[model_name]]$y_pred <- y_pred
    predicted_probs_list[[model_name]]$y_pred_class <- y_pred_class
  }
  return(predicted_probs_list)
}


predict_classes_from_list_RF <- function(list_of_models, X){
  predicted_probs_list <- list()
  
  # Loop through each model and predict probabilities
  for (model_name in names(list_of_models)) {
    
    # get test set based on seed specific train ids
    trainIndex <- list_of_models[[model_name]]$train_ids
    X_test <- X[-trainIndex,]
    
    # Extract the model
    model <- list_of_models[[model_name]]$model
    
    # Predict probabilities using the model
    y_pred <- predict(model, 
                      newdata = X_test, 
                      type = "prob") # predict to obtain class probabiities
    
    y_pred_class <- predict(model, 
                            newdata = X_test, 
                            type = "raw") # predict in order to obtain predicted classes
    
    
    # Store predicted probabilities in the list
    predicted_probs_list[[model_name]]$y_pred <- y_pred[,2]
    predicted_probs_list[[model_name]]$y_pred_class <- y_pred_class
  }
  return(predicted_probs_list)
}




predict_classes_from_list_SVM <- function(list_of_models, X){
  predicted_probs_list <- list()
  
  # Loop through each model and predict probabilities
  for (model_name in names(list_of_models)) {
    
    
    # get test set based on seed specific train ids
    trainIndex <- list_of_models[[model_name]]$train_ids
    X_test <- X[-trainIndex,]
    
    # Extract the model
    model <- list_of_models[[model_name]]$tune
    
    
    # Predict probabilities using the model
    y_pred_class <- predict(model, 
                            X_test,
                            type = "raw")
    
    y_pred_prob <- predict(model, 
                           X_test,
                           type = "prob")
    
    # Store predicted probabilities in the list
    
    predicted_probs_list[[model_name]]$y_pred_class <- y_pred_class
    predicted_probs_list[[model_name]]$y_pred_prob <- y_pred_prob
  }
  return(predicted_probs_list)
}


predict_classes_from_list_xGBM <- function(list_of_models, X){
  predicted_probs_list <- list()
  
  # Loop through each model and predict probabilities
  for (model_name in names(list_of_models)) {
    
    # get test set based on seed specific train ids
    trainIndex <- list_of_models[[model_name]]$train_ids
    X_test <- X[-trainIndex,]
    
    # Extract the model
    model_lin <- list_of_models[[model_name]]$linear
    model_tree <- list_of_models[[model_name]]$tree
    
    # Predict probabilities using the model
    y_pred_lin <- predict(model_lin, 
                          newdata = X_test, 
                          type = "prob") # predict to obtain class probabiities of linear model
    
    y_pred_tree <- predict(model_tree, 
                           newdata = X_test, 
                           type = "prob") # predict in order to obtain class probabilities of tree model
    
    
    # Store predicted probabilities in the list
    predicted_probs_list[[model_name]]$y_pred_linear <- y_pred_lin[, 2]
    predicted_probs_list[[model_name]]$y_pred_tree <- y_pred_tree[, 2]
  }
  return(predicted_probs_list)
}



# setup 
# ----

load_packages( c("tidyverse", "dplyr", "caret", "pROC", "tidymodels", "ranger", "kernlab", "gglasso", "glmnet" )) #"data.table",
set.seed(3010)
setwd("/your/path/")



# main
# ----


# load test data

ds_bal <- readRDS("/your/path/devds_balanced_prep.rds") 
ds_unbal <- readRDS("/your/path/devds_original_prep.rds")

# separate predictors and outcome
X_bal <- ds_bal %>% select(-c(AKI_bin))
y_bal <- ds_bal %>% select(c(AKI_bin))
X_unbal <- ds_unbal %>% select(-c(AKI_bin))
y_unbal <- ds_unbal %>% select(c(AKI_bin))


# Logistic Regression
grpLasso_bal <- read_rds("/your/path/list_grpLASSO_bal.rds")
grpLasso_unbal <- read_rds("/your/path/list_grpLASSO_unbal.rds")
Lasso_bal <- read_rds("/your/path/list_ngrpLASSO_bal.rds")
Lasso_unbal <- read_rds("/your/path/list_ngrpLASSO_unbal.rds")
Ridge_bal <- read_rds("/your/path/list_ngrpRidge_bal.rds")
Ridge_unbal <- read_rds("/your/path/list_ngrpRidge_unbal.rds")
Enet_bal <- read_rds("/your/path/list_ngrpEN_bal.rds")
Enet_unbal <- read_rds("/your/path/list_ngrpEN_unbal.rds")


# Random Forest
RF_bal1 <- read_rds("/your/path/RF_balanced_1.rds")
RF_bal2 <- readRDS("/your/path/RF_balanced_2.rds")
RF_bal <- append(RF_bal1, RF_bal2)
RF_unbal1 <- read_rds("/your/path/RF_unbalanced_1.rds")
RF_unbal2 <- readRDS("/your/path/RF_unbalanced_2.rds")
RF_unbal <- append(RF_unbal1, RF_unbal2)

rm(RF_bal1, RF_bal2, RF_unbal1, RF_unbal2)


# SVM
SVM_bal <- readRDS("/your/path/list_SVM_bal.rds")
SVM_unbal <- readRDS("/your/path/list_SVM_unbal.rds")


# extreme Gradient Boosting Machines
xGBM_bal1 <- readRDS("/your/path/list_XGBM_bal.rds")
xGBM_bal2 <- readRDS("/your/path/list_XGBM_bal_2.rds")
xGBM_bal3 <- readRDS("/your/path/list_XGBM_bal_3.rds")
xGBM_bal4 <- readRDS("/your/path/list_XGBM_bal_4.rds")
xGBM_bal <- c(xGBM_bal1, xGBM_bal2, xGBM_bal3, xGBM_bal4)
xGBM_unbal1 <- readRDS("/your/path/list_XGBM_unbal.rds")
xGBM_unbal2 <- readRDS("/your/path/list_XGBM_unbal_2.rds")
xGBM_unbal3 <- readRDS("/your/path/list_XGBM_unbal_3.rds")
xGBM_unbal4 <- readRDS("/your/path/list_XGBM_unbal_4.rds")
xGBM_unbal <- c(xGBM_unbal1, xGBM_unbal2, xGBM_unbal3, xGBM_unbal4)

rm(xGBM_bal1, xGBM_bal2, xGBM_bal3, xGBM_bal4, xGBM_unbal1, xGBM_unbal2, xGBM_unbal3, xGBM_unbal4)





# ______________________________________________________________________________

#                     Support Vector Machine

# ______________________________________________________________________________
# ----


# preparation

# ==========
# reencode levels for application of kernlab based train

levels(y_bal$AKI_bin) <- c("no_event", "event")
levels(y_unbal$AKI_bin) <- c("no_event", "event")




# Prediction and Performance estimation


# ==========
# predict classes with optimal model
# predict classes and class probabilities from listed models
pprobs_SVM_bal <- predict_classes_from_list_SVM(SVM_bal, X_bal)
pprobs_SVM_unbal <- predict_classes_from_list_SVM(SVM_unbal, X_unbal)
pprobs_SVM_unbal[61]<- NULL #remove due to failure

# ==========
# calculate Performances


# Balanced
# create dataframe with performance metrics
metrics_SVM_bal <- data.frame()

for (name in names(pprobs_SVM_bal)) {
  trainIndex <- SVM_bal[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  model_metr <- calculate_metrics(pprobs_SVM_bal[[name]]$y_pred_prob$event,
                                  pprobs_SVM_bal[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_SVM_bal <- rbind(metrics_SVM_bal, row) # add row to metrics_ATU dataframe
  
}
metrics_SVM_bal[,2:13] <- as.data.frame(lapply(metrics_SVM_bal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_SVM_bal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                           "Pos Pred Value", "Neg Pred Value", "Precision",
                           "Recall", "F1", "Prevalence", "Detection Rate", 
                           "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_SVM_bal <- rbind(metrics_SVM_bal, c("Mean", colMeans(metrics_SVM_bal[,2:13])))# add mean row
metrics_SVM_bal[,2:13] <- as.data.frame(lapply(metrics_SVM_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_SVM_bal <- rbind(metrics_SVM_bal,c("SD", sapply(metrics_SVM_bal[1:100,2:13],sd)))# add standard deviation row
metrics_SVM_bal[,2:13] <- as.data.frame(lapply(metrics_SVM_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_SVM_bal <- rbind(metrics_SVM_bal, c("Upper", sapply(metrics_SVM_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_SVM_bal[,2:13] <- as.data.frame(lapply(metrics_SVM_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_SVM_bal <- rbind(metrics_SVM_bal, c("Lower", sapply(metrics_SVM_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_SVM_bal[,2:13] <- as.data.frame(lapply(metrics_SVM_bal[,2:13], function(x) as.numeric(as.character(x))))




# save dataframe

write.csv2(metrics_SVM_bal, "/your/path/performance_SVM_bal.csv")


# unbalanced
# create dataframe with performance metrics
metrics_SVM_unbal <- data.frame()
for (name in names(pprobs_SVM_unbal)) {
  trainIndex <- SVM_unbal[[name]]$train_ids
  y_test <- as.factor(y_unbal$AKI_bin[-trainIndex])
  model_metr <- calculate_metrics(pprobs_SVM_unbal[[name]]$y_pred_prob$event,
                                  pprobs_SVM_unbal[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_SVM_unbal <- rbind(metrics_SVM_unbal, row) # add row to metrics_ATU dataframe
  
}
metrics_SVM_unbal[,2:13] <- as.data.frame(lapply(metrics_SVM_unbal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_SVM_unbal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                               "Pos Pred Value", "Neg Pred Value", "Precision",
                               "Recall", "F1", "Prevalence", "Detection Rate", 
                               "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_SVM_unbal <- rbind(metrics_SVM_unbal, c("Mean", colMeans(metrics_SVM_unbal[,2:13])))# add mean row
metrics_SVM_unbal[,2:13] <- as.data.frame(lapply(metrics_SVM_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_SVM_unbal <- rbind(metrics_SVM_unbal,c("SD", sapply(metrics_SVM_unbal[1:99,2:13],sd)))# add standard deviation row
metrics_SVM_unbal[,2:13] <- as.data.frame(lapply(metrics_SVM_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_SVM_unbal <- rbind(metrics_SVM_unbal, c("Upper", sapply(metrics_SVM_unbal[1:99, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_SVM_unbal[,2:13] <- as.data.frame(lapply(metrics_SVM_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_SVM_unbal <- rbind(metrics_SVM_unbal, c("Lower", sapply(metrics_SVM_unbal[1:99, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_SVM_unbal[,2:13] <- as.data.frame(lapply(metrics_SVM_unbal[,2:13], function(x) as.numeric(as.character(x))))




# save dataframe

write.csv2(metrics_SVM_unbal, "/your/path/performance_SVM_unbal.csv")





# ______________________________________________________________________________

#                           Random Forest

# ______________________________________________________________________________
# ----

# preparation

# ==========
# reencode levels for application of kernlab based train

levels(y_bal$AKI_bin) <- c("no_event", "event")
levels(y_unbal) <- c("no_event", "event")




# Prediction and Performance estimation


# ==========
# predict classes with optimal model
# predict classes and class probabilities from listed models
pprobs_RF_bal <- predict_classes_from_list_RF(RF_bal, X_bal)
pprobs_RF_unbal <- predict_classes_from_list_RF(RF_unbal, X_unbal)

# ==========
# calculate Performances


# Balanced
# create dataframe with performance metrics
metrics_RF_bal <- data.frame()

for (name in names(pprobs_RF_bal)) {
  trainIndex <- RF_bal[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  model_metr <- calculate_metrics(pprobs_RF_bal[[name]]$y_pred,
                                  pprobs_RF_bal[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_RF_bal <- rbind(metrics_RF_bal, row) # add row to metrics_ATU dataframe
}
metrics_RF_bal[,2:13] <- as.data.frame(lapply(metrics_RF_bal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_RF_bal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                               "Pos Pred Value", "Neg Pred Value", "Precision",
                               "Recall", "F1", "Prevalence", "Detection Rate", 
                               "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_RF_bal <- rbind(metrics_RF_bal, c("Mean", colMeans(metrics_RF_bal[,2:13])))# add mean row
metrics_RF_bal[,2:13] <- as.data.frame(lapply(metrics_RF_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_RF_bal <- rbind(metrics_RF_bal,c("SD", sapply(metrics_RF_bal[1:100,2:13],sd)))# add standard deviation row
metrics_RF_bal[,2:13] <- as.data.frame(lapply(metrics_RF_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_RF_bal <- rbind(metrics_RF_bal, c("Upper", sapply(metrics_RF_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_RF_bal[,2:13] <- as.data.frame(lapply(metrics_RF_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_RF_bal <- rbind(metrics_RF_bal, c("Lower", sapply(metrics_RF_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_RF_bal[,2:13] <- as.data.frame(lapply(metrics_RF_bal[,2:13], function(x) as.numeric(as.character(x))))




# save dataframe

write.csv2(metrics_RF_bal, "/your/path/performance_RF_bal.csv")


# unbalanced
# create dataframe with performance metrics
metrics_RF_unbal <- data.frame()

for (name in names(pprobs_RF_unbal)) {
  trainIndex <- RF_unbal[[name]]$train_ids
  y_test <- as.factor(y_unbal$AKI_bin[-trainIndex])
  model_metr <- calculate_metrics(pprobs_RF_unbal[[name]]$y_pred,
                                  pprobs_RF_unbal[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_RF_unbal <- rbind(metrics_RF_unbal, row) # add row to metrics_ATU dataframe
  
}

metrics_RF_unbal[,2:13] <- as.data.frame(lapply(metrics_RF_unbal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_RF_unbal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                 "Pos Pred Value", "Neg Pred Value", "Precision",
                                 "Recall", "F1", "Prevalence", "Detection Rate", 
                                 "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_RF_unbal <- rbind(metrics_RF_unbal, c("Mean", colMeans(metrics_RF_unbal[,2:13])))# add mean row
metrics_RF_unbal[,2:13] <- as.data.frame(lapply(metrics_RF_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_RF_unbal <- rbind(metrics_RF_unbal,c("SD", sapply(metrics_RF_unbal[1:100,2:13],sd)))# add standard deviation row
metrics_RF_unbal[,2:13] <- as.data.frame(lapply(metrics_RF_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_RF_unbal <- rbind(metrics_RF_unbal, c("Upper", sapply(metrics_RF_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_RF_unbal[,2:13] <- as.data.frame(lapply(metrics_RF_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_RF_unbal <- rbind(metrics_RF_unbal, c("Lower", sapply(metrics_RF_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_RF_unbal[,2:13] <- as.data.frame(lapply(metrics_RF_unbal[,2:13], function(x) as.numeric(as.character(x))))




# save dataframe

write.csv2(metrics_RF_unbal, "/your/path/performance_RF_unbal.csv")





# ______________________________________________________________________________

#                   extreme GRADIENT BOOSTING MACHINES

# ______________________________________________________________________________
# ----

# preparation



# ==========
# reverse previous reencoding

levels(y_bal$AKI_bin) <- c("0", "1")
levels(y_unbal$AKI_bin) <- c("0", "1")




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




# Prediction and Performance estimation


# ==========
# predict classes with optimal model
# predict classes and class probabilities from listed models
pprobs_xGBM_bal <- predict_classes_from_list_xGBM(xGBM_bal, X_bal)
pprobs_xGBM_unbal <- predict_classes_from_list_xGBM(xGBM_unbal, X_unbal)

# ==========
# calculate Performances


# Balanced Linear
# create dataframe with performance metrics
metrics_xGBMlin_bal <- data.frame()

for (name in names(pprobs_xGBM_bal)) {
  trainIndex <- xGBM_bal[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  model_metr <- calculate_metrics(pprobs_xGBM_bal[[name]]$y_pred_linear,
                                  as.factor(ifelse(pprobs_xGBM_bal[[name]]$y_pred_linear > 0.5, 1, 0)), 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_xGBMlin_bal <- rbind(metrics_xGBMlin_bal, row) # add row to metrics_ATU dataframe
}
metrics_xGBMlin_bal[,2:13] <- as.data.frame(lapply(metrics_xGBMlin_bal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_xGBMlin_bal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                              "Pos Pred Value", "Neg Pred Value", "Precision",
                              "Recall", "F1", "Prevalence", "Detection Rate", 
                              "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_xGBMlin_bal <- rbind(metrics_xGBMlin_bal, c("Mean", colMeans(metrics_xGBMlin_bal[,2:13])))# add mean row
metrics_xGBMlin_bal[,2:13] <- as.data.frame(lapply(metrics_xGBMlin_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMlin_bal <- rbind(metrics_xGBMlin_bal,c("SD", sapply(metrics_xGBMlin_bal[1:100,2:13],sd)))# add standard deviation row
metrics_xGBMlin_bal[,2:13] <- as.data.frame(lapply(metrics_xGBMlin_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMlin_bal <- rbind(metrics_xGBMlin_bal, c("Upper", sapply(metrics_xGBMlin_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_xGBMlin_bal[,2:13] <- as.data.frame(lapply(metrics_xGBMlin_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMlin_bal <- rbind(metrics_xGBMlin_bal, c("Lower", sapply(metrics_xGBMlin_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_xGBMlin_bal[,2:13] <- as.data.frame(lapply(metrics_xGBMlin_bal[,2:13], function(x) as.numeric(as.character(x))))




# save dataframe

write.csv2(metrics_xGBMlin_bal, "/your/path/performance_xGBMlin_bal.csv")


# Balanced Tree
# create dataframe with performance metrics
metrics_xGBMtree_bal <- data.frame()

for (name in names(pprobs_xGBM_bal)) {
  trainIndex <- xGBM_bal[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  model_metr <- calculate_metrics(pprobs_xGBM_bal[[name]]$y_pred_tree,
                                  as.factor(ifelse(pprobs_xGBM_bal[[name]]$y_pred_tree > 0.5, 1, 0)), 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_xGBMtree_bal <- rbind(metrics_xGBMtree_bal, row) # add row to metrics_ATU dataframe
}
metrics_xGBMtree_bal[,2:13] <- as.data.frame(lapply(metrics_xGBMtree_bal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_xGBMtree_bal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                   "Pos Pred Value", "Neg Pred Value", "Precision",
                                   "Recall", "F1", "Prevalence", "Detection Rate", 
                                   "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_xGBMtree_bal <- rbind(metrics_xGBMtree_bal, c("Mean", colMeans(metrics_xGBMtree_bal[,2:13])))# add mean row
metrics_xGBMtree_bal[,2:13] <- as.data.frame(lapply(metrics_xGBMtree_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMtree_bal <- rbind(metrics_xGBMtree_bal,c("SD", sapply(metrics_xGBMtree_bal[1:100,2:13],sd)))# add standard deviation row
metrics_xGBMtree_bal[,2:13] <- as.data.frame(lapply(metrics_xGBMtree_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMtree_bal <- rbind(metrics_xGBMtree_bal, c("Upper", sapply(metrics_xGBMtree_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_xGBMtree_bal[,2:13] <- as.data.frame(lapply(metrics_xGBMtree_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMtree_bal <- rbind(metrics_xGBMtree_bal, c("Lower", sapply(metrics_xGBMtree_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_xGBMtree_bal[,2:13] <- as.data.frame(lapply(metrics_xGBMtree_bal[,2:13], function(x) as.numeric(as.character(x))))




# save dataframe

write.csv2(metrics_xGBMtree_bal, "/your/path/performance_xGBMtree_bal.csv")



# ==========
# calculate Performances


# unbalanced Linear
# create dataframe with performance metrics
metrics_xGBMlin_unbal <- data.frame()

for (name in names(pprobs_xGBM_unbal)) {
  trainIndex <- xGBM_unbal[[name]]$train_ids
  y_test <- as.factor(y_unbal$AKI_bin[-trainIndex])
  model_metr <- calculate_metrics(pprobs_xGBM_unbal[[name]]$y_pred_linear,
                                  as.factor(ifelse(pprobs_xGBM_unbal[[name]]$y_pred_linear > 0.5, 1, 0)), 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_xGBMlin_unbal <- rbind(metrics_xGBMlin_unbal, row) # add row to metrics_ATU dataframe
}
metrics_xGBMlin_unbal[,2:13] <- as.data.frame(lapply(metrics_xGBMlin_unbal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_xGBMlin_unbal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                   "Pos Pred Value", "Neg Pred Value", "Precision",
                                   "Recall", "F1", "Prevalence", "Detection Rate", 
                                   "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_xGBMlin_unbal <- rbind(metrics_xGBMlin_unbal, c("Mean", colMeans(metrics_xGBMlin_unbal[,2:13])))# add mean row
metrics_xGBMlin_unbal[,2:13] <- as.data.frame(lapply(metrics_xGBMlin_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMlin_unbal <- rbind(metrics_xGBMlin_unbal,c("SD", sapply(metrics_xGBMlin_unbal[1:100,2:13],sd)))# add standard deviation row
metrics_xGBMlin_unbal[,2:13] <- as.data.frame(lapply(metrics_xGBMlin_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMlin_unbal <- rbind(metrics_xGBMlin_unbal, c("Upper", sapply(metrics_xGBMlin_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_xGBMlin_unbal[,2:13] <- as.data.frame(lapply(metrics_xGBMlin_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMlin_unbal <- rbind(metrics_xGBMlin_unbal, c("Lower", sapply(metrics_xGBMlin_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_xGBMlin_unbal[,2:13] <- as.data.frame(lapply(metrics_xGBMlin_unbal[,2:13], function(x) as.numeric(as.character(x))))




# save dataframe

write.csv2(metrics_xGBMlin_unbal, "/your/path/performance_xGBMlin_unbal.csv")


# unbalanced Tree
# create dataframe with performance metrics
metrics_xGBMtree_unbal <- data.frame()

for (name in names(pprobs_xGBM_unbal)) {
  trainIndex <- xGBM_unbal[[name]]$train_ids
  y_test <- as.factor(y_unbal$AKI_bin[-trainIndex])
  model_metr <- calculate_metrics(pprobs_xGBM_unbal[[name]]$y_pred_tree,
                                  as.factor(ifelse(pprobs_xGBM_unbal[[name]]$y_pred_tree > 0.5, 1, 0)), 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_xGBMtree_unbal <- rbind(metrics_xGBMtree_unbal, row) # add row to metrics_ATU dataframe
}
metrics_xGBMtree_unbal[,2:13] <- as.data.frame(lapply(metrics_xGBMtree_unbal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_xGBMtree_unbal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                    "Pos Pred Value", "Neg Pred Value", "Precision",
                                    "Recall", "F1", "Prevalence", "Detection Rate", 
                                    "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_xGBMtree_unbal <- rbind(metrics_xGBMtree_unbal, c("Mean", colMeans(metrics_xGBMtree_unbal[,2:13])))# add mean row
metrics_xGBMtree_unbal[,2:13] <- as.data.frame(lapply(metrics_xGBMtree_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMtree_unbal <- rbind(metrics_xGBMtree_unbal,c("SD", sapply(metrics_xGBMtree_unbal[1:100,2:13],sd)))# add standard deviation row
metrics_xGBMtree_unbal[,2:13] <- as.data.frame(lapply(metrics_xGBMtree_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMtree_unbal <- rbind(metrics_xGBMtree_unbal, c("Upper", sapply(metrics_xGBMtree_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_xGBMtree_unbal[,2:13] <- as.data.frame(lapply(metrics_xGBMtree_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_xGBMtree_unbal <- rbind(metrics_xGBMtree_unbal, c("Lower", sapply(metrics_xGBMtree_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_xGBMtree_unbal[,2:13] <- as.data.frame(lapply(metrics_xGBMtree_unbal[,2:13], function(x) as.numeric(as.character(x))))




# save dataframe

write.csv2(metrics_xGBMtree_unbal, "/your/path/performance_xGBMtree_unbal.csv")








# ______________________________________________________________________________

#                   Logistic Regression

# ______________________________________________________________________________
# ----

# preparation

# recode response into -1 and 1
y_bal_grp <- ifelse(y_bal == 1, 1, -1)# recode response to be numeric -1 and 1
y_unbal_grp <- data.frame(ifelse(y_unbal == 1, 1, -1))# recode response to be numeric -1 and 1



# Prediction and Performance estimation


# _________________________
# Group Regression
# _________________________



# ==========
# predict classes and class probabilities from listed models
pprobs_grLASSO_bal <- predict_classes_from_list_gr(grpLasso_bal, X_bal)
pprobs_grLASSO_unbal <- predict_classes_from_list_gr(grpLasso_unbal, X_unbal)

# ==========
# calculate Performances


# Balanced 
# create dataframe with performance metrics
metrics_grpLasso_bal <- data.frame()

for (name in names(pprobs_grLASSO_bal)) {
  trainIndex <- grpLasso_bal[[name]]$train_ids
  y_test <- y_bal_grp[-trainIndex]
  model_metr <- calculate_metrics(pprobs_grLASSO_bal[[name]]$y_pred,
                                  pprobs_grLASSO_bal[[name]]$y_pred_class, 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_grpLasso_bal <- rbind(metrics_grpLasso_bal, row) # add row to metrics_ATU dataframe
}
metrics_grpLasso_bal[,2:13] <- as.data.frame(lapply(metrics_grpLasso_bal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_grpLasso_bal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                   "Pos Pred Value", "Neg Pred Value", "Precision",
                                   "Recall", "F1", "Prevalence", "Detection Rate", 
                                   "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_grpLasso_bal <- rbind(metrics_grpLasso_bal, c("Mean", colMeans(metrics_grpLasso_bal[,2:13])))# add mean row
metrics_grpLasso_bal[,2:13] <- as.data.frame(lapply(metrics_grpLasso_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_grpLasso_bal <- rbind(metrics_grpLasso_bal,c("SD", sapply(metrics_grpLasso_bal[1:100,2:13],sd)))# add standard deviation row
metrics_grpLasso_bal[,2:13] <- as.data.frame(lapply(metrics_grpLasso_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_grpLasso_bal <- rbind(metrics_grpLasso_bal, c("Upper", sapply(metrics_grpLasso_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_grpLasso_bal[,2:13] <- as.data.frame(lapply(metrics_grpLasso_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_grpLasso_bal <- rbind(metrics_grpLasso_bal, c("Lower", sapply(metrics_grpLasso_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_grpLasso_bal[,2:13] <- as.data.frame(lapply(metrics_grpLasso_bal[,2:13], function(x) as.numeric(as.character(x))))




# save dataframe

write.csv2(metrics_grpLasso_bal, "/your/path/performance_grpLasso_bal.csv")

for (name in names(pprobs_grLASSO_unbal)) {
    pprobs_grLASSO_unbal[[name]]$y_pred_class <- as.factor(pprobs_grLASSO_unbal[[name]]$y_pred_class)
    levels(pprobs_grLASSO_unbal[[name]]$y_pred_class) <- c("-1", "1")
}

# unbalanced 
# create dataframe with performance metrics
metrics_grpLasso_unbal <- data.frame()

for (name in names(pprobs_grLASSO_unbal)) {
  trainIndex <- grpLasso_unbal[[name]]$train_ids
  y_test <- y_unbal_grp$AKI_bin[-trainIndex]
  model_metr <- calculate_metrics(pprobs_grLASSO_unbal[[name]]$y_pred,
                                  pprobs_grLASSO_unbal[[name]]$y_pred_class, 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_grpLasso_unbal <- rbind(metrics_grpLasso_unbal, row) # add row to metrics_ATU dataframe
}
metrics_grpLasso_unbal[,2:13] <- as.data.frame(lapply(metrics_grpLasso_unbal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_grpLasso_unbal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                    "Pos Pred Value", "Neg Pred Value", "Precision",
                                    "Recall", "F1", "Prevalence", "Detection Rate", 
                                    "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_grpLasso_unbal <- rbind(metrics_grpLasso_unbal, c("Mean", colMeans(metrics_grpLasso_unbal[,2:13], na.rm = TRUE)))# add mean row
metrics_grpLasso_unbal[,2:13] <- as.data.frame(lapply(metrics_grpLasso_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_grpLasso_unbal <- rbind(metrics_grpLasso_unbal,c("SD", sapply(metrics_grpLasso_unbal[1:100,2:13],function(x) sd(x, na.rm = TRUE))))# add standard deviation row
metrics_grpLasso_unbal[,2:13] <- as.data.frame(lapply(metrics_grpLasso_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_grpLasso_unbal <- rbind(metrics_grpLasso_unbal, c("Upper", sapply(metrics_grpLasso_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_grpLasso_unbal[,2:13] <- as.data.frame(lapply(metrics_grpLasso_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_grpLasso_unbal <- rbind(metrics_grpLasso_unbal, c("Lower", sapply(metrics_grpLasso_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_grpLasso_unbal[,2:13] <- as.data.frame(lapply(metrics_grpLasso_unbal[,2:13], function(x) as.numeric(as.character(x))))




# save dataframe

write.csv2(metrics_grpLasso_unbal, "/your/path/performance_grpLasso_unbal.csv")





# _________________________
# LASSO Regression
# _________________________



# ==========
# predict classes and class probabilities from listed models
pprobs_LASSO_bal <- predict_classes_from_list_ngr(Lasso_bal, X_bal)
pprobs_LASSO_unbal <- predict_classes_from_list_ngr(Lasso_unbal, X_unbal)

# ==========
# calculate Performances


# Balanced 
# create dataframe with performance metrics
metrics_Lasso_bal <- data.frame()

for (name in names(pprobs_LASSO_bal)) {
  trainIndex <- Lasso_bal[[name]]$train_ids
  y_test <- y_bal[-trainIndex]
  model_metr <- calculate_metrics(pprobs_LASSO_bal[[name]]$y_pred,
                                  pprobs_LASSO_bal[[name]]$y_pred_class, 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_Lasso_bal <- rbind(metrics_Lasso_bal, row) # add row to metrics_ATU dataframe
}
metrics_Lasso_bal[,2:13] <- as.data.frame(lapply(metrics_Lasso_bal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_Lasso_bal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                    "Pos Pred Value", "Neg Pred Value", "Precision",
                                    "Recall", "F1", "Prevalence", "Detection Rate", 
                                    "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_Lasso_bal <- rbind(metrics_Lasso_bal, c("Mean", colMeans(metrics_Lasso_bal[,2:13])))# add mean row
metrics_Lasso_bal[,2:13] <- as.data.frame(lapply(metrics_Lasso_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Lasso_bal <- rbind(metrics_Lasso_bal,c("SD", sapply(metrics_Lasso_bal[1:100,2:13],sd)))# add standard deviation row
metrics_Lasso_bal[,2:13] <- as.data.frame(lapply(metrics_Lasso_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Lasso_bal <- rbind(metrics_Lasso_bal, c("Upper", sapply(metrics_Lasso_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_Lasso_bal[,2:13] <- as.data.frame(lapply(metrics_Lasso_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Lasso_bal <- rbind(metrics_Lasso_bal, c("Lower", sapply(metrics_Lasso_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_Lasso_bal[,2:13] <- as.data.frame(lapply(metrics_Lasso_bal[,2:13], function(x) as.numeric(as.character(x))))


# save dataframe

write.csv2(metrics_Lasso_bal, "/your/path/performance_Lasso_bal.csv")






# unbalanced 


# adjust for missing factors in case of all zero classifications by model
for (name in names(pprobs_LASSO_unbal)) {
  pprobs_LASSO_unbal[[name]]$y_pred_class <- as.factor(pprobs_LASSO_unbal[[name]]$y_pred_class)
  levels(pprobs_LASSO_unbal[[name]]$y_pred_class) <- c("0", "1")
}


# create dataframe with performance metrics
metrics_Lasso_unbal <- data.frame()

for (name in names(pprobs_LASSO_unbal)) {
  trainIndex <- Lasso_unbal[[name]]$train_ids
  y_test <- y_unbal[-trainIndex]
  model_metr <- calculate_metrics(pprobs_LASSO_unbal[[name]]$y_pred,
                                  pprobs_LASSO_unbal[[name]]$y_pred_class, 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_Lasso_unbal <- rbind(metrics_Lasso_unbal, row) # add row to metrics_ATU dataframe
}
metrics_Lasso_unbal[,2:13] <- as.data.frame(lapply(metrics_Lasso_unbal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_Lasso_unbal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                 "Pos Pred Value", "Neg Pred Value", "Precision",
                                 "Recall", "F1", "Prevalence", "Detection Rate", 
                                 "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_Lasso_unbal <- rbind(metrics_Lasso_unbal, c("Mean", colMeans(metrics_Lasso_unbal[,2:13], na.rm = TRUE)))# add mean row
metrics_Lasso_unbal[,2:13] <- as.data.frame(lapply(metrics_Lasso_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Lasso_unbal <- rbind(metrics_Lasso_unbal,c("SD", sapply(metrics_Lasso_unbal[1:100,2:13], function(x) sd(x, na.rm = TRUE))))# add standard deviation row
metrics_Lasso_unbal[,2:13] <- as.data.frame(lapply(metrics_Lasso_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Lasso_unbal <- rbind(metrics_Lasso_unbal, c("Upper", sapply(metrics_Lasso_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_Lasso_unbal[,2:13] <- as.data.frame(lapply(metrics_Lasso_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Lasso_unbal <- rbind(metrics_Lasso_unbal, c("Lower", sapply(metrics_Lasso_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_Lasso_unbal[,2:13] <- as.data.frame(lapply(metrics_Lasso_unbal[,2:13], function(x) as.numeric(as.character(x))))


# save dataframe

write.csv2(metrics_Lasso_unbal, "/your/path/performance_Lasso_unbal.csv")






# _________________________
# Ridge Regression
# _________________________



# ==========
# predict classes and class probabilities from listed models
pprobs_Ridge_bal <- predict_classes_from_list_ngr(Ridge_bal, X_bal)
pprobs_Ridge_unbal <- predict_classes_from_list_ngr(Ridge_unbal, X_unbal)

# ==========
# calculate Performances


# Balanced 
# create dataframe with performance metrics
metrics_Ridge_bal <- data.frame()

for (name in names(pprobs_Ridge_bal)) {
  trainIndex <- Ridge_bal[[name]]$train_ids
  y_test <- y_bal[-trainIndex]
  model_metr <- calculate_metrics(pprobs_Ridge_bal[[name]]$y_pred,
                                  pprobs_Ridge_bal[[name]]$y_pred_class, 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_Ridge_bal <- rbind(metrics_Ridge_bal, row) # add row to metrics_ATU dataframe
} 
metrics_Ridge_bal[,2:13] <- as.data.frame(lapply(metrics_Ridge_bal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_Ridge_bal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                   "Pos Pred Value", "Neg Pred Value", "Precision",
                                   "Recall", "F1", "Prevalence", "Detection Rate", 
                                   "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_Ridge_bal <- rbind(metrics_Ridge_bal, c("Mean", colMeans(metrics_Ridge_bal[,2:13], na.rm = TRUE)))# add mean row
metrics_Ridge_bal[,2:13] <- as.data.frame(lapply(metrics_Ridge_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Ridge_bal <- rbind(metrics_Ridge_bal,c("SD", sapply(metrics_Ridge_bal[1:100,2:13], function(x) sd(x, na.rm = TRUE))))# add standard deviation row
metrics_Ridge_bal[,2:13] <- as.data.frame(lapply(metrics_Ridge_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Ridge_bal <- rbind(metrics_Ridge_bal, c("Upper", sapply(metrics_Ridge_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_Ridge_bal[,2:13] <- as.data.frame(lapply(metrics_Ridge_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Ridge_bal <- rbind(metrics_Ridge_bal, c("Lower", sapply(metrics_Ridge_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_Ridge_bal[,2:13] <- as.data.frame(lapply(metrics_Ridge_bal[,2:13], function(x) as.numeric(as.character(x))))

# save dataframe

write.csv2(metrics_Ridge_bal, "/your/path/performance_Ridge_bal.csv")






# unbalanced 


# adjust for missing factors in case of all zero classifications by model
for (name in names(pprobs_Ridge_unbal)) {
  pprobs_Ridge_unbal[[name]]$y_pred_class <- as.factor(pprobs_Ridge_unbal[[name]]$y_pred_class)
  levels(pprobs_Ridge_unbal[[name]]$y_pred_class) <- c("0", "1")
}


# create dataframe with performance metrics
metrics_Ridge_unbal <- data.frame()

for (name in names(pprobs_Ridge_unbal)) {
  trainIndex <- Ridge_unbal[[name]]$train_ids
  y_test <- y_unbal[-trainIndex]
  model_metr <- calculate_metrics(pprobs_Ridge_unbal[[name]]$y_pred,
                                  pprobs_Ridge_unbal[[name]]$y_pred_class, 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_Ridge_unbal <- rbind(metrics_Ridge_unbal, row) # add row to metrics_ATU dataframe
}
metrics_Ridge_unbal[,2:13] <- as.data.frame(lapply(metrics_Ridge_unbal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_Ridge_unbal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                   "Pos Pred Value", "Neg Pred Value", "Precision",
                                   "Recall", "F1", "Prevalence", "Detection Rate", 
                                   "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_Ridge_unbal <- rbind(metrics_Ridge_unbal, c("Mean", colMeans(metrics_Ridge_unbal[,2:13], na.rm = TRUE)))# add mean row
metrics_Ridge_unbal[,2:13] <- as.data.frame(lapply(metrics_Ridge_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Ridge_unbal <- rbind(metrics_Ridge_unbal,c("SD", sapply(metrics_Ridge_unbal[1:100,2:13], function(x) sd(x, na.rm = TRUE))))# add standard deviation row
metrics_Ridge_unbal[,2:13] <- as.data.frame(lapply(metrics_Ridge_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Ridge_unbal <- rbind(metrics_Ridge_unbal, c("Upper", sapply(metrics_Ridge_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_Ridge_unbal[,2:13] <- as.data.frame(lapply(metrics_Ridge_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_Ridge_unbal <- rbind(metrics_Ridge_unbal, c("Lower", sapply(metrics_Ridge_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_Ridge_unbal[,2:13] <- as.data.frame(lapply(metrics_Ridge_unbal[,2:13], function(x) as.numeric(as.character(x))))


# save dataframe

write.csv2(metrics_Ridge_unbal, "/your/path/performance_Ridge_unbal.csv")






# _________________________
# ENet Regression
# _________________________



# ==========
# predict classes and class probabilities from listed models
pprobs_ENet_bal <- predict_classes_from_list_ngr(Enet_bal, X_bal)
pprobs_ENet_unbal <- predict_classes_from_list_ngr(Enet_unbal, X_unbal)

# ==========
# calculate Performances


# Balanced 
# create dataframe with performance metrics
metrics_ENet_bal <- data.frame()

for (name in names(pprobs_ENet_bal)) {
  trainIndex <- Enet_bal[[name]]$train_ids
  y_test <- y_bal[-trainIndex]
  model_metr <- calculate_metrics(pprobs_ENet_bal[[name]]$y_pred,
                                  pprobs_ENet_bal[[name]]$y_pred_class, 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_ENet_bal <- rbind(metrics_ENet_bal, row) # add row to metrics_ATU dataframe
}
metrics_ENet_bal[,2:13] <- as.data.frame(lapply(metrics_ENet_bal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_ENet_bal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                 "Pos Pred Value", "Neg Pred Value", "Precision",
                                 "Recall", "F1", "Prevalence", "Detection Rate", 
                                 "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_ENet_bal <- rbind(metrics_ENet_bal, c("Mean", colMeans(metrics_ENet_bal[,2:13])))# add mean row
metrics_ENet_bal[,2:13] <- as.data.frame(lapply(metrics_ENet_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_ENet_bal <- rbind(metrics_ENet_bal,c("SD", sapply(metrics_ENet_bal[1:100,2:13],sd)))# add standard deviation row
metrics_ENet_bal[,2:13] <- as.data.frame(lapply(metrics_ENet_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_ENet_bal <- rbind(metrics_ENet_bal, c("Upper", sapply(metrics_ENet_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_ENet_bal[,2:13] <- as.data.frame(lapply(metrics_ENet_bal[,2:13], function(x) as.numeric(as.character(x))))
metrics_ENet_bal <- rbind(metrics_ENet_bal, c("Lower", sapply(metrics_ENet_bal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_ENet_bal[,2:13] <- as.data.frame(lapply(metrics_ENet_bal[,2:13], function(x) as.numeric(as.character(x))))


# save dataframe

write.csv2(metrics_ENet_bal, "/your/path/performance_ENet_bal.csv")






# unbalanced 


# adjust for missing factors in case of all zero classifications by model
for (name in names(pprobs_ENet_unbal)) {
  pprobs_ENet_unbal[[name]]$y_pred_class <- as.factor(pprobs_ENet_unbal[[name]]$y_pred_class)
  levels(pprobs_ENet_unbal[[name]]$y_pred_class) <- c("0", "1")
}


# create dataframe with performance metrics
metrics_ENet_unbal <- data.frame()

for (name in names(pprobs_ENet_unbal)) {
  trainIndex <- Enet_unbal[[name]]$train_ids
  y_test <- y_unbal[-trainIndex]
  model_metr <- calculate_metrics(pprobs_ENet_unbal[[name]]$y_pred,
                                  pprobs_ENet_unbal[[name]]$y_pred_class, 
                                  y_test, 
                                  "1") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_ENet_unbal <- rbind(metrics_ENet_unbal, row) # add row to metrics_ATU dataframe
}
metrics_ENet_unbal[,2:13] <- as.data.frame(lapply(metrics_ENet_unbal[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_ENet_unbal) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                                   "Pos Pred Value", "Neg Pred Value", "Precision",
                                   "Recall", "F1", "Prevalence", "Detection Rate", 
                                   "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_ENet_unbal <- rbind(metrics_ENet_unbal, c("Mean", colMeans(metrics_ENet_unbal[,2:13], na.rm = TRUE)))# add mean row
metrics_ENet_unbal[,2:13] <- as.data.frame(lapply(metrics_ENet_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_ENet_unbal <- rbind(metrics_ENet_unbal,c("SD", sapply(metrics_ENet_unbal[1:100,2:13], function(x) sd(x, na.rm = TRUE))))# add standard deviation row
metrics_ENet_unbal[,2:13] <- as.data.frame(lapply(metrics_ENet_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_ENet_unbal <- rbind(metrics_ENet_unbal, c("Upper", sapply(metrics_ENet_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_ENet_unbal[,2:13] <- as.data.frame(lapply(metrics_ENet_unbal[,2:13], function(x) as.numeric(as.character(x))))
metrics_ENet_unbal <- rbind(metrics_ENet_unbal, c("Lower", sapply(metrics_ENet_unbal[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_ENet_unbal[,2:13] <- as.data.frame(lapply(metrics_ENet_unbal[,2:13], function(x) as.numeric(as.character(x))))


# save dataframe

write.csv2(metrics_ENet_unbal, "/your/path/performance_ENet_unbal.csv")



# aggregate dataframes in list
cv_performance <- list(metrics_grpLasso_bal, metrics_Lasso_bal,
                       metrics_Ridge_bal, metrics_ENet_bal, 
                       metrics_SVM_bal, metrics_xGBMlin_bal,
                       metrics_xGBMtree_bal, metrics_RF_bal,
                       metrics_grpLasso_unbal, metrics_Lasso_unbal,
                       metrics_Ridge_unbal, metrics_ENet_unbal, 
                       metrics_SVM_unbal, metrics_xGBMlin_unbal,
                       metrics_xGBMtree_unbal, metrics_RF_unbal)

rm(metrics_grpLasso_bal, metrics_Lasso_bal,
   metrics_Ridge_bal, metrics_ENet_bal, 
   metrics_SVM_bal, metrics_xGBMlin_bal,
   metrics_xGBMtree_bal, metrics_RF_bal,
   metrics_grpLasso_unbal, metrics_Lasso_unbal,
   metrics_Ridge_unbal, metrics_ENet_unbal, 
   metrics_SVM_unbal, metrics_xGBMlin_unbal,
   metrics_xGBMtree_unbal, metrics_RF_unbal)

# create summary dataframes for Mean and Standard deviation
assign <- data.frame(Data=c("Bal", "Bal", "Bal", "Bal", "Bal", "Bal", "Bal", "Bal",
                            "Unbal", "Unbal", "Unbal", "Unbal", "Unbal", "Unbal", "Unbal", "Unbal"),
                     Model = c("Group LASSO", "LASSO", "Ridge", "Elastic Net", "SVM", "Linear xGBM", "Tree xGBM", "RF",
                               "Group LASSO", "LASSO", "Ridge", "Elastic Net", "SVM", "Linear xGBM", "Tree xGBM", "RF"))


df_mean <- bind_rows(lapply(cv_performance, function(df) as.data.frame(df) %>% filter(`Model seed` == "Mean")), .id = "Source")
df_mean <- df_mean %>% select(-Source, -`Model seed`)
df_mean <- cbind(assign, df_mean)
df_sd <- bind_rows(lapply(cv_performance, function(df) df %>% filter(`Model seed` == "SD")), .id = "Source")
df_sd <- df_sd %>% select(-Source, -`Model seed`)
df_sd <- cbind(assign, df_sd)


# combined table

means_rnd <- round(df_mean[,-c(1:2)],3)
sds_rnd <- round(df_sd[,-c(1:2)],3)

# Combine the rounded means and SDs with a ± sign element-wise
combined_df <- as.data.frame(mapply(function(m, s) paste(m, "±", s), means_rnd, sds_rnd))
combined_df <- cbind(assign, combined_df)

combined_df_reduced <- combined_df %>% select(-c("Pos Pred Value", "Recall", "Prevalence", "Detection Prevalence", "Detection Rate"))


# save data frames

write.csv2(combined_df, "/your/path/var_est_train_full.csv")
write.csv2(combined_df_reduced, "/your/path/var_est_train_red.csv")


# Prepare Lower CI dataframe
df_lower <- bind_rows(lapply(cv_performance, function(df) as.data.frame(df) %>% filter(`Model seed` == "Lower")), .id = "Source")
df_lower <- df_lower %>% select(-Source, -`Model seed`)
df_lower <- cbind(assign, df_lower)

# Prepare Upper CI dataframe
df_upper <- bind_rows(lapply(cv_performance, function(df) as.data.frame(df) %>% filter(`Model seed` == "Upper")), .id = "Source")
df_upper <- df_upper %>% select(-Source, -`Model seed`)
df_upper <- cbind(assign, df_upper)

# Round both Lower and Upper values
lower_rnd <- round(df_lower[,-c(1:2)], 3)
upper_rnd <- round(df_upper[,-c(1:2)], 3)

# Combine the Lower and Upper CI with a "—" (en-dash) separator
combined_ci_df <- as.data.frame(mapply(function(l, u) paste(l, "—", u), lower_rnd, upper_rnd))
combined_ci_df <- cbind(assign, combined_ci_df)

# Optionally, reduce columns if needed (like before)
combined_ci_df_reduced <- combined_ci_df %>%
  select(-c("Pos Pred Value", "Recall", "Prevalence", "Detection Prevalence", "Detection Rate"))


# save data frames

write.csv2(combined_ci_df, "/your/path/var_CI_train_full.csv")
write.csv2(combined_ci_df_reduced, "/your/path/var_CI_train_red.csv")




# compare bal and unbal approaches

sd_bal <- df_sd[df_sd$Data == "Bal",]
sd_unbal <- df_sd[df_sd$Dat == "Unbal",]
mean_bal <- df_mean[df_mean$Data == "Bal",]
mean_unbal <- df_mean[df_mean$Dat == "Unbal",]


# boxplots for AUC, Balanced Accuracy and F1 Score

mean_long <- df_mean[,c(1,14)] %>%
  pivot_longer(-"Data", names_to = "Metric", values_to = "Value")


ggplot(mean_long, aes(x = "Metric", y = Value, fill = .data[["Data"]])) +
  geom_boxplot(alpha = 0.6) +
  theme_minimal() +
  labs(title = "Comparison of Variables from Two Datasets",
       x = "Variable",
       y = "Value") +
  scale_fill_manual(values = c("blue", "red"))
# test for difference between approaches
# t-test assuming equal variance


boxplot_metric <- function(df, group_col ){
  df_long <- df %>%
    pivot_longer(-colnames(df)[2], names_to = colnames(df)[2], values_to = "Value")
  
  
  ggplot(df, aes(x = .data[[colnames(df)[2]]], y = Value, fill = .data[[colnames(df)[1]]])) +
    geom_boxplot(alpha = 0.6) +
    theme_minimal() +
    labs(title = "Comparison of Variables from Two Datasets",
         x = "Variable",
         y = "Value") +
    scale_fill_manual(values = c("blue", "red"))
}









#end
#----



