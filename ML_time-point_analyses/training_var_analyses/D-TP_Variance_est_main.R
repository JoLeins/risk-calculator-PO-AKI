# ==============================================================================
# Estimation of Input based training variance in PO-AKI Prediction Models 
# (time point) based on 100-fold nested cross-validation on the development data
# set.
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
dsets <- readRDS("/your/path/Timepoint_datasets.rds") 




# Timepoint models
praeop <- readRDS("/your/path/RF_praeop.rds")
induct <- readRDS("/your/path/RF_einleitung.rds")
indu30 <- readRDS("/your/path/RF_einlplus30.rds")
indu60 <- readRDS("/your/path/RF_einlplus60.rds")
indu90 <- readRDS("/your/path/RF_einlplus90.rds")
ind120 <- readRDS("/your/path/RF_einlplus120.rds")
ind150 <- readRDS("/your/path/RF_einlplus150.rds")
postop <- readRDS("/your/path/RF_postop.rds")



# ______________________________________________________________________________ 
#                               preoperative
# ______________________________________________________________________________
# ----

# ==========
# preparation

# define X and y
X_bal <- dsets$praeop$train %>% select(-c(AKI_bin))
y_bal <- dsets$praeop$train %>% select(c(AKI_bin))


# ==========
# Prediction and Performance estimation

# predict classes and class probabilities from listed models
pprobs_preop <- predict_classes_from_list_RF(praeop, X_bal)


# ==========
# calculate Performances

# Balanced
# create dataframe with performance metrics
metrics_preop <- data.frame()

for (name in names(pprobs_preop)) {
  trainIndex <- praeop[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  levels(y_test) <- c("no_event", "event")
  model_metr <- calculate_metrics(pprobs_preop[[name]]$y_pred,
                                  pprobs_preop[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_preop <- rbind(metrics_preop, row) # add row to metrics_ATU dataframe
}
metrics_preop[,2:13] <- as.data.frame(lapply(metrics_preop[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_preop) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                               "Pos Pred Value", "Neg Pred Value", "Precision",
                               "Recall", "F1", "Prevalence", "Detection Rate", 
                               "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_preop <- rbind(metrics_preop, c("Mean", colMeans(metrics_preop[,2:13])))# add mean row
metrics_preop[,2:13] <- as.data.frame(lapply(metrics_preop[,2:13], function(x) as.numeric(as.character(x))))
metrics_preop <- rbind(metrics_preop, c("SD", sapply(metrics_preop[1:100,2:13],sd)))# add standard deviation row
metrics_preop[,2:13] <- as.data.frame(lapply(metrics_preop[,2:13], function(x) as.numeric(as.character(x))))
metrics_preop <- rbind(metrics_preop, c("Upper", sapply(metrics_preop[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_preop[,2:13] <- as.data.frame(lapply(metrics_preop[,2:13], function(x) as.numeric(as.character(x))))
metrics_preop <- rbind(metrics_preop, c("Lower", sapply(metrics_preop[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_preop[,2:13] <- as.data.frame(lapply(metrics_preop[,2:13], function(x) as.numeric(as.character(x))))


# save dataframe
write.csv2(metrics_preop, "/your/path/performance_preop.csv")



# ______________________________________________________________________________ 
#                               Induction
# ______________________________________________________________________________
# ----

# ==========
# preparation

# define X and y
X_bal <- dsets$einleitung$train %>% select(-c(AKI_bin))
y_bal <- dsets$einleitung$train %>% select(c(AKI_bin))


# ==========
# Prediction and Performance estimation

# predict classes and class probabilities from listed models
pprobs_induc <- predict_classes_from_list_RF(induct, X_bal)


# ==========
# calculate Performances

# Balanced
# create dataframe with performance metrics
metrics_indu <- data.frame()

for (name in names(pprobs_induc)) {
  trainIndex <- induct[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  levels(y_test) <- c("no_event", "event")
  model_metr <- calculate_metrics(pprobs_induc[[name]]$y_pred,
                                  pprobs_induc[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_indu <- rbind(metrics_indu, row) # add row to metrics_ATU dataframe
}
metrics_indu[,2:13] <- as.data.frame(lapply(metrics_indu[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_indu) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                              "Pos Pred Value", "Neg Pred Value", "Precision",
                              "Recall", "F1", "Prevalence", "Detection Rate", 
                              "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_indu <- rbind(metrics_indu, c("Mean", colMeans(metrics_indu[,2:13])))# add mean row
metrics_indu[,2:13] <- as.data.frame(lapply(metrics_indu[,2:13], function(x) as.numeric(as.character(x))))
metrics_indu <- rbind(metrics_indu, c("SD", sapply(metrics_indu[1:100,2:13],sd)))# add standard deviation row
metrics_indu[,2:13] <- as.data.frame(lapply(metrics_indu[,2:13], function(x) as.numeric(as.character(x))))
metrics_indu <- rbind(metrics_indu, c("Upper", sapply(metrics_indu[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_indu[,2:13] <- as.data.frame(lapply(metrics_indu[,2:13], function(x) as.numeric(as.character(x))))
metrics_indu <- rbind(metrics_indu, c("Lower", sapply(metrics_indu[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_indu[,2:13] <- as.data.frame(lapply(metrics_indu[,2:13], function(x) as.numeric(as.character(x))))


# save dataframe
write.csv2(metrics_indu, "/your/path/performance_induc.csv")




# ______________________________________________________________________________ 
#                           Induction plus 30 min
# ______________________________________________________________________________
# ----

# ==========
# preparation

# define X and y
X_bal <- dsets$einlplus30$train %>% select(-c(AKI_bin))
y_bal <- dsets$einlplus30$train %>% select(c(AKI_bin))


# ==========
# Prediction and Performance estimation

# predict classes and class probabilities from listed models
pprobs_ind30 <- predict_classes_from_list_RF(indu30, X_bal)


# ==========
# calculate Performances

# Balanced
# create dataframe with performance metrics
metrics_ind30 <- data.frame()

for (name in names(pprobs_ind30)) {
  trainIndex <- indu30[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  levels(y_test) <- c("no_event", "event")
  model_metr <- calculate_metrics(pprobs_ind30[[name]]$y_pred,
                                  pprobs_ind30[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_ind30 <- rbind(metrics_ind30, row) # add row to metrics_ATU dataframe
}
metrics_ind30[,2:13] <- as.data.frame(lapply(metrics_ind30[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_ind30) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                            "Pos Pred Value", "Neg Pred Value", "Precision",
                            "Recall", "F1", "Prevalence", "Detection Rate", 
                            "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_ind30 <- rbind(metrics_ind30, c("Mean", colMeans(metrics_ind30[,2:13])))# add mean row
metrics_ind30[,2:13] <- as.data.frame(lapply(metrics_ind30[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind30 <- rbind(metrics_ind30, c("SD", sapply(metrics_ind30[1:100,2:13],sd)))# add standard deviation row
metrics_ind30[,2:13] <- as.data.frame(lapply(metrics_ind30[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind30 <- rbind(metrics_ind30, c("Upper", sapply(metrics_ind30[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_ind30[,2:13] <- as.data.frame(lapply(metrics_ind30[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind30 <- rbind(metrics_ind30, c("Lower", sapply(metrics_ind30[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_ind30[,2:13] <- as.data.frame(lapply(metrics_ind30[,2:13], function(x) as.numeric(as.character(x))))


# save dataframe
write.csv2(metrics_ind30, "/your/path/performance_ind30.csv")




# ______________________________________________________________________________ 
#                           Induction plus 60 min
# ______________________________________________________________________________
# ----

# ==========
# preparation

# define X and y
X_bal <- dsets$einlplus60$train %>% select(-c(AKI_bin))
y_bal <- dsets$einlplus60$train %>% select(c(AKI_bin))


# ==========
# Prediction and Performance estimation

# predict classes and class probabilities from listed models
pprobs_ind60 <- predict_classes_from_list_RF(indu60, X_bal)


# ==========
# calculate Performances

# Balanced
# create dataframe with performance metrics
metrics_ind60 <- data.frame()

for (name in names(pprobs_ind60)) {
  trainIndex <- indu60[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  levels(y_test) <- c("no_event", "event")
  model_metr <- calculate_metrics(pprobs_ind60[[name]]$y_pred,
                                  pprobs_ind60[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_ind60 <- rbind(metrics_ind60, row) # add row to metrics_ATU dataframe
}
metrics_ind60[,2:13] <- as.data.frame(lapply(metrics_ind60[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_ind60) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                             "Pos Pred Value", "Neg Pred Value", "Precision",
                             "Recall", "F1", "Prevalence", "Detection Rate", 
                             "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_ind60 <- rbind(metrics_ind60, c("Mean", colMeans(metrics_ind60[,2:13])))# add mean row
metrics_ind60[,2:13] <- as.data.frame(lapply(metrics_ind60[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind60 <- rbind(metrics_ind60, c("SD", sapply(metrics_ind60[1:100,2:13],sd)))# add standard deviation row
metrics_ind60[,2:13] <- as.data.frame(lapply(metrics_ind60[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind60 <- rbind(metrics_ind60, c("Upper", sapply(metrics_ind60[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_ind60[,2:13] <- as.data.frame(lapply(metrics_ind60[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind60 <- rbind(metrics_ind60, c("Lower", sapply(metrics_ind60[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_ind60[,2:13] <- as.data.frame(lapply(metrics_ind60[,2:13], function(x) as.numeric(as.character(x))))

# save dataframe
write.csv2(metrics_ind60, "/your/path/performance_ind60.csv")




# ______________________________________________________________________________ 
#                           Induction plus 90 min
# ______________________________________________________________________________
# ----

# ==========
# preparation

# define X and y
X_bal <- dsets$einlplus90$train %>% select(-c(AKI_bin))
y_bal <- dsets$einlplus90$train %>% select(c(AKI_bin))


# ==========
# Prediction and Performance estimation

# predict classes and class probabilities from listed models
pprobs_ind90 <- predict_classes_from_list_RF(indu90, X_bal)


# ==========
# calculate Performances

# Balanced
# create dataframe with performance metrics
metrics_ind90 <- data.frame()

for (name in names(pprobs_ind90)) {
  trainIndex <- indu90[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  levels(y_test) <- c("no_event", "event")
  model_metr <- calculate_metrics(pprobs_ind90[[name]]$y_pred,
                                  pprobs_ind90[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_ind90 <- rbind(metrics_ind90, row) # add row to metrics_ATU dataframe
}
metrics_ind90[,2:13] <- as.data.frame(lapply(metrics_ind90[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_ind90) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                             "Pos Pred Value", "Neg Pred Value", "Precision",
                             "Recall", "F1", "Prevalence", "Detection Rate", 
                             "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_ind90 <- rbind(metrics_ind90, c("Mean", colMeans(metrics_ind90[,2:13])))# add mean row
metrics_ind90[,2:13] <- as.data.frame(lapply(metrics_ind90[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind90 <- rbind(metrics_ind90, c("SD", sapply(metrics_ind90[1:100,2:13],sd)))# add standard deviation row
metrics_ind90[,2:13] <- as.data.frame(lapply(metrics_ind90[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind90 <- rbind(metrics_ind90, c("Upper", sapply(metrics_ind90[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_ind90[,2:13] <- as.data.frame(lapply(metrics_ind90[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind90 <- rbind(metrics_ind90, c("Lower", sapply(metrics_ind90[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_ind90[,2:13] <- as.data.frame(lapply(metrics_ind90[,2:13], function(x) as.numeric(as.character(x))))

# save dataframe
write.csv2(metrics_ind90, "/your/path/performance_ind90.csv")





# ______________________________________________________________________________ 
#                           Induction plus 120 min
# ______________________________________________________________________________
# ----

# ==========
# preparation

# define X and y
X_bal <- dsets$einlplus120$train %>% select(-c(AKI_bin))
y_bal <- dsets$einlplus120$train %>% select(c(AKI_bin))


# ==========
# Prediction and Performance estimation

# predict classes and class probabilities from listed models
pprobs_ind120 <- predict_classes_from_list_RF(ind120, X_bal)


# ==========
# calculate Performances

# Balanced
# create dataframe with performance metrics
metrics_ind120 <- data.frame()

for (name in names(pprobs_ind120)) {
  trainIndex <- ind120[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  levels(y_test) <- c("no_event", "event")
  model_metr <- calculate_metrics(pprobs_ind120[[name]]$y_pred,
                                  pprobs_ind120[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_ind120 <- rbind(metrics_ind120, row) # add row to metrics_ATU dataframe
}
metrics_ind120[,2:13] <- as.data.frame(lapply(metrics_ind120[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_ind120) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                             "Pos Pred Value", "Neg Pred Value", "Precision",
                             "Recall", "F1", "Prevalence", "Detection Rate", 
                             "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_ind120 <- rbind(metrics_ind120, c("Mean", colMeans(metrics_ind120[,2:13])))# add mean row
metrics_ind120[,2:13] <- as.data.frame(lapply(metrics_ind120[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind120 <- rbind(metrics_ind120, c("SD", sapply(metrics_ind120[1:100,2:13],sd)))# add standard deviation row
metrics_ind120[,2:13] <- as.data.frame(lapply(metrics_ind120[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind120 <- rbind(metrics_ind120, c("Upper", sapply(metrics_ind120[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_ind120[,2:13] <- as.data.frame(lapply(metrics_ind120[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind120 <- rbind(metrics_ind120, c("Lower", sapply(metrics_ind120[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_ind120[,2:13] <- as.data.frame(lapply(metrics_ind120[,2:13], function(x) as.numeric(as.character(x))))

# save dataframe
write.csv2(metrics_ind120, "/your/path/performance_ind120.csv")




# ______________________________________________________________________________ 
#                           Induction plus 150 min
# ______________________________________________________________________________
# ----

# ==========
# preparation

# define X and y
X_bal <- dsets$einlplus150$train %>% select(-c(AKI_bin))
y_bal <- dsets$einlplus150$train %>% select(c(AKI_bin))


# ==========
# Prediction and Performance estimation

# predict classes and class probabilities from listed models
pprobs_ind150 <- predict_classes_from_list_RF(ind150, X_bal)


# ==========
# calculate Performances

# Balanced
# create dataframe with performance metrics
metrics_ind150 <- data.frame()

for (name in names(pprobs_ind150)) {
  trainIndex <- ind150[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  levels(y_test) <- c("no_event", "event")
  model_metr <- calculate_metrics(pprobs_ind150[[name]]$y_pred,
                                  pprobs_ind150[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_ind150 <- rbind(metrics_ind150, row) # add row to metrics_ATU dataframe
}
metrics_ind150[,2:13] <- as.data.frame(lapply(metrics_ind150[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_ind150) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                             "Pos Pred Value", "Neg Pred Value", "Precision",
                             "Recall", "F1", "Prevalence", "Detection Rate", 
                             "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_ind150 <- rbind(metrics_ind150, c("Mean", colMeans(metrics_ind150[,2:13])))# add mean row
metrics_ind150[,2:13] <- as.data.frame(lapply(metrics_ind150[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind150 <- rbind(metrics_ind150, c("SD", sapply(metrics_ind150[1:100,2:13],sd)))# add standard deviation row
metrics_ind150[,2:13] <- as.data.frame(lapply(metrics_ind150[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind150 <- rbind(metrics_ind150, c("Upper", sapply(metrics_ind150[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_ind150[,2:13] <- as.data.frame(lapply(metrics_ind150[,2:13], function(x) as.numeric(as.character(x))))
metrics_ind150 <- rbind(metrics_ind150, c("Lower", sapply(metrics_ind150[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_ind150[,2:13] <- as.data.frame(lapply(metrics_ind150[,2:13], function(x) as.numeric(as.character(x))))

# save dataframe
write.csv2(metrics_ind150, "/your/path/performance_ind150.csv")




# ______________________________________________________________________________ 
#                               postoperative
# ______________________________________________________________________________
# ----

# ==========
# preparation

# define X and y
X_bal <- dsets$postop$train %>% select(-c(AKI_bin))
y_bal <- dsets$postop$train %>% select(c(AKI_bin))


# ==========
# Prediction and Performance estimation

# predict classes and class probabilities from listed models
pprobs_postop <- predict_classes_from_list_RF(postop, X_bal)


# ==========
# calculate Performances

# Balanced
# create dataframe with performance metrics
metrics_postop <- data.frame()

for (name in names(pprobs_postop)) {
  trainIndex <- postop[[name]]$train_ids
  y_test <- as.factor(y_bal$AKI_bin[-trainIndex])
  levels(y_test) <- c("no_event", "event")
  model_metr <- calculate_metrics(pprobs_postop[[name]]$y_pred,
                                  pprobs_postop[[name]]$y_pred_class, 
                                  y_test, 
                                  "event") # call calculate_metrics() to evaluate model performance
  row <- append(name, model_metr) # create row with Model name and the calculated metrics
  metrics_postop <- rbind(metrics_postop, row) # add row to metrics_ATU dataframe
}
metrics_postop[,2:13] <- as.data.frame(lapply(metrics_postop[,2:13], function(x) as.numeric(as.character(x))))
colnames(metrics_postop) <- c("Model seed", "AUC", "Sensitivity", "Specificity",
                             "Pos Pred Value", "Neg Pred Value", "Precision",
                             "Recall", "F1", "Prevalence", "Detection Rate", 
                             "Detection Prevalence", "Balanced Accuracy") # modify colnames
metrics_postop <- rbind(metrics_postop, c("Mean", colMeans(metrics_postop[,2:13])))# add mean row
metrics_postop[,2:13] <- as.data.frame(lapply(metrics_postop[,2:13], function(x) as.numeric(as.character(x))))
metrics_postop <- rbind(metrics_postop, c("SD", sapply(metrics_postop[1:100,2:13],sd)))# add standard deviation row
metrics_postop[,2:13] <- as.data.frame(lapply(metrics_postop[,2:13], function(x) as.numeric(as.character(x))))
metrics_postop <- rbind(metrics_postop, c("Upper", sapply(metrics_postop[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.975)))
metrics_postop[,2:13] <- as.data.frame(lapply(metrics_postop[,2:13], function(x) as.numeric(as.character(x))))
metrics_postop <- rbind(metrics_postop, c("Lower", sapply(metrics_postop[1:100, 2:13], quantile, na.rm = TRUE, probs = 0.025)))
metrics_postop[,2:13] <- as.data.frame(lapply(metrics_postop[,2:13], function(x) as.numeric(as.character(x))))

# save dataframe
write.csv2(metrics_postop, "/your/path/performance_postop.csv")



# ______________________________________________________________________________
#                            Summary Table
# ______________________________________________________________________________


# aggregate dataframes in list
cv_performance <- list(metrics_preop, metrics_indu,
                       metrics_ind30, metrics_ind60, 
                       metrics_ind90, metrics_ind120,
                       metrics_ind150, metrics_postop
                       )

rm(metrics_indu,metrics_ind30,metrics_ind60,metrics_ind90,metrics_ind120
   ,metrics_ind150,metrics_postop,metrics_preop,postop)
#rm(metrics_preop, metrics_indu,
#   metrics_ind30, metrics_ind60, 
#   metrics_ind90, metrics_ind120,
#   metrics_ind150, metrics_postop)

# create summary dataframes for Mean and Standard deviation
assign <- data.frame(Timepoint=c("preop", "ind", "ind30", "ind60", "ind90", "ind120", "ind150", "postop")
                     )


df_mean <- bind_rows(lapply(cv_performance, function(df) as.data.frame(df) %>% filter(`Model seed` == "Mean")), .id = "Source")
df_mean <- df_mean %>% select(-Source, -`Model seed`)
df_mean <- cbind(assign, df_mean)
df_sd <- bind_rows(lapply(cv_performance, function(df) df %>% filter(`Model seed` == "SD")), .id = "Source")
df_sd <- df_sd %>% select(-Source, -`Model seed`)
df_sd <- cbind(assign, df_sd)


# combined table

means_rnd <- round(df_mean[,-1],3)
sds_rnd <- round(df_sd[,-1],3)

# Combine the rounded means and SDs with a ± sign element-wise
combined_df <- as.data.frame(mapply(function(m, s) paste(m, "±", s), means_rnd, sds_rnd))
combined_df <- cbind(assign, combined_df)

combined_df_reduced <- combined_df %>% select(-c("Pos Pred Value", "Recall", "Prevalence", "Detection Prevalence", "Detection Rate"))


# save data frames
write.csv2(combined_df, "/your/path/TP_var_est_train_full.csv")
write.csv2(combined_df_reduced, "/your/path/TP_var_est_train_red.csv")


# Prepare Lower CI dataframe
df_lower <- bind_rows(lapply(cv_performance, function(df) as.data.frame(df) %>% filter(`Model seed` == "Lower")), .id = "Source")
df_lower <- df_lower %>% select(-Source, -`Model seed`)
df_lower <- cbind(assign, df_lower)

# Prepare Upper CI dataframe
df_upper <- bind_rows(lapply(cv_performance, function(df) as.data.frame(df) %>% filter(`Model seed` == "Upper")), .id = "Source")
df_upper <- df_upper %>% select(-Source, -`Model seed`)
df_upper <- cbind(assign, df_upper)

# Round both Lower and Upper values
lower_rnd <- round(df_lower[,-1], 3)
upper_rnd <- round(df_upper[,-1], 3)

# Combine the Lower and Upper CI with a "—" (en-dash) separator
combined_ci_df <- as.data.frame(mapply(function(l, u) paste(l, "—", u), lower_rnd, upper_rnd))
combined_ci_df <- cbind(assign, combined_ci_df)

# Optionally, reduce columns if needed (like before)
combined_ci_df_reduced <- combined_ci_df %>%
  select(-c("Pos Pred Value", "Recall", "Prevalence", "Detection Prevalence", "Detection Rate"))


# save data frames

write.csv2(combined_ci_df, "/your/path/TP_var_CI_train_full.csv")
write.csv2(combined_ci_df_reduced, "/your/path/TP_var_CI_train_red.csv")


#end
#----

