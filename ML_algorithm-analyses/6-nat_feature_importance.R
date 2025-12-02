# ==============================================================================
# Evaluation of ML-Models (non-time points) native Feature Importance Estimations
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

# group one-hot encoded features' importance estimations for comparison
group_imp <- function(x, mapping){
  
  # flatten mapping
  mapping_df <- stack(mapping) %>%
    rename(Features = values, original_feature = ind)
  
  # Ensure column names exist before merging
  if (!"Features" %in% names(x)) {
    stop("Error: 'x' must have a column named 'Features'.")
  }
  if (!"Overall" %in% names(x)) {
    stop("Error: 'x' must have a column named 'Overall'.")
  }
  # melt data for easier processing
  combined_importances <- x %>% 
    left_join(mapping_df, by = "Features") %>%
    mutate(original_feature = coalesce(original_feature, Features)) %>%
    group_by(original_feature) %>%
    dplyr::summarize(combined_importance = mean(Overall, na.rm = TRUE), .groups = "drop")
  
  return(combined_importances)
}

# rank features based on estimated importance
rank_importance <- function(x){
  dataframe_with_ranks <- x %>%
    mutate(rank = ifelse(Overall == 0, NA, row_number())) %>%
    fill(rank, .direction = "downup") %>%
    mutate(rank =ifelse(Overall == 0, max(rank, na.rm = TRUE) + 1, rank)) %>%
    arrange(desc(Overall))
  return(dataframe_with_ranks)
}


capitalize <- function(column, key){
  column <- ifelse(column %in% names(key),
                   key[column],
                   column)
  return(column)
}



# ----
# setup

#load packages

load_packages( c("tidyverse", "dplyr", "glmnet", "gglasso", "mltools", "data.table", 
                 "grpnet", "caret", "ggplot2", "ROCR", "tidymodels", "fastDummies", "ranger"))
setwd("/your/path/")
set.seed(3010)





# ----
# main


# ==========
# load and extract best performing models
final_models <- readRDS("/your/path/final_models.rds")


# grouping and averaging feature importances of one-hot-encoded features for direct comparison 
one_hot_mapping <- list(
  asa = c("asa_2", "asa_3", "asa_4", "asa_5"),
  nyha = c("nyha_1", "nyha_2", "nyha_3", "nyha_4"),
  rcri = c("rcri_1", "rcri_2", "rcri_3"),
  urgency = c("urgency_1", "urgency_2", "urgency_3")
)



# Create key to capitalize certain abbreviations

# Key of name replacements
key <- c("asa" = "ASA", "rcri" = "RCRI", "bmi" = "BMI", "nyha" = "NYHA")


# Adjusted Feature Importance Estimations (grouping of one-hot-encoded Features)
# ==========
# calculate importances

# Group LASSO

adj_imp_grLASSO <- coef(final_models$balanced$group_lasso$gglasso.fit, s=final_models$balanced$group_lasso$lambda.min)
adj_imp_grLASSO <- data.frame( "Importance" = abs(adj_imp_grLASSO))
adj_imp_grLASSO <- subset(adj_imp_grLASSO, rownames(adj_imp_grLASSO) != "(Intercept)") #remove intercept term 
colnames(adj_imp_grLASSO) <- c("Overall")

#condense one-hot encoded variables
adj_imp_grLASSO$Features <- rownames(adj_imp_grLASSO)
adj_imp_grLASSO <- group_imp(adj_imp_grLASSO, one_hot_mapping)
colnames(adj_imp_grLASSO) <- c("Features", "Overall")


#scale to max importance
adj_imp_grLASSO$Overall <- adj_imp_grLASSO$Overall/max(adj_imp_grLASSO$Overall)
adj_imp_grLASSO$Overall <- adj_imp_grLASSO$Overall*100
adj_imp_grLASSO <- adj_imp_grLASSO %>% arrange(desc(Overall))

# add ranking column
adj_imp_grLASSO <- rank_importance(adj_imp_grLASSO)

# capitalize keyed abbreviations
adj_imp_grLASSO$Features <-  capitalize(adj_imp_grLASSO$Features, key)

adj_imp_grLASSO

saveRDS(adj_imp_grLASSO, "/your/path/adj_feature_imp_bal_grLASSO.rds")
write.csv2(adj_imp_grLASSO, "/your/path/adj_feature_imp_bal_grLasso.csv")



# non-Group LASSO
adj_imp_ngrLASSO <- varImp(final_models$balanced$lasso$glmnet.fit, lambda = final_models$balanced$lasso$lambda.min)

#condense one-hot encoded variables
adj_imp_ngrLASSO$Features <- rownames(adj_imp_ngrLASSO)
adj_imp_ngrLASSO <- group_imp(adj_imp_ngrLASSO, one_hot_mapping)
colnames(adj_imp_ngrLASSO) <- c("Features", "Overall")

#scale to max importance
adj_imp_ngrLASSO$Overall <- adj_imp_ngrLASSO$Overall/max(adj_imp_ngrLASSO$Overall)
adj_imp_ngrLASSO$Overall <- adj_imp_ngrLASSO$Overall*100
adj_imp_ngrLASSO <- adj_imp_ngrLASSO %>% arrange(desc(Overall))
colnames(adj_imp_ngrLASSO) <- c("Features", "Overall")

# add ranking column
adj_imp_ngrLASSO <- rank_importance(adj_imp_ngrLASSO)

# capitalize keyed abbreviations
adj_imp_ngrLASSO$Features <-  capitalize(adj_imp_ngrLASSO$Features, key)

adj_imp_ngrLASSO

saveRDS(adj_imp_ngrLASSO, "/your/path/adj_feature_imp_bal_ngrLASSO.rds")
write.csv2(adj_imp_ngrLASSO, "/your/path/adj_feature_imp_bal_ngrLasso.csv")


# Ridge
adj_imp_ridge <- varImp(final_models$balanced$ridge$glmnet.fit, lambda = final_models$balanced$ridge$lambda.min)
adj_imp_ridge <- adj_imp_ridge %>% arrange(desc(Overall))
adj_imp_ridge

#condense one-hot encoded variables
adj_imp_ridge$Features <- rownames(adj_imp_ridge)
adj_imp_ridge <- group_imp(adj_imp_ridge, one_hot_mapping)
colnames(adj_imp_ridge) <- c("Features", "Overall")

#scale to max importance
adj_imp_ridge$Overall <- adj_imp_ridge$Overall/max(adj_imp_ridge$Overall)
adj_imp_ridge$Overall <- adj_imp_ridge$Overall*100
adj_imp_ridge <- adj_imp_ridge %>% arrange(desc(Overall))

# add ranking column
adj_imp_ridge <- rank_importance(adj_imp_ridge)

# capitalize keyed abbreviations
adj_imp_ridge$Features <-  capitalize(adj_imp_ridge$Features, key)

adj_imp_ridge

saveRDS(adj_imp_ridge, "/your/path/adj_feature_imp_bal_ridge.rds")
write.csv2(adj_imp_ridge, "/your/path/adj_feature_imp_bal_ridge.csv")


# Elastic Net

adj_imp_ngrEN <- varImp(final_models$balanced$enet$glmnet.fit, lambda = final_models$balanced$enet$lambda.min)

#condense one-hot encoded variables
adj_imp_ngrEN$Features <- rownames(adj_imp_ngrEN)
adj_imp_ngrEN <- group_imp(adj_imp_ngrEN, one_hot_mapping)
colnames(adj_imp_ngrEN) <- c("Features", "Overall")

#scale to max importance
adj_imp_ngrEN$Overall <- adj_imp_ngrEN$Overall/max(adj_imp_ngrEN$Overall)
adj_imp_ngrEN$Overall <- adj_imp_ngrEN$Overall*100
adj_imp_ngrEN <- adj_imp_ngrEN %>% arrange(desc(Overall))

# add ranking column
adj_imp_ngrEN <- rank_importance(adj_imp_ngrEN)

# capitalize keyed abbreviations
adj_imp_ngrEN$Features <-  capitalize(adj_imp_ngrEN$Features, key)

adj_imp_ngrEN

saveRDS(adj_imp_ngrEN, "/your/path/adj_feature_imp_bal_EN.rds")
write.csv2(adj_imp_ngrEN, "/your/path/adj_feature_imp_bal_EN.csv")


# Random Forest
adj_imp_RF <- varImp(final_models$balanced$r_forest)
adj_imp_RF <- adj_imp_RF$importance %>% arrange(desc(Overall))
adj_imp_RF$Features <- rownames(adj_imp_RF)
adj_imp_RF

# add ranking column
adj_imp_RF <- rank_importance(adj_imp_RF)

# capitalize keyed abbreviations
adj_imp_RF$Features <-  capitalize(adj_imp_RF$Features, key)

adj_imp_RF

saveRDS(adj_imp_RF, "/your/path/adj_feature_imp_bal_Random_Forest.rds")
write.csv2(adj_imp_RF, "/your/path/adj_feature_imp_bal_Random_Forest.csv")


# Support Vector Machine
adj_imp_SVM <- varImp(final_models$balanced$svm)
adj_imp_SVM <- adj_imp_SVM$importance %>% arrange(desc(event))
adj_imp_SVM <- subset(adj_imp_SVM, select = - no_event)
colnames(adj_imp_SVM) <- c("Overall")
adj_imp_SVM$Features <- rownames(adj_imp_SVM)

# add ranking column
adj_imp_SVM <- rank_importance(adj_imp_SVM)

# capitalize keyed abbreviations
adj_imp_SVM$Features <-  capitalize(adj_imp_SVM$Features, key)

adj_imp_SVM 

saveRDS(adj_imp_SVM, "/your/path/adj_feature_imp_bal_SVM.rds")
write.csv2(adj_imp_SVM, "/your/path/adj_feature_imp_bal_SVM.csv")


# eXtreme Gradient Boosting Machine Linear Classifier
adj_imp_XGBMlin <- varImp(final_models$balanced$lin_xGBM) 
adj_imp_XGBMlin <- adj_imp_XGBMlin$importance 

#condense one-hot encoded variables
adj_imp_XGBMlin$Features <- rownames(adj_imp_XGBMlin)
adj_imp_XGBMlin <- group_imp(adj_imp_XGBMlin, one_hot_mapping)
colnames(adj_imp_XGBMlin) <- c("Features", "Overall")
adj_imp_XGBMlin <- adj_imp_XGBMlin %>% arrange(desc(Overall))

# add ranking column
adj_imp_XGBMlin <- rank_importance(adj_imp_XGBMlin)

# capitalize keyed abbreviations
adj_imp_XGBMlin$Features <-  capitalize(adj_imp_XGBMlin$Features, key)

adj_imp_XGBMlin

saveRDS(adj_imp_XGBMlin, "/your/path/adj_feature_imp_bal_xGBM_Lin.rds")
write.csv2(adj_imp_XGBMlin, "/your/path/adj_feature_imp_bal_xGBM_Lin.csv")


# eXtreme Gradient Boosting Machine Tree Classifier
adj_imp_XGBMtree <- varImp(final_models$balanced$tree_xGBM) 
adj_imp_XGBMtree <- adj_imp_XGBMtree$importance

#condense one-hot encoded variables
adj_imp_XGBMtree$Features <- rownames(adj_imp_XGBMtree)
adj_imp_XGBMtree <- group_imp(adj_imp_XGBMtree, one_hot_mapping)
colnames(adj_imp_XGBMtree) <- c("Features", "Overall")
adj_imp_XGBMtree <- adj_imp_XGBMtree %>% arrange(desc(Overall))

# add ranking column
adj_imp_XGBMtree <- rank_importance(adj_imp_XGBMtree)

# capitalize keyed abbreviations
adj_imp_XGBMtree$Features <-  capitalize(adj_imp_XGBMtree$Features, key)

adj_imp_XGBMtree

saveRDS(adj_imp_XGBMtree, "/your/path/adj_feature_imp_bal_xGBM_Tree.rds")
write.csv2(adj_imp_XGBMtree, "/your/path/adj_feature_imp_bal_xGBM_Tree.csv")






# ------------------------------------------------------------------------------
# create summary tables
# =====

# combine raw feature importance estimations
fimp_comb <- merge(adj_imp_grLASSO %>% select(-c(rank)), 
                   adj_imp_ngrLASSO %>% select(-c(rank)), 
                   by="Features", 
                   all=TRUE)
colnames(fimp_comb) <- c("Features","Group LASSO", "non-Group LASSO")

fimp_comb2 <- merge(adj_imp_ridge %>% select(-rank), 
                    adj_imp_ngrEN%>% select(-rank), 
                    by="Features", 
                    all=TRUE)
colnames(fimp_comb2) <- c("Features","Ridge", "Elastic Net")

fimp_comb3 <- merge(adj_imp_SVM %>% select(-rank), 
                    adj_imp_XGBMlin %>% select(-rank),  
                    by="Features", 
                    all=TRUE)
colnames(fimp_comb3) <- c("Features","SVM", "xGBM Linear")

fimp_comb4 <- merge(adj_imp_XGBMtree %>% select(-rank), 
                    adj_imp_RF %>% select(-rank),  
                    by="Features", 
                    all=TRUE)
colnames(fimp_comb4) <- c("Features","xGBM Tree", "Random Forest")




#combine all models grouped feature importances 
fimp_comb_all <- merge(fimp_comb, fimp_comb2, by="Features", all=TRUE)
fimp_comb_all <- merge(fimp_comb_all, fimp_comb3, by="Features", all=TRUE)
fimp_comb_all <- merge(fimp_comb_all, fimp_comb4, by="Features", all=TRUE)

#sort Decreasing by mean
row_means <- rowMeans(as.data.frame(fimp_comb_all[, c(2:9)]))
fimp_comb_all_sort <- fimp_comb_all[order(row_means, decreasing = TRUE),]
fimp_comb_all_sort[, c(2:9)] <- round(fimp_comb_all_sort[, c(2:9)], digits = 3)

write.csv2(fimp_comb_all_sort, "/your/path/feature_imp_all_grouped.csv")
saveRDS(fimp_comb_all_sort, "/your/path/fimp_all.rds")





# rank tables

# combine raw feature importance estimations
frank_comb <- merge(adj_imp_grLASSO %>% select(-c(Overall)), 
                   adj_imp_ngrLASSO %>% select(-c(Overall)), 
                   by="Features", 
                   all=TRUE)
colnames(frank_comb) <- c("Features","Group LASSO", "non-Group LASSO")

frank_comb2 <- merge(adj_imp_ridge %>% select(-Overall), 
                    adj_imp_ngrEN%>% select(-Overall), 
                    by="Features", 
                    all=TRUE)
colnames(frank_comb2) <- c("Features","Ridge", "Elastic Net")

frank_comb3 <- merge(adj_imp_SVM %>% select(-Overall), 
                    adj_imp_XGBMlin %>% select(-Overall),  
                    by="Features", 
                    all=TRUE)
colnames(frank_comb3) <- c("Features","SVM", "xGBM Linear")

frank_comb4 <- merge(adj_imp_XGBMtree %>% select(-Overall), 
                    adj_imp_RF %>% select(-Overall),  
                    by="Features", 
                    all=TRUE)
colnames(frank_comb4) <- c("Features","xGBM Tree", "Random Forest")




#combine all models grouped feature importances 
frank_comb_all <- merge(frank_comb, frank_comb2, by="Features", all=TRUE)
frank_comb_all <- merge(frank_comb_all, frank_comb3, by="Features", all=TRUE)
frank_comb_all <- merge(frank_comb_all, frank_comb4, by="Features", all=TRUE)

#sort Decreasing by mean
row_means <- rowMeans(as.data.frame(fimp_comb_all[, c(2:9)]))
frank_comb_all_sort <- frank_comb_all[order(row_means, decreasing = TRUE),]



write.csv2(frank_comb_all, "/your/path/feature_ranks_all_grouped.csv")
saveRDS(frank_comb_all, "/your/path/frank_all.rds")


################################################################################
################################################################################


fimp <- as.matrix(fimp_comb_all_sort[,2:9])
frank <- as.matrix(frank_comb_all_sort[,2:9])

combined <- matrix(paste0(fimp, " (", frank, ")"),
                   nrow = nrow(fimp), ncol = ncol(fimp))

rownames(combined) <- fimp_comb_all_sort$Features
colnames(combined) <- colnames(fimp_comb_all_sort[,2:9])

combined_df <- as.data.frame(combined)

combined_df <- cbind(Features = rownames(combined_df), combined_df)

#install.packages("knitr")
library(knitr)
install.packages("kableExtra")  # only needed once
library(kableExtra)
latex_table <- kable(combined_df, format = "latex", longtable = TRUE, booktabs = TRUE,
                     escape = FALSE, linesep = "", caption = "Combined FIMP")%>%
  kable_styling(latex_options = c("hold_position", "repeat_header"))

# Speichern
writeLines(latex_table, con = "/your/path/Feature_Importance.tex")



################################################################################
################################################################################




# Visualisation
# ==============================================================================


# Plot boxplots of feature Importance estimations (20 highest by mean)
fimp <-as.data.frame(fimp_comb_all)
rownames(fimp) <- fimp$Features
fimp <- fimp[, -which(names(fimp) == "Features")]

# identify rows with the top 5 importances per model
top_rows <- apply(fimp, 2, function(col) col %in% sort(col, decreasing = TRUE)[1:5])

top_fimp <- fimp[apply(top_rows, 1, any), ]

row_means <- rowMeans(as.data.frame(top_fimp))
sorted_fimp <- top_fimp[order(row_means, decreasing = TRUE),]

subsett <- sorted_fimp %>% t()
#subsett <- subsett[, 1:20]

# Create the boxplot
tiff("/your/path/Fimp_BoxPlot.tif", 
     units="px", width=2244, height=1496, res=356, compression = 'none')

par(mar = c(8, 4, 4, 2))
boxplot(subsett,
        main = "Boxplot for all Top 5 Feature Importances",
        xlab = "",
        ylab = "Relaticve Importance",
        las = 2, # Rotate x-axis labels
        col = rainbow(20),
        cex.axis = 0.8)

dev.off()

# Convert dataframe to long format
#df_long <- reshape2::melt(subsett)

# Create boxplot
#ggplot(df_long, aes(x = Var2, y = value)) +
#  geom_boxplot() +
#  labs(title = "Boxplot of DataFrame Columns",
#       x = "Columns",
#       y = "Values") +
#  coord_flip()+
#  theme_minimal()


# plot feature Ranks

frank <-as.data.frame(frank_comb_all)
rownames(frank) <- frank$Features
frank <- frank[, -which(names(frank) == "Features")]
frank_normalized <- as.data.frame(apply(frank, 2, function(x) x/max(x)))

# identify rows with the top 5 importances per model
top_rows <- apply(frank, 2, function(col) col %in% sort(col, decreasing = FALSE)[1:5])
top_rows_single <- apply(frank[, c(1:5)], 2, function(col) col %in% sort(col, decreasing = FALSE)[1:5])
top_rows_ensemb <- apply(frank[, c(6:8)], 2, function(col) col %in% sort(col, decreasing = FALSE)[1:5])

top_frank <- frank[apply(top_rows, 1, any), ]
top_frank_single <- frank[apply(top_rows_single, 1, any), ]
top_frank_ensemb <- frank[apply(top_rows_ensemb, 1, any), ]
top_frank_normalized <- frank_normalized[apply(top_rows, 1, any), ]
top_frank_normalized_single <- frank_normalized[apply(top_rows_single, 1, any), ]
top_frank_normalized_ensemb <- frank_normalized[apply(top_rows_ensemb, 1, any), ]

row_means <- rowMeans(as.data.frame(top_frank))
row_means_single <- rowMeans(as.data.frame(top_frank_single))
row_means_ensemb <- rowMeans(as.data.frame(top_frank_ensemb))
sorted_frank <- top_frank[order(row_means, decreasing = TRUE),]
sorted_frank_single <- top_frank_single[order(row_means_single, decreasing = TRUE),]
sorted_frank_ensemb <- top_frank_ensemb[order(row_means_ensemb, decreasing = TRUE),]
sorted_frank_normalized <- top_frank_normalized[order(row_means, decreasing = TRUE),]
sorted_frank_normalized_single <- top_frank_normalized_single[order(row_means_single, decreasing = TRUE),]
sorted_frank_normalized_ensemb <- top_frank_normalized_ensemb[order(row_means_ensemb, decreasing = TRUE),]

#row_means <- rowMeans(as.data.frame(frank_normalized))
#sorted_frank <- frank_normalized[order(row_means, decreasing = FALSE),]
#frank <- frank[order(row_means),]

sorted_frank$Features <- rownames(sorted_frank)
sorted_frank_single$Features <- rownames(sorted_frank_single)
sorted_frank_ensemb$Features <- rownames(sorted_frank_ensemb)
sorted_frank_normalized$Features <- rownames(sorted_frank_normalized)
sorted_frank_normalized_single$Features <- rownames(sorted_frank_normalized_single)
sorted_frank_normalized_ensemb$Features <- rownames(sorted_frank_normalized_ensemb)




# This mapping needs to get adjusted for other seeds/models!!!

# create shape mapping
group_shapes <- c("RCRI" = 15,
                  "clinic_cat_TCH" = 15,
                  "clinic_cat_GCH" = 15,
                  "liverCirrh" = 15,
                  "stdAnes" = 16,
                  "meanAnes" = 16,
                  "ASA" = 15,
                  "doseSurgSteroInf" = 16,
                  "doseInduPropoPerfu" = 16, 
                  "ltmDiu" = 15,
                  "age" = 16, 
                  "clinic_cat_NCH" = 15, 
                  "clinic_cat_URO" = 15,
                  "clinic_cat_ACH" = 15,
                  "anestDurat" = 16,
                  "eGFRPreSurg" = 16 
                  )

# create shape mapping
group_shapes_single <- c("RCRI" = 15,
                         "clinic_cat_TCH" = 15,
                         "clinic_cat_GCH" = 15,
                         "liverCirrh" = 15,
                         "ASA" = 15,
                         "doseInduPropoPerfu" = 16,
                         "ltmDiu" = 15, 
                         "clinic_cat_NCH" = 15, 
                         "clinic_cat_URO" = 15,
                         "clinic_cat_ACH" = 15,
                         "anestDurat" = 16,
                         "eGFRPreSurg" = 16 
)

# create shape mapping
group_shapes_ensemb <- c("stdAnes" = 16,
                         "meanAnes" = 16,
                         "doseSurgSteroInf" = 16,
                         "doseInduPropoPerfu" = 16,
                         "age" = 16,
                         "anestDurat" = 16,
                         "eGFRPreSurg" = 16 
)

type <- c("disc",
          "disc",
          "disc",
          "disc",
          "cont",
          "cont", 
          "disc", 
          "cont",  
          "cont", 
          "disc", 
          "cont",  
          "disc", 
          "disc", 
          "disc", 
          "cont", 
          "cont")

type_single <- c("disc",
          "disc",
          "disc",
          "disc",
          "disc",
          "cont", 
          "disc",
          "disc",  
          "disc",
          "disc",
          "cont",
          "cont")

type_ensemb <- c("cont",
          "cont",
          "cont",
          "cont",
          "cont",
          "cont",
          "cont" 
          )

sorted_frank_normalized$Type <- as.factor(type)
sorted_frank_normalized_single$Type <- as.factor(type_single)
sorted_frank_normalized_ensemb$Type <- as.factor(type_ensemb)
sorted_frank$Type <- as.factor(type)
sorted_frank_single$Type <- as.factor(type_single)
sorted_frank_ensemb$Type <- as.factor(type_ensemb)
names(sorted_frank_ensemb)[2] <- "LASSO"
names(sorted_frank_single)[2] <- "LASSO"



# ___________________
# CREATE RANK PLOTS
# ___________________

pdf("/your/path/fRANKS_norm.pdf", height = 6, width = 8)
#tiff("./../graphics/25_09_15_fRANKS_norm.tif", 
#     units="px", width=2244, height=1796, res=356, compression = 'none')

GGally::ggparcoord(sorted_frank_normalized,
                   columns = 1:8,
                   groupColumn = "Features",
                   scale = "globalminmax",
                   showPoints = TRUE,
                   mapping = ggplot2::aes(shape = factor(Type, levels = c(1, 2), labels = c("Continuous", "Discrete")))
                   #title = "Normalized Feature Importance Ranks"
)+
  #scale_color_manual(values=c25) +  # Assign specific colors
  #scale_shape_manual(values = c(19,19,17,19,17,17,17,17,19,19,19,17,17,17,19,19,17))+
  theme_minimal()+
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )+
  scale_y_reverse()+
  labs(x="Model", y="Rank",
       shape="Type"
       )+
  guides(shape = "none")

dev.off()



pdf("/your/path/fRANKS_single_norm.pdf", height = 6, width = 8)
#tiff("./../graphics/25_09_15_fRANKS_single_norm.tif", #     units="px", width=2244, height=1796, res=356, compression = 'none')

GGally::ggparcoord(sorted_frank_normalized_single,
                   columns = 1:8,
                   groupColumn = "Features",
                   scale = "globalminmax",
                   showPoints = TRUE,
                   mapping = ggplot2::aes(shape = factor(Type, levels = c(1, 2), labels = c("Continuous", "Discrete")))
                   #title = "Normalized Feature Importance Ranks"
)+
  #scale_color_manual(values=c25) +  # Assign specific colors
  theme_minimal()+
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )+
  scale_y_reverse()+
  labs(x="Model", y="Rank",shape="Type")+
  guides(shape = "none")

dev.off()



pdf("/your/path/fRANKS_ensemb_norm.pdf", height = 6, width = 8)
#tiff("./../graphics/25_09_15_fRANKS_ensemb_norm.tif", 
#     units="px", width=2244, height=1796, res=356, compression = 'none')

GGally::ggparcoord(sorted_frank_normalized_ensemb,
                   columns = 1:8,
                   groupColumn = "Features",
                   scale = "globalminmax",
                   showPoints = TRUE,
                   mapping = ggplot2::aes(shape = factor(Type, levels = c(1, 2), labels = c("Continuous", "Discrete")))
                   #title = "Normalized Feature Importance Ranks"
)+
  #scale_color_manual(values=c25) +  # Assign specific colors
  theme_minimal()+
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )+
  scale_y_reverse()+
  labs(x="Model", y="Rank",shape="Type")+
  guides(shape = "none")

dev.off()


c25 <- c(
  "dodgerblue2", "#E31A1C", # red
  "green4",
  "#6A3D9A", # purple
  "#FF7F00", # orange
  "black", "gold1",
  "skyblue2", "#FB9A99", # lt pink
  "palegreen2",
  "#FDBF6F", # lt orange
  "maroon", "orchid1", "deeppink1", "blue1", "steelblue4",
  "darkturquoise", "green1",
  "darkorange4"
)

c16 <- c("dodgerblue2", "#E31A1C", 
         "green4", "#6A3D9A", 
         "#FF7F00", "black", 
         "gold1", "skyblue2", 
         "palegreen2", "#FDBF6F", 
         "blue1", "maroon", 
         "orchid1", "darkturquoise", 
         "darkorange4", "deeppink1",
         "brown")

pdf("/your/path/fRANKS.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_15_fRANKS.tif", 
#     units="px", width=2244, height=1796, res=356, compression = 'none')

frank_plot_all <- GGally::ggparcoord(sorted_frank,
                   columns = 1:8,
                   groupColumn = "Features",
                   scale = "globalminmax",
                   showPoints = TRUE,
                   mapping = ggplot2::aes(shape = factor(Type, levels = c(1, 2), labels = c("Continuous", "Discrete")))
                   #title = "Feature Importance Ranks"
)+
  #scale_color_manual(values = c16) +  # Assign specific colors
  theme_minimal()+
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )+
  scale_y_reverse()+
  labs(x="Model", y="Rank", shape="Type")+
  guides(shape = "none")

frank_plot_all

dev.off()


pdf("/your/path/fRANKS_single.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_15_fRANKS_single.tif", 
#     units="px", width=2244, height=1796, res=356, compression = 'none')

frank_plot_single <- GGally::ggparcoord(sorted_frank_single,
                   columns = 1:8,
                   groupColumn = "Features",
                   scale = "globalminmax",
                   showPoints = TRUE,
                   mapping = ggplot2::aes(shape = factor(Type, levels = c(1, 2), labels = c("Continuous", "Discrete")))
                   #title = "Feature Importance Ranks"
)+
 # scale_color_manual(values = c16) +  # Assign specific colors
  theme_minimal()+
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )+
  scale_y_reverse()+
  labs(x="Model", y="Rank", shape="Type")+
  guides(shape = "none")

frank_plot_single

dev.off()


pdf("/your/path/fRANKS_ensemb.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_15_fRANKS_ensemb.tif", 
#     units="px", width=2244, height=1796, res=356, compression = 'none')

frank_plot_ensemb <- GGally::ggparcoord(sorted_frank_ensemb,
                   columns = 1:8,
                   groupColumn = "Features",
                   scale = "globalminmax",
                   showPoints = TRUE,
                   mapping = ggplot2::aes(shape = factor(Type, levels = c(1, 2), labels = c("Continuous", "Discrete")))
                   #title = "Feature Importance Ranks"
)+
  #scale_color_manual(values=c16) +  # Assign specific colors
  theme_minimal()+
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )+
  scale_y_reverse()+
  labs(x="Model", y="Rank", shape="Type")+
  guides(shape = "none")

frank_plot_ensemb

dev.off()


# combined plot
library(ggpubr)
pdf("/your/path/comb_fRANKS.pdf", height = 9, width = 6)
#tiff("./../graphics/25_09_15_comb_fRANKS.tif", 
#     units="px", width=2244, height=2991, res=356, compression = 'none')

frank_combined <- ggarrange(frank_plot_single, 
                            frank_plot_ensemb, 
                            ncol = 1, 
                            labels = c("a", "b"), 
                            common.legend = FALSE, 
                            legend = "right")
frank_combined

dev.off()

# Plot individual feature Importances

imp_grLASSO <- adj_imp_grLASSO %>% select(-rank) %>% as.data.frame()
rownames(imp_grLASSO) <- imp_grLASSO$Features
imp_ngrLASSO <- adj_imp_ngrLASSO %>% select(-rank) %>% as.data.frame()
rownames(imp_ngrLASSO) <- imp_ngrLASSO$Features
imp_ridge <- adj_imp_ridge %>% select(-rank) %>% as.data.frame()
rownames(imp_ridge) <- imp_ridge$Features
imp_ngrEN <- adj_imp_ngrEN %>% select(-rank) %>% as.data.frame()
rownames(imp_ngrEN) <- imp_ngrEN$Features
imp_RF <- adj_imp_RF %>% select(-rank) %>% as.data.frame()
rownames(imp_RF) <- imp_RF$Features
imp_SVM <- adj_imp_SVM %>% select(-rank) %>% as.data.frame()
rownames(imp_SVM) <- imp_SVM$Features
imp_XGBMlin <- adj_imp_XGBMlin %>% select(-rank) %>% as.data.frame()
rownames(imp_XGBMlin) <- imp_XGBMlin$Features
imp_XGBMtree <- adj_imp_XGBMtree %>% select(-rank) %>% as.data.frame()
rownames(imp_XGBMtree) <- imp_XGBMtree$Features

pdf("/your/path/fIMP_group_LASSO.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_15_fIMP_group_LASSO.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_grLASSO  <- ggplot(head(imp_grLASSO, 30),
                            aes(x=reorder(rownames(head(imp_grLASSO, 30)), Overall), 
                                y=Overall))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_grLASSO, 30)), xend = rownames(head(imp_grLASSO,30)), y=0, yend = Overall),
               color="skyblue")+
  xlab("Variable")+
  ylab("Overall Importance")+
  #ggtitle("Group Lasso Regression")+
  coord_flip()
impplot_grLASSO

dev.off()

saveRDS(impplot_grLASSO, "/your/path/feature_importance_grLASSO_plot.rds")


pdf("/your/path/fIMP_non-group_LASSO.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_15_fIMP_non-group_LASSO.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_ngrLASSO  <- ggplot(head(imp_ngrLASSO, 30),
                            aes(x=reorder(rownames(head(imp_ngrLASSO, 30)), Overall), 
                            y=Overall))+
                            geom_point(color="blue", size=4, alpha=0.6)+
                            geom_segment(aes(x=rownames(head(imp_ngrLASSO, 30)), xend = rownames(head(imp_ngrLASSO,30)), y=0, yend = Overall),
                                         color="skyblue")+
                            xlab("Variable")+
                            ylab("Overall Importance")+
                           # ggtitle("Non-Group Lasso Regression")+
                            coord_flip()
impplot_ngrLASSO

dev.off()

saveRDS(impplot_ngrLASSO, "/your/path/feature_importance_ngrLASSO_plot.rds")



pdf("/your/path/fIMP_Ridge.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_15_fIMP_Ridge.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_Ridge  <- ggplot(head(imp_ridge, 30),
                         aes(x=reorder(rownames(head(imp_ridge, 30)), Overall), 
                             y=Overall))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_ridge, 30)), xend = rownames(head(imp_ridge,30)), y=0, yend = Overall),
               color="skyblue")+
  xlab("Variable")+
  ylab("Overall Importance")+
  #ggtitle("Ridge Regression")+
  coord_flip()
impplot_Ridge

dev.off()

saveRDS(impplot_Ridge, "/your/path/feature_importance_Ridge_plot.rds")


pdf("/your/path/fIMP_non-group_EN.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_15_fIMP_non-group_EN.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_ngrEN  <- ggplot(head(imp_ngrEN, 30),
                            aes(x=reorder(rownames(head(imp_ngrEN, 30)), Overall), 
                                y=Overall))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_ngrEN, 30)), xend = rownames(head(imp_ngrEN,30)), y=0, yend = Overall),
               color="skyblue")+
  xlab("Variable")+
  ylab("Overall Importance")+
  #ggtitle("Elastic-Net Regression")+
  coord_flip()
impplot_ngrEN

dev.off()

saveRDS(impplot_ngrEN, "/your/path/feature_importance_ngrEN_plot.rds")



pdf("/your/path/fIMP_Random_Forest.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_15_fIMP_Random_Forest.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_RF <-  ggplot(head(imp_RF, 30), #change value to see more or less features
                      aes(x=reorder(rownames(head(imp_RF, 30)), Overall), 
                          y=Overall))+
                      geom_point(color="blue", size=4, alpha=0.6)+
                      geom_segment(aes(x=rownames(head(imp_RF, 30)), xend = rownames(head(imp_RF,30)), y=0, yend = Overall),
                                   color="skyblue")+
                      xlab("Variable")+
                      ylab("Overall Importance")+
                      #ggtitle("Random Forest")+
                      coord_flip()

impplot_RF

dev.off()

saveRDS(impplot_RF, "/your/path/feature_importance_RF_plot.rds")



pdf("/your/path/fIMP_SVM.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_15_fIMP_SVM.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_SVM <- ggplot(head(imp_SVM, 30), 
                      aes(x=reorder(rownames(head(imp_SVM, 30)), Overall), 
                          y=Overall))+
                      geom_point(color="blue", size=4, alpha=0.6)+
                      geom_segment(aes(x=rownames(head(imp_SVM, 30)), xend = rownames(head(imp_SVM, 30)), y=0, yend = Overall),
                                   color="skyblue")+
                      xlab("Variable")+
                      ylab("Overall Importance")+
                      #ggtitle("Support Vector Machine")+
                      coord_flip()

impplot_SVM

dev.off()

saveRDS(impplot_SVM, "/your/path/feature_importance_SVM_plot.rds")



pdf("/your/path/fIMP_xGBM_Linear.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_15_fIMP_xGBM_Linear.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_XGBMlin <-  ggplot(head(imp_XGBMlin, 30), 
                        aes(x=reorder(rownames(head(imp_XGBMlin, 30)), Overall), 
                            y=Overall))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_XGBMlin, 30)), xend = rownames(head(imp_XGBMlin, 30)), y=0, yend = Overall),
               color="skyblue")+
  xlab("Variable")+
  ylab("Overall Importance")+
  #ggtitle("Extreme Gradient Boosting Linear")+
  coord_flip()

impplot_XGBMlin

dev.off()

saveRDS(impplot_XGBMlin, "/your/path/feature_importance_XGBM_Linear_plot.rds")




pdf("/your/path/fIMP_xGBM_Tree.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_15_fIMP_xGBM_Tree.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_XGBMtree <-  ggplot(head(imp_XGBMtree, 30), 
                        aes(x=reorder(rownames(head(imp_XGBMtree, 30)), Overall), 
                            y=Overall))+
                        geom_point(color="blue", size=4, alpha=0.6)+
                        geom_segment(aes(x=rownames(head(imp_XGBMtree, 30)), xend = rownames(head(imp_XGBMtree, 30)), y=0, yend = Overall),
                                     color="skyblue")+
                        xlab("Variable")+
                        ylab("Overall Importance")+
                        #ggtitle("Extreme Gradient Boosting Tree")+
                        coord_flip()

impplot_XGBMtree

dev.off()

saveRDS(impplot_XGBMtree, "/your/path/feature_importance_XGBM_Tree_plot.rds")





# ----
# end



# Native Feature Importance Estimations
# ==========
# calculate importances

# Group LASSO
imp_grLASSO <- coef(final_models$balanced$group_lasso$gglasso.fit, s=final_models$balanced$group_lasso$lambda.min)
imp_grLASSO <- data.frame( "Importance" = abs(imp_grLASSO))
imp_grLASSO <- imp_grLASSO %>% arrange(desc(X1))
imp_grLASSO <- subset(imp_grLASSO, rownames(imp_grLASSO) != "(Intercept)") #remove intercept term 
colnames(imp_grLASSO) <- c("Overall")

#scale to max importance
imp_grLASSO$Overall <- imp_grLASSO$Overall/max(imp_grLASSO)
imp_grLASSO$Overall <- imp_grLASSO$Overall*100

# add ranking column
imp_grLASSO$Rank <- seq(from = 1, to = 73)

imp_grLASSO

saveRDS(imp_grLASSO, "/your/path/feature_imp_bal_grLASSO.rds")
write.csv2(imp_grLASSO, "/your/path/feature_imp_bal_grLasso.csv")


# non-Group LASSO
imp_ngrLASSO <- varImp(final_models$balanced$lasso$glmnet.fit, lambda = final_models$balanced$lasso$lambda.min)
imp_ngrLASSO <- imp_ngrLASSO %>% arrange(desc(Overall))
imp_ngrLASSO

#scale to max importance
imp_ngrLASSO$Overall <- imp_ngrLASSO$Overall/max(imp_ngrLASSO)
imp_ngrLASSO$Overall <- imp_ngrLASSO$Overall*100

# add ranking column
imp_ngrLASSO$Rank <- seq(from = 1, to = 73)

imp_ngrLASSO

saveRDS(imp_ngrLASSO, "/your/path/feature_imp_bal_ngrLASSO.rds")
write.csv2(imp_ngrLASSO, "/your/path/feature_imp_bal_ngrLasso.csv")


# Ridge
imp_ridge <- varImp(final_models$balanced$ridge$glmnet.fit, lambda = final_models$balanced$ridge$lambda.min)
imp_ridge <- imp_ridge %>% arrange(desc(Overall))
imp_ridge

#scale to max importance
imp_ridge$Overall <- imp_ridge$Overall/max(imp_ridge)
imp_ridge$Overall <- imp_ridge$Overall*100

# add ranking column
imp_ridge$Rank <- seq(from = 1, to = 73)

imp_ridge

saveRDS(imp_ngrLASSO, "/your/path/feature_imp_bal_ridge.rds")
write.csv2(imp_ngrLASSO, "/your/path/feature_imp_bal_ridge.csv")


# Elastic Net

imp_ngrEN <- varImp(final_models$balanced$enet$glmnet.fit, lambda = final_models$balanced$enet$lambda.min)
imp_ngrEN <- imp_ngrEN %>% arrange(desc(Overall))
imp_ngrEN

#scale to max importance
imp_ngrEN$Overall <- imp_ngrEN$Overall/max(imp_ngrEN)
imp_ngrEN$Overall <- imp_ngrEN$Overall*100

# add ranking column
imp_ngrEN$Rank <- seq(from = 1, to = 73)

imp_ngrEN

saveRDS(imp_ngrEN, "/your/path/feature_imp_bal_EN.rds")
write.csv2(imp_ngrEN, "/your/path/feature_imp_bal_EN.csv")


# Random Forest
imp_RF <- varImp(final_models$balanced$r_forest)
imp_RF <- imp_RF$importance %>% arrange(desc(Overall))
imp_RF

# add ranking column
imp_RF$Rank <- seq(from = 1, to = 62)

imp_RF

saveRDS(imp_RF, "/your/path/feature_imp_bal_Random_Forest.rds")
write.csv2(imp_RF, "/your/path/feature_imp_bal_Random_Forest.csv")


# Support Vector Machine
imp_SVM <- varImp(final_models$balanced$svm)
imp_SVM <- imp_SVM$importance %>% arrange(desc(event))

imp_SVM <- subset(imp_SVM, select = - no_event)

# add ranking column
imp_SVM$Rank <- seq(from = 1, to = 62)

imp_SVM 

saveRDS(imp_SVM, "/your/path/feature_imp_bal_SVM.rds")
write.csv2(imp_SVM, "/your/path/feature_imp_bal_SVM.csv")


# eXtreme Gradient Boosting Machine Linear Classifier
imp_XGBMlin <- varImp(final_models$balanced$lin_xGBM) 
imp_XGBMlin <- imp_XGBMlin$importance 

# add ranking column
imp_XGBMlin$Rank <- seq(from = 1, to = 73)

imp_XGBMlin

saveRDS(imp_XGBMlin, "/your/path/feature_imp_bal_xGBM_Lin.rds")
write.csv2(imp_XGBMlin, "/your/path/feature_imp_bal_xGBM_Lin.csv")


# eXtreme Gradient Boosting Machine Tree Classifier
imp_XGBMtree <- varImp(final_models$balanced$tree_xGBM) 
imp_XGBMtree <- imp_XGBMtree$importance

# add ranking column
imp_XGBMtree$Rank <- seq(from = 1, to = 73)

imp_XGBMtree

saveRDS(imp_XGBMtree, "/your/path/feature_imp_bal_xGBM_Tree.rds")
write.csv2(imp_XGBMtree, "/your/path/feature_imp_bal_xGBM_Tree.csv")


# combine raw feature importance estimations
fimp_comb <- merge(imp_grLASSO %>% select(-Rank), 
                   imp_ngrLASSO %>% select(-Rank), 
                   by="row.names", 
                   all=TRUE)
colnames(fimp_comb) <- c("Features","Group Lasso", "non-Group LASSO")
fimp_comb2 <- merge(imp_ridge %>% select(-Rank),
                    imp_ngrEN %>% select(-Rank), 
                    by="row.names", 
                    all=TRUE)
colnames(fimp_comb2) <- c("Features", "Ridge", "Elastic Net")
fimp_comb3 <- merge(imp_XGBMlin %>% select(-Rank), 
                    imp_XGBMtree%>% select(-Rank), 
                    by="row.names", 
                    all=TRUE)
colnames(fimp_comb3) <- c("Features","xGBM Linear", "xGBM Tree")

#combine all models feature importances where categorcal features were one-hot-encoded
fimp_comb <- merge(fimp_comb, fimp_comb2, by="Features", all=TRUE)
imp_XGBMtree["Features"] <- rownames(imp_XGBMtree)
fimp_comb <- merge(fimp_comb, fimp_comb3, by="Features", all=TRUE)
imp_XGBMtree["Features"] <- rownames(imp_XGBMtree)
#colnames(fimp_comb) <- c("Features","Group Lasso", "non-Group LASSO", "Ridge", "Elastic Net", "xGBM Linear", "xGBM Tree")

write.csv2(fimp_comb, "/your/path/feature_imp_oh_abs.csv")

#combine those that do not have one hot encoding
fimp_comb4 <- merge(imp_RF %>% select(-Rank), 
                    imp_SVM %>% select(-Rank),  
                    by="row.names", 
                    all=TRUE)
colnames(fimp_comb4) <- c("Features","Random Forest", "SVM")

write.csv2(fimp_comb4, "/your/path/feature_imp_abs.csv")


# grouping and averaging feature importances of one-hot-encoded features for direct comparison 

one_hot_mapping <- list(
  asa = c("asa_2", "asa_3", "asa_4", "asa_5"),
  decrease = c("decrease_1", "decrease_2"),
  nyha = c("nyha_1", "nyha_2", "nyha_3", "nyha_4"),
  rcri = c("rcri_1", "rcri_2", "rcri_3"),
  urgency = c("urgency_1", "urgency_2", "urgency_3")
)


# flatten mapping
mapping_df <- stack(one_hot_mapping) %>%
  rename(Features = values, original_feature = ind)

# melt data for easier processing
combined_importances <- fimp_comb %>% 
  pivot_longer(cols = -Features, names_to = "model", values_to = "importance") %>%
  left_join(mapping_df, by = "Features") %>%
  mutate(original_feature = coalesce(original_feature, Features)) %>%
  group_by(original_feature, model) %>%
  summarize(combined_importance = mean(importance, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = "model", values_from = "combined_importance")

print(combined_importances)

combined_importances <- as.data.frame(combined_importances)
colnames(combined_importances) <- c("Features", "Elastic Net", "Group Lasso", "Ridge", "non-Group LASSO", "xGBM Linear", "xGBM Tree")

all_combined_imp <- merge(combined_importances, fimp_comb3, by="Features", all=TRUE)
colnames(all_combined_imp) <- c("Features", "Elastic Net", "Group Lasso", "Ridge", "non-Group LASSO", "xGBM Linear", "xGBM Tree", "Random forest", "SVM")

saveRDS(all_combined_imp, "/your/path/fimp_all.rds")
write.csv2(all_combined_imp, "/your/path/feature_imp_all.csv")


# combine feature importance ranks
frank_comb <- merge(imp_grLASSO %>% select(-Overall), 
                    imp_ngrLASSO %>% select(-Overall), 
                    by="row.names", 
                    all=TRUE)
colnames(frank_comb) <- c("Features", "Group Lasso", "non-Group LASSO")
frank_comb2 <- merge(imp_ridge %>% select(-Overall), 
                     imp_ngrEN%>% select(-Overall), 
                     by="row.names", 
                     all=TRUE)
colnames(frank_comb2) <- c("Features", "Ridge", "Elastic Net")
frank_comb3 <- merge(imp_XGBMlin %>% select(-Overall), 
                     imp_XGBMtree%>% select(-Overall, -Features), 
                     by="row.names", 
                     all=TRUE)
colnames(frank_comb3) <- c("Features","xGBM Linear", "xGBM Tree")

#combine all models feature importances where categorcal features were one-hot-encoded
frank_comb <- merge(frank_comb, frank_comb2, by="Features", all=TRUE)
imp_XGBMtree["Features"] <- rownames(imp_XGBMtree)
frank_comb <- merge(frank_comb, frank_comb3, by="Features", all=TRUE)
colnames(frank_comb) <- c("Features", "Group Lasso", "non-Group LASSO", "Ridge", "Elastic Net", "xGBM Linear", "xGBM Tree")

write.csv2(fimp_comb, "/your/path/feature_ranks_oh.csv")

#combine those that do not have one hot encoding
frank_comb4 <- merge(imp_RF %>% select(-Overall), 
                     imp_SVM %>% select(-event),  
                     by="row.names", 
                     all=TRUE)
colnames(frank_comb4) <- c("Features","Random Forest", "SVM")

write.csv2(frank_comb4, "/your/path/feature_ranks.csv")


