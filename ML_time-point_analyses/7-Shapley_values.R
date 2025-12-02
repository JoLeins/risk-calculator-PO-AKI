# ==============================================================================
# Evaluation of ML-Models Feature Importance Estimations using Shapley-Values
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

capitalize <- function(column, key){
  column <- ifelse(column %in% names(key),
                   key[column],
                   column)
  return(column)
}


load_packages( c("tidyverse", "dplyr", "glmnet", "gglasso", "mltools", "data.table", "grpnet", "caret", "ggplot2", "ROCR", "tidymodels", "fastDummies", "ranger"))
setwd("/your/path/")
set.seed(3010)



# ______________________________________________________________________________

#                               Shapley Values

#_______________________________________________________________________________

#install.packages("shapr")
library(shapr)


# load development and validation data
devds_bal <- readRDS("/your/path/devds_balanced_prep.rds")
valds <- readRDS("/your/path/valds_prep.rds")

x_train <- devds_bal %>% select(-AKI_bin)
x_val <- valds %>% select(-AKI_bin)
y_train <- devds_bal %>% select(AKI_bin)


# load models
models <- readRDS("/your/path/final_models.rds")


# ______________________________________________________________________________
#install.packages("kernelshap")
library(kernelshap)

s1 <- kernelshap(models$balanced$svm, X = x_train, type = "prob")
s2 <- kernelshap(models$balanced$r_forest, X = x_train, type = "prob")

#levels(valds$AKI_bin) <- c("0", "1")

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

cont <- x_train %>% select(all_of(continuousNames))
disc <- x_train %>% select(- c(all_of(continuousNames)))
rm(continuousNames)


# ==========
# encoding discrete data in dummy variables
dummies <- fastDummies::dummy_cols(disc[, 2:5], remove_first_dummy = TRUE)
disc <- disc[,-c(2:5)] # remove multi-level features
disc <- cbind(disc, dummies[, 5:18])


# ==========
# convert predictor dataframes (X_train, X_test) to matrix
x_train <- as.matrix(cbind(disc, cont))

# convert Matrix from character to numeric
x_train <- apply(x_train, 2 ,as.numeric)

s3 <- kernelshap(models$balanced$lin_xGBM, X = x_train, type = "prob")
s4 <- kernelshap(models$balanced$tree_xGBM, X = x_train, type = "prob")
s5 <- kernelshap(models$balanced$group_lasso, X = x_train, type = "link")
s6 <- kernelshap(models$balanced$lasso, X = x_train, type = "link")
s7 <- kernelshap(models$balanced$ridge, X = x_train, type = "link")
s8 <- kernelshap(models$balanced$enet, X = x_train, type = "link")

# save s objects
saveRDS(s1, "/your/path/shap_SVM.rds")
saveRDS(s2, "/your/path/shap_RF.rds")
saveRDS(s3, "/your/path/shap_xGBM_lin.rds")
saveRDS(s4, "/your/path/shap_xGBM_tree.rds")
saveRDS(s5, "/your/path/shap_gLASSO.rds")
saveRDS(s6, "/your/path/shap_LASSO.rds")
saveRDS(s7, "/your/path/shap_Ridge.rds")
saveRDS(s8, "/your/path/shap_Enet.rds")

rm(s1, s2, s3, s4, s5, s6, s7, s8)


sh_svm <- readRDS("/your/path/shap_SVM.rds")
sh_rf <- readRDS("/your/path/shap_RF.rds")
sh_lin_xgbm <- readRDS("/your/path/shap_xGBM_lin.rds")
sh_tree_xgbm <- readRDS("/your/path/shap_xGBM_tree.rds")
sh_grLASSO <- readRDS("/your/path/shap_gLASSO.rds")
sh_LASSO <- readRDS("/your/path/shap_LASSO.rds")
sh_ridge <- readRDS("/your/path/shap_Ridge.rds")
sh_enet <- readRDS("/your/path/shap_Enet.rds")




# visualize shapley values
install.packages("shapviz")
library(shapviz)


# Analyze with shapviz

sh_svm <- shapviz(sh_svm)
sh_rf <- shapviz(sh_rf)
sh_lin_xgbm <- shapviz(sh_lin_xgbm)
sh_tree_xgbm <- shapviz(sh_tree_xgbm)
sh_grLASSO <- shapviz(sh_grLASSO)
sh_LASSO <- shapviz(sh_LASSO)
sh_ridge <- shapviz(sh_ridge)
sh_enet <- shapviz(sh_enet)




# extract average shapley values
sv_svm <- as.data.frame(sort(colMeans(abs(get_shap_values(sh_svm$event))), decreasing = TRUE))
sv_rf <- as.data.frame(sort(colMeans(abs(get_shap_values(sh_svm$event))), decreasing = TRUE))
sv_lin_xgbm <- as.data.frame(sort(colMeans(abs(get_shap_values(sh_lin_xgbm$`1`))), decreasing = TRUE))
sv_tree_xgbm <- as.data.frame(sort(colMeans(abs(get_shap_values(sh_tree_xgbm$`1`))), decreasing = TRUE))
sv_grLASSO <- as.data.frame(sort(colMeans(abs(get_shap_values(sh_grLASSO))), decreasing = TRUE))
sv_LASSO <- as.data.frame(sort(colMeans(abs(get_shap_values(sh_LASSO))), decreasing = TRUE))
sv_ridge <- as.data.frame(sort(colMeans(abs(get_shap_values(sh_ridge))), decreasing = TRUE))
sv_enet <- as.data.frame(sort(colMeans(abs(get_shap_values(sh_enet))), decreasing = TRUE))

colnames(sv_svm) <- c("SHAP")
colnames(sv_rf) <- c("SHAP")
colnames(sv_lin_xgbm) <- c("SHAP")
colnames(sv_tree_xgbm) <- c("SHAP")
colnames(sv_grLASSO) <- c("SHAP")
colnames(sv_LASSO) <- c("SHAP")
colnames(sv_ridge) <- c("SHAP")
colnames(sv_enet) <- c("SHAP")

# additively combine average shapley values of ohe features 


one_hot_mapping <- list(
  asa = c("asa_2", "asa_3", "asa_4", "asa_5"),
  nyha = c("nyha_1", "nyha_2", "nyha_3", "nyha_4"),
  rcri = c("rcri_1", "rcri_2", "rcri_3"),
  urgency = c("urgency_1", "urgency_2", "urgency_3")
)
# group one-hot encoded features' importance estimations for comparison
group_imp <- function(x, mapping){
  
  # Convert rownames to a column
  x <- x %>%
    rownames_to_column(var = "FeatureName")  # Extract rownames
  # flatten mapping
  mapping_df <- stack(mapping) %>%
    rename(FeatureName = values, Feature = ind)
  
  # melt data for easier processing
  combined_importances <- x %>% 
    left_join(mapping_df, by = "FeatureName") %>%
    mutate(Feature = coalesce(Feature, FeatureName)) %>%
    group_by(Feature) %>%
    dplyr::summarize(SHAP = sum(SHAP, na.rm = TRUE), .groups = "drop")
  
  return(combined_importances)
}

sv_svm <- group_imp(sv_svm, one_hot_mapping)
sv_rf <- group_imp(sv_rf, one_hot_mapping)
sv_lin_xgbm <- group_imp(sv_lin_xgbm, one_hot_mapping)
sv_tree_xgbm <- group_imp(sv_tree_xgbm, one_hot_mapping)
sv_grLASSO <- group_imp(sv_grLASSO, one_hot_mapping)
sv_LASSO <- group_imp(sv_LASSO, one_hot_mapping)
sv_ridge <- group_imp(sv_ridge, one_hot_mapping)
sv_enet <- group_imp(sv_enet, one_hot_mapping)



# add ranks to absolute values

# rank features based on estimated importance
rank_importance <- function(x){
  dataframe_with_ranks <- x %>%
    arrange(desc(SHAP)) %>%
    mutate(rank = ifelse(SHAP == 0, NA, row_number())) %>%
    fill(rank, .direction = "downup") %>%
    mutate(rank =ifelse(SHAP == 0, max(rank, na.rm = TRUE) + 1, rank)) %>%
    arrange(desc(SHAP))
  return(dataframe_with_ranks)
}

sv_svm <- rank_importance(sv_svm)
sv_rf <- rank_importance(sv_rf)
sv_lin_xgbm <- rank_importance(sv_lin_xgbm)
sv_tree_xgbm <- rank_importance(sv_tree_xgbm)
sv_grLASSO <- rank_importance(sv_grLASSO)
sv_LASSO <- rank_importance(sv_LASSO)
sv_ridge <- rank_importance(sv_ridge)
sv_enet <- rank_importance(sv_enet)


# Key of name replacements
key <- c("asa" = "ASA", "rcri" = "RCRI", "decrease" = "DECREASE", "bmi" = "BMI", "nyha" = "NYHA")

# capitalize keyed abbreviations
sv_svm$Feature <-  capitalize(sv_svm$Feature, key)
sv_rf$Feature <-  capitalize(sv_rf$Feature, key)
sv_lin_xgbm$Feature <-  capitalize(sv_lin_xgbm$Feature, key)
sv_tree_xgbm$Feature <-  capitalize(sv_tree_xgbm$Feature, key)
sv_grLASSO$Feature <-  capitalize(sv_grLASSO$Feature, key)
sv_LASSO$Feature <-  capitalize(sv_LASSO$Feature, key)
sv_ridge$Feature <-  capitalize(sv_ridge$Feature, key)
sv_enet$Feature <-  capitalize(sv_enet$Feature, key)


# join in one data-frame

combine_second_columns <- function(df_list, origin_df_names) {
  
  # Merge all second columns into one dataframe
  combined_df <- Reduce(function(x, y) full_join(x, y, by = colnames(df_list[[1]])[1]), df_list)
  
  # Set the first column as rownames
  combined_df <- combined_df %>%
    column_to_rownames(var = colnames(df_list[[1]])[1])
  
  # Rename columns using origin_df_names
  colnames(combined_df) <- origin_df_names
  
  return(combined_df)
}

df_shap <- list(as.data.frame(sv_grLASSO[, c(1,2)]), 
                as.data.frame(sv_LASSO[, c(1,2)]),
                as.data.frame(sv_ridge[, c(1,2)]),
                as.data.frame(sv_enet[, c(1,2)]),
                as.data.frame(sv_svm[, c(1,2)]),
                as.data.frame(sv_lin_xgbm[, c(1,2)]),
                as.data.frame(sv_tree_xgbm[, c(1,2)]),
                as.data.frame(sv_rf[, c(1,2)]))

df_sh_ranks <- list(as.data.frame(sv_grLASSO[, c(1,3)]), 
                    as.data.frame(sv_LASSO[, c(1,3)]),
                    as.data.frame(sv_ridge[, c(1,3)]),
                    as.data.frame(sv_enet[, c(1,3)]),
                    as.data.frame(sv_svm[, c(1,3)]),
                    as.data.frame(sv_lin_xgbm[, c(1,3)]),
                    as.data.frame(sv_tree_xgbm[, c(1,3)]),
                    as.data.frame(sv_rf[, c(1,3)]))

origin_df_names <- c("Group LASSO", "non-Group LASSO", 
                     "Ridge", "Elastic Net", 
                     "SVM", "xGBM Linear", 
                     "xGBM Tree", "Random Forest")

# Apply function
final_df_shap <- combine_second_columns(df_shap, origin_df_names)
final_df_ranks <- combine_second_columns(df_sh_ranks, origin_df_names)

row_means <- rowMeans(as.data.frame(final_df_shap))
final_df_shap_sort <- final_df_shap[order(row_means, decreasing = TRUE),]
final_df_ranks_sort <- final_df_ranks[order(row_means, decreasing = TRUE),]
final_df_shap_sort <- round(final_df_shap_sort, digits = 4)

write.csv2(final_df_shap_sort, "/your/path/Shap_values.csv")
saveRDS(final_df_shap_sort, "/your/path/Shap_values.rds")

write.csv2(final_df_ranks, "/your/path/Shap_ranks.csv")
saveRDS(final_df_ranks, "/your/path/Shap_ranks.rds")


################################################################################
################################################################################


fimp <- as.matrix(final_df_shap_sort)
frank <- as.matrix(final_df_ranks_sort)

combined <- matrix(paste0(fimp, " (", frank, ")"),
                   nrow = nrow(fimp), ncol = ncol(fimp))

rownames(combined) <- rownames(final_df_ranks_sort)
colnames(combined) <- colnames(final_df_ranks_sort)

combined_df <- as.data.frame(combined)

combined_df <- cbind(Features = rownames(combined_df), combined_df)

#install.packages("knitr")
library(knitr)
#install.packages("kableExtra")  # only needed once
library(kableExtra)
latex_table <- kable(combined_df, format = "latex", longtable = TRUE, booktabs = TRUE,
                     escape = FALSE, linesep = "", caption = "Combined FIMP")%>%
  kable_styling(latex_options = c("hold_position", "repeat_header"))

# Speichern
writeLines(latex_table, con = "/your/path/Shapley.tex")



################################################################################
################################################################################

# set up plots

# identify rows with the top 5 importances per model
top_rows <- apply(final_df_shap, 
                  2, 
                  function(col) col %in% sort(col, decreasing = TRUE)[1:5])
top_shap <- final_df_shap[apply(top_rows, 1, any), ]
row_means <- rowMeans(as.data.frame(top_shap))
top_shap <- top_shap[order(row_means, decreasing = TRUE),]

subsett <- top_shap %>% t()
#subsett <- subsett[, 1:20]

# Create the boxplot
#tiff("./../graphics/25_09_16_SHAP_BoxPlot.tif", 
#     units="px", width=2244, height=1496, res=356, compression = 'none')

# Create the boxplot
par(mar = c(8, 4, 4, 2))
boxplot(subsett,
        main = "Boxplot for all Top 5 Shapley Values",
        xlab = "",
        ylab = "Mean Shapley Values",
        las = 2, # Rotate x-axis labels
        col = rainbow(20),
        cex.axis = 0.8)

dev.off()


# ranks

# identify rows with the top 5 importances per model
top_rows <- apply(final_df_ranks, 2, function(col) col %in% sort(col, decreasing = FALSE)[1:5])
top_rows_single <- apply(final_df_ranks[, c(1:5)], 2, function(col) col %in% sort(col, decreasing = FALSE)[1:5])
top_rows_ensemb <- apply(final_df_ranks[, c(6:8)], 2, function(col) col %in% sort(col, decreasing = FALSE)[1:5])

top_rank <- final_df_ranks[apply(top_rows, 1, any), ]
top_rank_single <- final_df_ranks[apply(top_rows_single, 1, any), ]
top_rank_ensemb <- final_df_ranks[apply(top_rows_ensemb, 1, any), ]

row_means <- rowMeans(as.data.frame(top_rank))
row_means_single <- rowMeans(as.data.frame(top_rank_single))
row_means_ensemb <- rowMeans(as.data.frame(top_rank_ensemb))
top_rank <- top_rank[order(row_means, decreasing = TRUE),]
top_rank_single <- top_rank_single[order(row_means_single, decreasing = TRUE),]
top_rank_ensemb <- top_rank_ensemb[order(row_means_ensemb, decreasing = TRUE),]

top_rank$Features <- rownames(top_rank)
top_rank_single$Features <- rownames(top_rank_single)
top_rank_ensemb$Features <- rownames(top_rank_ensemb)

# create shape mapping
group_shapes <- c("stdAnes" = 16,
                  "age" = 16,
                  "clinic_cat_URO" = 15,
                  "ltmDiu" = 15,
                  "doseInduPropoPerfu" = 16,  
                  "ASA" = 15,
                  "anestDurat" = 16,
                  "eGFRPreSurg" = 16,
                  "clinic_cat_ACH" = 15
)

# create shape mapping
group_shapes_single <- c("clinic_cat_URO" = 15,
                         "ltmDiu" = 15,
                         "doseInduPropoPerfu" = 16,  
                         "asa" = 15,
                         "anestDurat" = 16,
                         "eGFRPreSurg" = 16,
                         "clinic_cat_ACH" = 15
)

# create shape mapping
group_shapes_ensemb <- c("stdAnes" = 16,
                         "age" = 16,
                         "doseInduPropoPerfu" = 16,
                         "asa" = 15,
                         "anestDurat" = 16,
                         "eGFRPreSurg" = 16,
                         "clinic_cat_ACH" = 15
)

type <- c("cont",
          "cont",
          "disc",
          "disc",
          "cont",
          "disc", 
          "cont",  
          "cont",  
          "disc" 
          )

type_single <- c("disc",
                "disc",
                "cont",
                "disc", 
                "cont",  
                "cont",  
                "disc" 
)

type_ensemb <- c("cont",
                 "cont",
                 "cont",
                 "disc", 
                 "cont",  
                 "cont",  
                 "disc" 
)

top_rank$Type <- as.factor(type)
top_rank_single$Type <- as.factor(type_single)
top_rank_ensemb$Type <- as.factor(type_ensemb)


final_df_shap$Feature <- rownames(final_df_shap)
shap_svm <- final_df_shap %>%
  select(Feature, SVM)%>%
  arrange(desc(SVM))


# plots 

pdf("/your/path/Shap_RANKS.pdf", height = 6, width = 8)
#tiff("./../graphics/25_09_16_Shap_RANKS.tif", 
#     units="px", width=2244, height=1683, res=356, compression = 'none')

GGally::ggparcoord(top_rank_ensemb,
                   columns = 1:8,
                   groupColumn = "Features",
                   scale = "globalminmax",
                   showPoints = TRUE,
                   mapping = ggplot2::aes(shape = factor(Type, levels = c(1, 2), labels = c("Continuous", "Discrete")))
                   #title = "Normalized Feature Importance Ranks"
)+
  #scale_color_manual(values=c25) +  # Assign specific colors
  theme_minimal(
  )+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, 
                                   #vjust = .5
                                   ))+
  scale_y_reverse()+
  labs(x="Model", y="Rank",shape="Type")+
  guides(shape = "none")

dev.off()


# ===========
# Full Importance plots
# ===========

# Preparation
imp_grLASSO <- sv_grLASSO %>% 
  select(-rank) %>% 
  as.data.frame() %>% 
  arrange(desc(SHAP))
#scale to max importance
imp_grLASSO$SHAP <- imp_grLASSO$SHAP/max(imp_grLASSO$SHAP)
imp_grLASSO$SHAP <- imp_grLASSO$SHAP*100
rownames(imp_grLASSO) <- imp_grLASSO$Feature

imp_ngrLASSO <- sv_LASSO %>% 
  select(-rank) %>% 
  as.data.frame() %>% 
  arrange(desc(SHAP))
#scale to max importance
imp_ngrLASSO$SHAP <- imp_ngrLASSO$SHAP/max(imp_ngrLASSO$SHAP)
imp_ngrLASSO$SHAP <- imp_ngrLASSO$SHAP*100
rownames(imp_ngrLASSO) <- imp_ngrLASSO$Feature

imp_ridge <- sv_ridge %>% 
  select(-rank) %>% 
  as.data.frame() %>% 
  arrange(desc(SHAP))
#scale to max importance
imp_ridge$SHAP <- imp_ridge$SHAP/max(imp_ridge$SHAP)
imp_ridge$SHAP <- imp_ridge$SHAP*100
rownames(imp_ridge) <- imp_ridge$Feature

imp_ngrEN <- sv_enet %>% 
  select(-rank) %>% 
  as.data.frame() %>% 
  arrange(desc(SHAP))
#scale to max importance
imp_ngrEN$SHAP <- imp_ngrEN$SHAP/max(imp_ngrEN$SHAP)
imp_ngrEN$SHAP <- imp_ngrEN$SHAP*100
rownames(imp_ngrEN) <- imp_ngrEN$Feature

imp_RF <- sv_rf %>% 
  select(-rank) %>% 
  as.data.frame() %>% 
  arrange(desc(SHAP))
#scale to max importance
imp_RF$SHAP <- imp_RF$SHAP/max(imp_RF$SHAP)
imp_RF$SHAP <- imp_RF$SHAP*100
rownames(imp_RF) <- imp_RF$Feature

imp_SVM <- sv_svm %>% 
  select(-rank) %>% 
  as.data.frame() %>% 
  arrange(desc(SHAP))
#scale to max importance
imp_SVM$SHAP <- imp_SVM$SHAP/max(imp_SVM$SHAP)
imp_SVM$SHAP <- imp_SVM$SHAP*100
rownames(imp_SVM) <- imp_SVM$Feature

imp_XGBMlin <- sv_lin_xgbm %>% 
  select(-rank) %>% 
  as.data.frame() %>% 
  arrange(desc(SHAP))
#scale to max importance
imp_XGBMlin$SHAP <- imp_XGBMlin$SHAP/max(imp_XGBMlin$SHAP)
imp_XGBMlin$SHAP <- imp_XGBMlin$SHAP*100
rownames(imp_XGBMlin) <- imp_XGBMlin$Feature

imp_XGBMtree <- sv_tree_xgbm %>% 
  select(-rank) %>% 
  as.data.frame() %>% 
  arrange(desc(SHAP))
#scale to max importance
imp_XGBMtree$SHAP <- imp_XGBMtree$SHAP/max(imp_XGBMtree$SHAP)
imp_XGBMtree$SHAP <- imp_XGBMtree$SHAP*100
rownames(imp_XGBMtree) <- imp_XGBMtree$Feature


# Plotting
pdf("/your/path/SHAP_SVM.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_16_SHAP_SVM.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_SVM  <- ggplot(head(imp_SVM, 30),
                         aes(x=reorder(rownames(head(imp_SVM, 30)), SHAP), 
                             y=SHAP))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_SVM, 30)), xend = rownames(head(imp_SVM,30)), y=0, yend = SHAP),
               color="skyblue")+
  xlab("Variable")+
  ylab("SHAP Importance")+
  #ggtitle("Ridge Regression")+
  coord_flip()
impplot_SVM

dev.off()



pdf("/your/path/SHAP_RF.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_16_SHAP_RF.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_RF  <- ggplot(head(imp_RF, 30),
                       aes(x=reorder(rownames(head(imp_RF, 30)), SHAP), 
                           y=SHAP))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_RF, 30)), xend = rownames(head(imp_RF,30)), y=0, yend = SHAP),
               color="skyblue")+
  xlab("Variable")+
  ylab("SHAP Importance")+
  #ggtitle("Ridge Regression")+
  coord_flip()
impplot_RF

dev.off()



pdf("/your/path/SHAP_XGBMlin.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_16_SHAP_XGBMlin.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_XGBMlin  <- ggplot(head(imp_XGBMlin, 30),
                       aes(x=reorder(rownames(head(imp_XGBMlin, 30)), SHAP), 
                           y=SHAP))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_XGBMlin, 30)), xend = rownames(head(imp_XGBMlin,30)), y=0, yend = SHAP),
               color="skyblue")+
  xlab("Variable")+
  ylab("SHAP Importance")+
  #ggtitle("Ridge Regression")+
  coord_flip()
impplot_XGBMlin

dev.off()



pdf("/your/path/SHAP_XGBMtree.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_16_SHAP_XGBMtree.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_XGBMtree  <- ggplot(head(imp_XGBMtree, 30),
                       aes(x=reorder(rownames(head(imp_XGBMtree, 30)), SHAP), 
                           y=SHAP))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_XGBMtree, 30)), xend = rownames(head(imp_XGBMtree,30)), y=0, yend = SHAP),
               color="skyblue")+
  xlab("Variable")+
  ylab("SHAP Importance")+
  #ggtitle("Ridge Regression")+
  coord_flip()
impplot_XGBMtree

dev.off()



pdf("/your/path/SHAP_grLASSO.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_16_SHAP_grLASSO.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_grLASSO  <- ggplot(head(imp_grLASSO, 30),
                       aes(x=reorder(rownames(head(imp_grLASSO, 30)), SHAP), 
                           y=SHAP))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_grLASSO, 30)), xend = rownames(head(imp_grLASSO,30)), y=0, yend = SHAP),
               color="skyblue")+
  xlab("Variable")+
  ylab("SHAP Importance")+
  #ggtitle("Ridge Regression")+
  coord_flip()
impplot_grLASSO

dev.off()




pdf("/your/path/SHAP_ngrLASSO.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_16_SHAP_ngrLASSO.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_ngrLASSO  <- ggplot(head(imp_ngrLASSO, 30),
                       aes(x=reorder(rownames(head(imp_ngrLASSO, 30)), SHAP), 
                           y=SHAP))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_ngrLASSO, 30)), xend = rownames(head(imp_ngrLASSO,30)), y=0, yend = SHAP),
               color="skyblue")+
  xlab("Variable")+
  ylab("SHAP Importance")+
  #ggtitle("Ridge Regression")+
  coord_flip()
impplot_ngrLASSO

dev.off()




pdf("/your/path/SHAP_ngrEN.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_16_SHAP_ngrEN.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_ngrEN  <- ggplot(head(imp_ngrEN, 30),
                       aes(x=reorder(rownames(head(imp_ngrEN, 30)), SHAP), 
                           y=SHAP))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_ngrEN, 30)), xend = rownames(head(imp_ngrEN,30)), y=0, yend = SHAP),
               color="skyblue")+
  xlab("Variable")+
  ylab("SHAP Importance")+
  #ggtitle("Ridge Regression")+
  coord_flip()
impplot_ngrEN

dev.off()




pdf("/your/path/SHAP_ridge.pdf", height = 6, width = 6)
#tiff("./../graphics/25_09_16_SHAP_ridge.tif", 
#     units="px", width=2244, height=2244, res=356, compression = 'none')

impplot_ridge  <- ggplot(head(imp_ridge, 30),
                       aes(x=reorder(rownames(head(imp_ridge, 30)), SHAP), 
                           y=SHAP))+
  geom_point(color="blue", size=4, alpha=0.6)+
  geom_segment(aes(x=rownames(head(imp_ridge, 30)), xend = rownames(head(imp_ridge,30)), y=0, yend = SHAP),
               color="skyblue")+
  xlab("Variable")+
  ylab("SHAP Importance")+
  #ggtitle("Ridge Regression")+
  coord_flip()
impplot_ridge
dev.off()

# ----
# end
