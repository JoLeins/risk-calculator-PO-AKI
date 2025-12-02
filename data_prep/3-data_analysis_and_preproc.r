# ==============================================================================
# Explorative analysis, visualization and subsequent preprocessing of development
# and validation data sets.
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
    #print(cont_devds[, name])
    column_means <- append(column_means, mean(unlist(table[, name])))# insert mean for every column in empty mean vector
    column_sd <- append(column_sd, sd(unlist(table[, name])))# insert sd for every column in empty sd vector
    column_min <- append(column_min, min(unlist(table[, name])))# insert sd for every column in empty sd vector
    column_max <- append(column_max, max(unlist(table[, name])))# insert sd for every column in empty sd vector
  }
  
  cont_summary_table <- devds.frame(column_names, column_means, column_sd, column_min, column_max, column_content)
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

# Create a function to scale devds using stored mean and standard deviation

scale_with_params <- function(devds, mean_vals, sd_vals) {
  scaled_devds <- scale(devds, center = mean_vals, scale = sd_vals)
  return(scaled_devds)
}



# flattenCorrMatrix
# cormat : matrix of the correlation coefficients
# pmat : matrix of the correlation p-values
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

# x is a matrix containing the data
# method : correlation method. "pearson"" or "spearman"" is supported
# removeTriangle : remove upper or lower triangle
# results :  if "html" or "latex"
# the results will be displayed in html or latex format
corstars <-function(x, method=c("pearson", "spearman"), removeTriangle=c("upper", "lower"),
                    result=c("none", "html", "latex")){
  #Compute correlation matrix
  require(Hmisc)
  x <- as.matrix(x)
  correlation_matrix<-rcorr(x, type=method[1])
  R <- correlation_matrix$r # Matrix of correlation coeficients
  p <- correlation_matrix$P # Matrix of p-value 
  
  ## Define notions for significance levels; spacing is important.
  mystars <- ifelse(p < .0001, "****", ifelse(p < .001, "*** ", ifelse(p < .01, "**  ", ifelse(p < .05, "*   ", "    "))))
  
  ## trunctuate the correlation matrix to two decimal
  R <- format(round(cbind(rep(-1.11, ncol(x)), R), 2))[,-1]
  
  ## build a new matrix that includes the correlations with their apropriate stars
  Rnew <- matrix(paste(R, mystars, sep=""), ncol=ncol(x))
  diag(Rnew) <- paste(diag(R), " ", sep="")
  rownames(Rnew) <- colnames(x)
  colnames(Rnew) <- paste(colnames(x), "", sep="")
  
  ## remove upper triangle of correlation matrix
  if(removeTriangle[1]=="upper"){
    Rnew <- as.matrix(Rnew)
    Rnew[upper.tri(Rnew, diag = TRUE)] <- ""
    Rnew <- as.data.frame(Rnew)
  }
  
  ## remove lower triangle of correlation matrix
  else if(removeTriangle[1]=="lower"){
    Rnew <- as.matrix(Rnew)
    Rnew[lower.tri(Rnew, diag = TRUE)] <- ""
    Rnew <- as.data.frame(Rnew)
  }
  
  ## remove last column and return the correlation matrix
  Rnew <- cbind(Rnew[1:length(Rnew)-1])
  if (result[1]=="none") return(Rnew)
  else{
    if(result[1]=="html") print(xtable(Rnew), type="html")
    else print(xtable(Rnew), type="latex") 
  }
} 



# ----
# setup

# load packages

load_packages( c("tidyverse","dplyr", "ggpubr", "ggplot2", "ggcorrplot", "caret", "gtsummary", "writexl", "Hmisc", "corrplot", "psych"))



set.seed(3010)
setwd("/Your/Path/")








# ----
# main

# ==============================================================================
# Explorative data analysis of the original development dataset
# ==============================================================================
# you could also use the full cohorts from "2-data_cleaning"
devds <- readRDS("/Your/Path/reduced_training_cohort.rds") # read devds from rds
valds <- readRDS("/Your/Path/reduced_validation_cohort.rds") # read valds from rds
str(devds) # investigate column types
colnames(devds)



# inspect data distributions

for(i in 1:ncol(devds)){
  print(names(devds)[i])
  print(table(devds[,i], useNA = "ifany"))
  cat("\n")
}

Hmisc::describe(devds)



#split continuous and categorical for further analyses

# define list of continuos feature names
cont_names <- c("inductDurat",
                "surgDurat",
                "anestDurat",
                "age",
                "weight",
                "height",
                "bmi",
                #sex,
                #rcri,
                #urgency,
                #decrease,
                #asa,
                #nyha,
                #chf,
                #cad,
                #cvd,
                #pad,
                #diab, # was davon?
                #diab_bin,
                #liverCirrh,
                #ekg,
                #ltmACE,
                #ltmSartan,
                #ltmBB,
                #ltmCCB,
                #ltmBig,
                #ltmIns,
                #ltmDiu,
                #ltmStat,
                #iuc, 
                #stomTube,
                "doseInduEtomidateBolus",
                "doseInduPropoBolus",
                "doseInduPropoPerfu",
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
                #medInduKateBolus_bin,
                #medSurgAtroBolus_bin,
                #medSurgKateBolus_bin,
                #clinic_cat_ACH,
                #clinic_cat_GCH,
                #clinic_cat_HCH,
                #clinic_cat_HNO,
                #clinic_cat_NCH,
                #clinic_cat_PWC,
                #clinic_cat_TCH,
                #clinic_cat_UCH,
                #clinic_cat_URO,
                #clinic_cat_Other,
                "minMAPcumu1MinAnes",
                "minMAPcumu5MinAnes",
                "aucMAPunder65Anes",
                "twaMAPunder65Anes",
                "meanAnes",              
                "stdAnes",                
                "entropyAnes",            
                "trendAnes",             
                "kurtosisAnes",           
                "skewnessAnes",
                "baselineMAP"
                #AKI_bin
)

cont <- devds %>% select(one_of(cont_names))

disc<- devds %>% select(-one_of(cont_names))


# adjust names for figures 
colnames(cont)[7] <- "BMI"
colnames(disc)[c(2,4,5,6)] <- c("RCRI", "DECREASE", "ASA", "NYHA")

# visualize continuous distributions

# create QQplots and Histograms of all non transformed and standardized variables
qq_and_hist(cont, "/Your/Path/dev_cont_QQ_plots_and_Histograms.pdf")


# investigate ratio of nonzero-values
columnnames_disc <- colnames(disc)
nz_disc <- colSums(disc != 0)/length(disc[, 1])

nonZero_disc <- data.frame(columnnames_disc, nz_disc)
below5pNonZero <- nonZero_disc[nonZero_disc[, 2] < 0.05, ]




# create Barcharts of all variables
pdf(file = "/Your/Path/discrete_dist_dev.pdf")

par(mfrow = c(2, 2))

for (name in colnames(disc)) {
  # Plot barchart
  plot_bar <- ggplot(data = disc, aes(x = .data[[name]])) +
    geom_bar() +
    ggtitle(name)  # Add title with column name
  plot(plot_bar)
}

dev.off()

rm(below5pNonZero, nonZero_disc, plot_bar, columnnames_disc, cont_names, name, nz_disc)


# ==============================================================================
# multicollinearity analyses of the original dataset
# ==============================================================================



# correlation matrix
res <- round(cor(cont), 2)
is.na(res) <- abs(res) < 0.6
testplot= ggcorrplot(res, 
                     #method = "circle", # change plot to circle
                     type = "lower", 
                     lab_size= 2, 
                     tl.cex = 8,
                     lab = "true", #add correlation coefficient
                     outline.col = "black", 
                     tl.col = "black")+
  theme(axis.text.y=element_text(size=9),
        axis.text.x = element_text(size=9, angle=35))
print(testplot)

# correlation matrix with significance levels
res2 <- rcorr(as.matrix(cont))
corrmat_cc <- res2
#is.na(corrmat_cc$r) <- abs(corrmat_cc2$r) < 0.6

pdf(file = "/Your/Path/vgl_test_dev_short.pdf",   # The directory you want to save the file in
    width = 10, # The width of the plot in inches
    height = 10) # The height of the plot in inches

corrplot_cc= ggcorrplot(res2$r, 
                     #method = "circle", # change plot to circle
                     type = "lower", 
                     lab_size= 2, 
                     tl.cex = 8,
                     lab = "true", #add correlation coefficient
                     outline.col = "black", 
                     tl.col = "black")+
  theme(axis.text.y=element_text(size=9),
        axis.text.x = element_text(size=9, angle=35))
print(corrplot_cc)

# Run dev.off() to create the file!
dev.off()



# =========
# visualizing as correlogram with insignificant correlation coefficients left out

res2 <- rcorr(as.matrix(cont))
diag(res2$r) <- 0 # set NAs in diagonal to 0 to enable plot

# open pdf file
pdf(file = "/Your/Path/vgl_test__corrplot_dev.pdf",   # The directory you want to save the file in
    width = 10, # The width of the plot in inches
    height = 10) # The height of the plot in inches

# draw plot
corrplot(res2$r, type="upper", order="hclust", 
          sig.level = 0.01, insig = "blank", tl.cex = 0.6, cl.cex = 0.6
         )

# Run dev.off() to create the file!
dev.off()


# create flattened matrix
flattenedCorrMatrix <- flattenCorrMatrix(res2$r, res2$P)
write.csv(flattenedCorrMatrix, "/Your/Path/flattenedCorrMatrix.csv")

rm(res, res2, flattenedCorrMatrix, testplot)

# ==============================================================================
# discrete devds
# ==============================================================================

# inspect devds
colSums(is.na(disc) > 0,) # check for missing values
str(disc) # devds is still numeric
colnames(disc) # check features

# change devdstype to factor

disc <- disc %>%
  mutate(across(where(is.numeric), as.factor)) # convert all numeric columns to factor

str(disc)

ord <- subset(disc, select = c(RCRI, DECREASE, NYHA, diab, urgency, ASA))
binary <- subset(disc, select = -c(RCRI, DECREASE, NYHA, diab, urgency, ASA))

rm(disc)


# ==========
# testing for correlation between binary variables     
summary_df_phi <- data.frame(0:31)
for(i in 1:ncol(binary)){
  namea <- colnames(binary)[i]
  newcol <- c()
  newcolphi <- c()
  for(e in 1:ncol(binary)){
    nameb <- colnames(binary)[e]
    
    # phi
    phi <- phi(table(binary[[namea]], binary[[nameb]]))
    newcolphi <- append(newcolphi, phi)
    
  }
  
  summary_df_phi <- cbind(summary_df_phi, newcolphi)
}


summary_df_phi <- summary_df_phi[, 2:33] # remove numbering column
colnames(summary_df_phi) <- colnames(binary) # attach correct names
rownames(summary_df_phi) <- colnames(binary) # attach correct names

Rnew1 <- as.matrix(summary_df_phi)
Rnew1[upper.tri(Rnew1, diag = TRUE)] <- ""
Rnew1 <- as.data.frame(Rnew1)

write.csv2(Rnew1, "/Your/Path/phi.csv") # save table




# convert factors to numeric
ord <- ord %>%
  mutate(across(where(is.factor), as.numeric)) # convert all numeric columns to factor
binary <- binary %>%
  mutate(across(where(is.factor), as.numeric)) # convert all numeric columns to factor

# ==========
# testing for correlation between ordinal and binary variables
summary_df_tau <- data.frame(1:32)
for(i in 1:ncol(ord)){
  namea <- colnames(ord)[i]
  newcol <- c()
  for(e in 1:ncol(binary)){
    nameb <- colnames(binary)[e]
    
    # kendalls tau
    tau <- cor(ord[[namea]], binary[[nameb]], method = "kendall")
    newcol <- append(newcol, tau)
    
  }
  
  summary_df_tau <- cbind(summary_df_tau, newcol)
}

summary_df_tau <- summary_df_tau[, 2:7] # remove numbering column
colnames(summary_df_tau) <- colnames(ord) # attach correct names
rownames(summary_df_tau) <- colnames(binary) # attach correct names

Rnew2 <- as.matrix(summary_df_tau)
Rnew2[upper.tri(Rnew2, diag = TRUE)] <- ""
Rnew2 <- as.data.frame(Rnew2)


write.csv2(Rnew2, "/Your/Path/tau_ord_and_bin.csv") # save as table




# ==========
# testing for correlation between ordinal variables
summary_df_ord_tau <- data.frame(1:6)
for(i in 1:ncol(ord)){
  namea <- colnames(ord)[i]
  newcol <- c()
  for(e in 1:ncol(ord)){
    nameb <- colnames(ord)[e]
    
    # kendalls tau
    tau <- cor(ord[[namea]], ord[[nameb]], method = "kendall")
    newcol <- append(newcol, tau)
    
  }
  
  summary_df_ord_tau <- cbind(summary_df_ord_tau, newcol)
}

summary_df_ord_tau <- summary_df_ord_tau[, 2:7] # remove numbering column
colnames(summary_df_ord_tau) <- colnames(ord) # attach correct names
rownames(summary_df_ord_tau) <- colnames(ord) # attach correct names

Rnew3 <- as.matrix(summary_df_ord_tau)
Rnew3[upper.tri(Rnew3, diag = TRUE)] <- ""
Rnew3 <- as.data.frame(Rnew3)


write.csv2(Rnew3, "/Your/Path/tau_ord_and_ord.csv")


# visualize corrmats

corrmat_bb <- summary_df_phi
corrmat_1bb <- corrmat_bb
is.na(corrmat_1bb) <- abs(corrmat_1bb) < 0.3


# open pdf file
pdf(file = "/Your/Path/correlation_corrplot_BB_dev.pdf",   # The directory you want to save the file in
    width = 5, # The width of the plot in inches
    height = 5) # The height of the plot in inches

corrplot_bb= ggcorrplot(corrmat_bb, 
                     #method = "circle", # change plot to circle
                     type = "lower", 
                     lab_size= 2, 
                     tl.cex = 8,
                     lab = "true", #add correlation coefficient
                     outline.col = "black", 
                     tl.col = "black")+
  theme(axis.text.y=element_text(size=9),
        axis.text.x = element_text(size=9, angle=35))
print(corrplot_bb)

# Run dev.off() to create the file!
dev.off()



corrmat_bo <- summary_df_tau
corrmat_bo1 <- corrmat_bo
is.na(corrmat_bo1) <- abs(corrmat_bo1) < 0.3


# open pdf file
pdf(file = "/Your/Path/correlation_corrplot_BO_dev.pdf",   # The directory you want to save the file in
    width = 5, # The width of the plot in inches
    height = 5) # The height of the plot in inches

corrplot_bo= ggcorrplot(corrmat_bo1, 
                     #method = "circle", # change plot to circle
                     type = "lower", 
                     lab_size= 2, 
                     tl.cex = 8,
                     lab = "true", #add correlation coefficient
                     outline.col = "black", 
                     tl.col = "black")+
  theme(axis.text.y=element_text(size=9),
        axis.text.x = element_text(size=9, angle=35))
print(corrplot_bo)

# Run dev.off() to create the file!
dev.off()



corrmat_oo <- summary_df_ord_tau
corrmat_oo1 <- corrmat_oo
is.na(corrmat_oo1) <- abs(corrmat_oo1) < 0.3

# open pdf file
pdf(file = "/Your/Path/correlation_corrplot_OO_dev.pdf",   # The directory you want to save the file in
    width = 5, # The width of the plot in inches
    height = 5) # The height of the plot in inches


corrplot_oo= ggcorrplot(corrmat_oo1, 
                     #method = "circle", # change plot to circle
                     type = "lower", 
                     lab_size= 2, 
                     tl.cex = 8,
                     lab = "true", #add correlation coefficient
                     outline.col = "black", 
                     tl.col = "black")+
  theme(axis.text.y=element_text(size=9),
        axis.text.x = element_text(size=9, angle=35))
print(corrplot_oo)

# Run dev.off() to create the file!
dev.off()





# continuous, binary

corrmat_cb <- biserial(cont, binary)
corrmat_1cb <- corrmat_cb
is.na(corrmat_1cb) <- abs(corrmat_1cb) < 0.3

# open pdf file
pdf(file = "/Your/Path/correlation_corrplot_cb_dev.pdf",   # The directory you want to save the file in
    width = 5, # The width of the plot in inches
    height = 5) # The height of the plot in inches

corrplot_cb= ggcorrplot(corrmat_1cb, 
                        #method = "circle", # change plot to circle
                        type = "lower", 
                        lab_size= 2, 
                        tl.cex = 8,
                        lab = "true", #add correlation coefficient
                        outline.col = "black", 
                        tl.col = "black")+
  theme(axis.text.y=element_text(size=9),
        axis.text.x = element_text(size=9, angle=35))
print(corrplot_cb)

# Run dev.off() to create the file!
dev.off()



# continuous ordinal

corrmat_co <- polyserial(cont, ord)

corrmat_1co <- corrmat_co
is.na(corrmat_1co) <- abs(corrmat_1co) < 0.3

# open pdf file
pdf(file = "/Your/Path/correlation_corrplot_co_dev.pdf",   # The directory you want to save the file in
    width = 5, # The width of the plot in inches
    height = 5) # The height of the plot in inches

corrplot_co= ggcorrplot(corrmat_co, 
                        #method = "circle", # change plot to circle
                        type = "lower", 
                        lab_size= 2, 
                        tl.cex = 8,
                        lab = "true", #add correlation coefficient
                        outline.col = "black", 
                        tl.col = "black")+
  theme(axis.text.y=element_text(size=9),
        axis.text.x = element_text(size=9, angle=35))
print(corrplot_co)

# Run dev.off() to create the file!
dev.off()



# Combine corrmats

Corrmat <- rbind(corrmat_cc$r, corrmat_cb, corrmat_co)
cp2 <- t(cbind(corrmat_cb, corrmat_bb, corrmat_bo))
cp3 <- t(cbind(corrmat_co, t(corrmat_bo), corrmat_oo))

Corrmat_full <- cbind(Corrmat, cp2, cp3)

Corrmat_red <- Corrmat_full
is.na(Corrmat_red) <- abs(Corrmat_red) < 0.6

# open pdf file
#pdf(file = "./../graphics/25_04_15_correlation_corrplot_full_dev.pdf",   # The directory you want to save the file in
#    width = 30, # The width of the plot in inches
#    height = 20) # The height of the plot in inches
tiff("/Your/Path/correlation_corrplot_full_dev.tif", 
     units="px", width=2244, height=1496, res=356, compression = 'none')

corrplot_full= ggcorrplot(Corrmat_full, 
                        #method = "circle", # change plot to circle
                        type = "lower", 
                        lab_size= 2, 
                        tl.cex = 8,
                        lab = "true", #add correlation coefficient
                        outline.col = "black", 
                        tl.col = "black")+
  theme(axis.text.y=element_text(size=9),
        axis.text.x = element_text(size=9, angle=35))
print(corrplot_full)

# Run dev.off() to create the file!
dev.off()



# open pdf file
pdf(file = "/Your/Path/correlation_corrplot_red_dev_title.pdf",   # The directory you want to save the file in
    width = 6, # The width of the plot in inches
    height = 4) # The height of the plot in inches
#tiff("./../graphics/25_05_16_correlation_corrplot_red_dev_title.tif", 
#     units="px", width=2244, height=1496, res=356, compression = 'none')

corrplot_red= ggcorrplot(Corrmat_red, 
                         #title = "Strongly Correlated Features",
                          #method = "circle", # change plot to circle
                          type = "lower", 
                          lab_size= 2, 
                          tl.cex = 8,
                          lab = "true", #add correlation coefficient
                          outline.col = "black", 
                          tl.col = "black")+
  theme(axis.text.y=element_text(size=9),
        axis.text.x = element_text(size=9, angle=35))
print(corrplot_red)

# Run dev.off() to create the file!
dev.off()



write_csv(as.data.frame(Corrmat_full), "/Your/Path/Full_Corrmat.csv") 


# ==============================================================================
# removing features from validation and development dataset
# ==============================================================================

# remove continuous features based on multicollinearity (cutoff =0.75)

devds <- subset(devds, select= -c(surgDurat,
                                weight,
                                doseInduSteroInf,
                                twaMAPunder65Anes,
                                diab, 
                                medSevoExp #based on logical redundancy
                                
                                
))

valds <- subset(valds, select= -c(surgDurat,
                                  weight,
                                  doseInduSteroInf,
                                  twaMAPunder65Anes,
                                  diab,
                                  medSevoExp #based on logical redundancy
                                  
))


# ==============================================================================
# data set preprocessing for ML
# ==============================================================================



# ==========
# split data sets into continuous and discrete features

# generate new set of continuous names
cont_names <- c("inductDurat",
                "anestDurat",
                "age",
                "height",
                "bmi",
                #sex,
                #rcri,
                #urgency,
                #decrease,
                #asa,
                #nyha,
                #chf,
                #cad,
                #cvd,
                #pad,
                #diab_bin,
                #liverCirrh,
                #ekg,
                #ltmACE,
                #ltmSartan,
                #ltmBB,
                #ltmCCB,
                #ltmBig,
                #ltmIns,
                #ltmDiu,
                #ltmStat,
                #iuc, 
                #stomTube,
                "doseInduEtomidateBolus",
                "doseInduPropoBolus",
                "doseInduPropoPerfu",# perfusor zu syringe?
                "doseInduRemifPerfu",
                "doseInduSufenBolus", #raus?
                "doseInduSufenPerfu",
                "doseInduThiopBolus",
                "doseSurgGelafInf",
                "doseSurgSteroInf",
                "maxSevoExp",
                "eGFRPreSurg",
                #medInduKateBolus_bin,
                #medSurgAtroBolus_bin,
                #medSurgKateBolus_bin,
                #clinic_cat_ACH,
                #clinic_cat_GCH,
                #clinic_cat_HCH,
                #clinic_cat_HNO,
                #clinic_cat_NCH,
                #clinic_cat_PWC,
                #clinic_cat_TCH,
                #clinic_cat_UCH,
                #clinic_cat_URO,
                #clinic_cat_Other,
                "minMAPcumu1MinAnes",
                "minMAPcumu5MinAnes",
                "aucMAPunder65Anes",
                "meanAnes",              
                "stdAnes",                
                "entropyAnes",            
                "trendAnes",             
                "kurtosisAnes",           
                "skewnessAnes",
                "baselineMAP"
                #AKI_bin
)

cont_dev <- devds %>% select(one_of(cont_names))
cont_val <- valds %>% select(one_of(cont_names))

disc_dev <- devds %>% select(-one_of(cont_names))
disc_val <- valds %>% select(-one_of(cont_names))



# ==========
# Transform and standardize continuous features


# log2 transformation (without skewness and trend)

summary(cont_dev)
cont_dev[c(1:22,24, 26)] <- log(cont_dev[c(1:22,24, 26)]+0.01, 2) # log 2 transformation after adding 0.01 to every value
cont_val[c(1:22,24, 26)] <- log(cont_val[c(1:22,24, 26)]+0.01, 2) # log 2 transformation after adding 0.01 to every value
summary(cont_dev)



# standardization using the transformed train set means
means_dev <- colMeans(cont_dev, na.rm = FALSE) # creating vector of means for standardization of train and validation data
sds_dev <- apply(cont_dev, 2, sd, na.rm = FALSE) # creating vector of standard deviations for standardization of train and validation data

means_val <- colMeans(cont_val, na.rm = FALSE) # creating vector of means for standardization of train and validation data
sds_val <- apply(cont_val, 2, sd, na.rm = FALSE) # creating vector of standard deviations for standardization of train and validation data

cont_dev <- scale_with_params(cont_dev, means_dev, sds_dev) # standardization of continuous train data 
cont_val <- scale_with_params(cont_val, means_val, sds_val) # standardization of continuous train data 

summary(cont_dev)



# visualization of transformed data
# create QQplots and Histograms of all transformed variables
qq_and_hist(cont_dev, "/Your/Path/dev_continuous_QQ_plots_and_Histograms_preprocessed.pdf")
qq_and_hist(cont_val, "/Your/Path/val_continuous_QQ_plots_and_Histograms_preprocessed.pdf")



# saving vectors for transformations
saveRDS(means_dev, "/Your/Path/means_dev.rds")
saveRDS(sds_dev, "/Your/Path/sds_dev.rds")
saveRDS(means_val, "/Your/Path/means_val.rds")
saveRDS(sds_val, "/Your/Path/sds_val.rds")

rm(means_dev, means_val, sds_dev, sds_val)


# ==========
# check formatting of discrete data

# check data types
sapply(disc_dev, class) # check data types
sapply(disc_val, class) # check data types

#convert all columns to factor
disc_dev <- disc_dev %>%
  mutate(across(where(is.numeric), as.factor)) # convert all numeric columns to factor
disc_val <- disc_val %>%
  mutate(across(where(is.numeric), as.factor)) # convert all numeric columns to factor

# recheck data types
sapply(disc_dev, class) # check data types
sapply(disc_val, class) # check data types

# remove DECREASE III Score based on medical utility
disc_val <- disc_val %>% select(- decrease)
disc_dev <- disc_dev %>% select(- decrease)

summary(disc_dev)
summary(disc_val)


# ==============================================================================
# rejoin continuous and discrete data to development and validation dataset
# ==============================================================================

devds <- cbind(cont_dev, disc_dev) #9260
valds <- cbind(cont_val, disc_val) #4630

# save development and validation data sets
saveRDS(devds, "/Your/Path/devds_original_prep.rds")
saveRDS(valds, "/Your/Path/valds_prep.rds") 



# ==============================================================================
# Balancing Training data set using random undersampling
# ==============================================================================

# Identify indices of minority class samples
minority_indices <- which(devds$AKI == "1")

# Sample randomly from the majority class to match the number of minority class samples
majority_indices <- which(devds$AKI == "0")

undersampled_majority_indices <- sample(majority_indices, length(minority_indices))

# Combine minority and undersampled majority indices
undersampled_indices <- c(minority_indices, undersampled_majority_indices)

# Create the undersampled data frame
devds_balanced <- devds[undersampled_indices,]
dim(devds_balanced)
# 2286   62


# save balanced development data set
saveRDS(devds_balanced, "/Your/Path/devds_balanced_prep.rds")



# ----
# end
