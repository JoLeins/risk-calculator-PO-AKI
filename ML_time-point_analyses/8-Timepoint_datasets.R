# ==============================================================================
# Preparation of Time Point Datasets for time point specific PO-AKI prediction
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

rm(list = ls())

# ----
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



# ----
# setup
# ----

load_packages( c("tidyverse", "dplyr", "caret", "pROC", "tidymodels", "ranger", "kernlab", "gglasso", "glmnet" )) #"data.table",
set.seed(3010)
setwd("/your/path/")




# ----
# main
# ----


####Vorbereiten des Datensatzes####

# Laden der CSV-Dateien mit "ID" als Zeilenbezeichnung
df_validation <- read.csv('/your/path/reduced_validation_cohort.csv', 
                          sep = ";", 
                          dec = ",",row.names = "X")

# Konvertieren aller Spalten in numerische Datentypen
df_validation[] <- lapply(df_validation, as.numeric)




full_data <- read.csv('/your/path/full_dataset.csv', 
                      sep = ";", 
                      dec = ",", 
                      row.names = "X")

# Konvertieren aller Spalten in numerische Datentypen
full_data[] <- lapply(full_data, as.numeric)



df_training_balanced <- readRDS('/your/path/devds_balanced_prep.rds')



sum(duplicated(rownames(df_training_balanced)))

# Anzahl der Zeilen mit fehlenden Zeilennamen
sum(is.na(rownames(df_training_balanced)))



# Behalte in df_training nur die Zeilen, die auch in df_training_balanced vorkommen
df_training_balanced <- full_data[rownames(full_data) %in% rownames(df_training_balanced), ]


# Function to count matching rows between two data frames
count_matching_rows <- function(df1, df2) {
  # Ensure both data frames have the same column order
  df1 <- df1[, colnames(df2), drop = FALSE]
  df2 <- df2[, colnames(df2), drop = FALSE]
  
  # Convert data frames to character rows to compare easily
  df1_as_char <- apply(df1, 1, paste, collapse = "|")
  df2_as_char <- apply(df2, 1, paste, collapse = "|")
  
  # Count how many rows in df1 appear in df2
  match_count <- sum(df1_as_char %in% df2_as_char)
  
  return(match_count)
}

# Call the function
matching_count <- count_matching_rows(df_training_balanced, df_validation)

# Print the result
print(paste("Number of matching rows:", matching_count))



# Function to count matching row names between two data frames
count_matching_rownames <- function(df1, df2) {
  # Extract row names from both data frames
  rownames_df1 <- rownames(df1)
  rownames_df2 <- rownames(df2)
  
  # Count matching row names
  match_count <- sum(rownames_df1 %in% rownames_df2)
  
  return(match_count)
}


# Call the function
#matching_rownames_count <- count_matching_rownames(df_training_balanced, devds_bal)

# Print the result
#print(paste("Number of matching row names:", matching_rownames_count))




# Variablen für die verschiedenen Zeitpunkte
variables_praeop <- c("AKI_bin",
                      "age", "weight", "height", "bmi", "sex", "rcri", "urgency",
                      "asa", "nyha", "chf", "cad", "cvd", "pad", "diab", "diab_bin", 
                      "liverCirrh", "ekg", "ltmACE", "ltmSartan", "ltmBB", "ltmCCB", 
                      "ltmBig", "ltmIns", "ltmDiu", "ltmStat", "iuc", "stomTube", "eGFRPreSurg", 
                      "baselineMAP", "clinic_cat_ACH", "clinic_cat_GCH", "clinic_cat_HCH", 
                      "clinic_cat_HNO", "clinic_cat_NCH", "clinic_cat_PWC", "clinic_cat_TCH", 
                      "clinic_cat_UCH", "clinic_cat_URO", "clinic_cat_Other"
)



variables_einleitung <- c(
  variables_praeop,
  "inductDurat", "doseInduEtomidateBolus", "doseInduPropoBolus", "doseInduPropoPerfu", "doseInduRemifPerfu", "doseInduSteroInf",
  "doseInduSufenBolus", "doseInduSufenPerfu", "doseInduThiopBolus", "maxSevoExp", "medSevoExp", "medInduKateBolus_bin",
  "minMAPcumu1MinIndu", "minMAPcumu5MinIndu", "aucMAPunder65Indu", "meanIndu", "stdIndu" 
) #- "twaMAPdecrBL10Indu", "twaMAPunder65Indu"


variables_einl_grundl <- c(
  variables_praeop,
  "inductDurat", "doseInduEtomidateBolus", "doseInduPropoBolus", "doseInduPropoPerfu", "doseInduRemifPerfu", "doseInduSteroInf",
  "doseInduSufenBolus", "doseInduSufenPerfu", "doseInduThiopBolus", "maxSevoExp", "medSevoExp", "medInduKateBolus_bin"
) #- "twaMAPdecrBL10Indu", "twaMAPunder65Indu","minMAPcumu1MinIndu", "minMAPcumu5MinIndu", "aucMAPunder65Indu", "meanIndu", "stdIndu" 


variables_einlplus30 <- c(
  variables_einl_grundl,
  "minMAPcumu1MinInduplus30", "minMAPcumu5MinInduplus30", "aucMAPunder65Induplus30", "meanInduplus30", "stdInduplus30"
) #-"twaMAPdecrBL10Induplus30", "twaMAPunder65Induplus30"

variables_einlplus60 <- c(
  variables_einl_grundl,
  "minMAPcumu1MinInduplus60", "minMAPcumu5MinInduplus60", "aucMAPunder65Induplus60", "meanInduplus60", "stdInduplus60"
)#-"twaMAPdecrBL10Induplus60", "twaMAPunder65Induplus60"

variables_einlplus90 <- c(
  variables_einl_grundl,
  "minMAPcumu1MinInduplus90", "minMAPcumu5MinInduplus90", "aucMAPunder65Induplus90", "meanInduplus90", "stdInduplus90"
)#-, "twaMAPdecrBL10Induplus90", "twaMAPunder65Induplus90"

variables_einlplus120 <- c(
  variables_einl_grundl,
  "minMAPcumu1MinInduplus120", "minMAPcumu5MinInduplus120", "aucMAPunder65Induplus120", "meanInduplus120", "stdInduplus120"
)#-"twaMAPdecrBL10Induplus120", "twaMAPunder65Induplus120"

variables_einlplus150 <- c(
  variables_einl_grundl,
  "minMAPcumu1MinInduplus150", "minMAPcumu5MinInduplus150", "aucMAPunder65Induplus150", "meanInduplus150", "stdInduplus150"
)#-"twaMAPdecrBL10Induplus150", "twaMAPunder65Induplus150"

variables_postop <- c(
  variables_einl_grundl,
  "surgDurat", "anestDurat", "doseSurgGelafInf", "doseSurgSteroInf", "medSurgAtroBolus_bin", 
  "medSurgKateBolus_bin","minMAPcumu1MinAnes", "minMAPcumu5MinAnes", "aucMAPunder65Anes", "meanAnes", "stdAnes", "entropyAnes", "trendAnes", 
  "kurtosisAnes", "skewnessAnes"
)#-"mean_Surg", "std_Surg", "twaMAPdecrBL10Anes","aucMAPunder65Surg", "twaMAPunder65Anes", "minMAPcumu1MinSurg","minMAPcumu5MinSurg" ,"twaMAPunder65Surg","twaMAPdecrBL10Surg"






# Anzahl der Variablen für jede Gruppe berechnen
num_variables <- list(
  variables_praeop = length(unique(variables_praeop)),
  variables_einleitung = length(unique(variables_einleitung)),
  variables_einlplus30 = length(unique(variables_einlplus30)),
  variables_einlplus60 = length(unique(variables_einlplus60)),
  variables_einlplus90 = length(unique(variables_einlplus90)),
  variables_einlplus120 = length(unique(variables_einlplus120)),
  variables_einlplus150 = length(unique(variables_einlplus150)),
  variables_postop = length(unique(variables_postop))
)
print(num_variables)



# Zu entfernende Variablen und Teilstrings
variables_to_remove <- c("surgDurat", "decrease", "weight", "doseInduSteroInf", "twaMAPunder65", "diab", "medSevoExp")

# Funktion, um Variablen basierend auf Teilstrings zu entfernen
remove_variables <- function(variable_list, to_remove) {
  variable_list[!sapply(variable_list, function(var) which(var %in% to_remove))]#any(grepl(paste(to_remove, collapse = "|"), var)))]
}

# Entfernen der Variablen aus allen Listen
variables_praeop <- setdiff(variables_praeop, variables_to_remove)
variables_einleitung <- setdiff(variables_einleitung, variables_to_remove)
variables_einlplus30 <- setdiff(variables_einlplus30, variables_to_remove)
variables_einlplus60 <- setdiff(variables_einlplus60, variables_to_remove)
variables_einlplus90 <- setdiff(variables_einlplus90, variables_to_remove)
variables_einlplus120 <- setdiff(variables_einlplus120, variables_to_remove)
variables_einlplus150 <- setdiff(variables_einlplus150, variables_to_remove)
variables_postop <- setdiff(variables_postop, variables_to_remove)

# Überprüfung: Anzahl der Variablen nach der Entfernung
cat("Anzahl der Variablen nach Entfernung:\n")
cat("variables_praeop:", length(variables_praeop), "\n")
cat("variables_einleitung:", length(variables_einleitung), "\n")
cat("variables_einlplus30:", length(variables_einlplus30), "\n")
cat("variables_einlplus60:", length(variables_einlplus60), "\n")
cat("variables_einlplus90:", length(variables_einlplus90), "\n")
cat("variables_einlplus120:", length(variables_einlplus120), "\n")
cat("variables_einlplus150:", length(variables_einlplus150), "\n")
cat("variables_postop:", length(variables_postop), "\n")


# Funktion zur Erstellung neuer Datensätze basierend auf einer Variablenliste
create_dataset <- function(df, variable_list) {
  # Auswahl der Spalten, die in der Variablenliste enthalten sind
  df[, intersect(variable_list, colnames(df)), drop = FALSE]
}

# Erstellung neuer Datensätze für jeden Zeitpunkt
df_validation_praeop <- create_dataset(df_validation, variables_praeop)
df_training_balanced_praeop <- create_dataset(df_training_balanced, variables_praeop)

df_validation_einleitung <- create_dataset(df_validation, variables_einleitung)
df_training_balanced_einleitung <- create_dataset(df_training_balanced, variables_einleitung)

df_validation_einlplus30 <- create_dataset(df_validation, variables_einlplus30)
df_training_balanced_einlplus30 <- create_dataset(df_training_balanced, variables_einlplus30)

df_validation_einlplus60 <- create_dataset(df_validation, variables_einlplus60)
df_training_balanced_einlplus60 <- create_dataset(df_training_balanced, variables_einlplus60)

df_validation_einlplus90 <- create_dataset(df_validation, variables_einlplus90)
df_training_balanced_einlplus90 <- create_dataset(df_training_balanced, variables_einlplus90)

df_validation_einlplus120 <- create_dataset(df_validation, variables_einlplus120)
df_training_balanced_einlplus120 <- create_dataset(df_training_balanced, variables_einlplus120)

df_validation_einlplus150 <- create_dataset(df_validation, variables_einlplus150)
df_training_balanced_einlplus150 <- create_dataset(df_training_balanced, variables_einlplus150)

df_validation_postop <- create_dataset(df_validation, variables_postop)
df_training_balanced_postop <- create_dataset(df_training_balanced, variables_postop)

# Liste der Zeitpunkte und zugehörige Datensätze
time_points <- c("praeop", "einleitung", "einlplus30", "einlplus60", 
                 "einlplus90", "einlplus120", "einlplus150", "postop")




















################################################################################
################################################################################


####Vorbereiten des Datensatzes####

# Laden der CSV-Dateien mit "ID" als Zeilenbezeichnung
df_validation_input <- read.csv('/your/path/reduced_validation_cohort.csv', 
                                sep = ";", 
                                dec = ",",
                                row.names = "X")

# Konvertieren aller Spalten in numerische Datentypen
df_validation_input[] <- lapply(df_validation_input, as.numeric)




full_data <- read.csv('/your/path/full_dataset.csv', 
                      sep = ";", 
                      dec = ",", 
                      row.names = "X")

map_features <- read.csv('/your/path/MAP_Features.csv', 
                         row.names = "X")

colnames(map_features)


# Liste der Begriffe, die enthalten sein sollen
variable_keywords <- c("Kurtosis", "Trend", "Skewness", "Mean", "STD", 
                       "Entropy", "AUC.of.MAP.under.65", 
                       "Lowest.MAP.for.cumulative.5.minutes", 
                       "Lowest.MAP.for.cumulative.1.minutes")

# Auswahl der Variablen, die eines der Keywords beinhalten
selected_variables <- colnames(map_features)[sapply(colnames(map_features), function(colname) {
  any(sapply(variable_keywords, function(keyword) grepl(keyword, colname, ignore.case = TRUE)))&&
    !grepl("decrease", colname, ignore.case = TRUE)
})]

# Überprüfen, ob ausgewählte Variablen existieren
if (length(selected_variables) == 0) {
  stop("Keine Spalten in 'map_features' enthalten die angegebenen Keywords.")
}

# Filtere die gewünschten Variablen aus map_features
selected_map_features <- map_features[, selected_variables, drop = FALSE]

colnames(selected_map_features)

# Alte Variablennamen (aus deiner Liste)
old_names <- c(
  "Lowest.MAP.for.cumulative.1.minutes..Narkose",
  "Lowest.MAP.for.cumulative.5.minutes..Narkose",
  "Lowest.MAP.for.cumulative.1.minutes..einl",
  "Lowest.MAP.for.cumulative.5.minutes..einl",
  "Lowest.MAP.for.cumulative.1.minutes..einlplus30",
  "Lowest.MAP.for.cumulative.5.minutes..einlplus30",
  "Lowest.MAP.for.cumulative.1.minutes..einlplus60",
  "Lowest.MAP.for.cumulative.5.minutes..einlplus60",
  "Lowest.MAP.for.cumulative.1.minutes..einlplus90",
  "Lowest.MAP.for.cumulative.5.minutes..einlplus90",
  "Lowest.MAP.for.cumulative.1.minutes..einlplus120",
  "Lowest.MAP.for.cumulative.5.minutes..einlplus120",
  "Lowest.MAP.for.cumulative.1.minutes..einlplus150",
  "Lowest.MAP.for.cumulative.5.minutes..einlplus150",
  "Lowest.MAP.for.cumulative.1.minutes..OP",
  "Lowest.MAP.for.cumulative.5.minutes..OP",
  "AUC.of.MAP.under.65..Narkose",
  "AUC.of.MAP.under.65..einl",
  "AUC.of.MAP.under.65..einlplus30",
  "AUC.of.MAP.under.65..einlplus60",
  "AUC.of.MAP.under.65..einlplus90",
  "AUC.of.MAP.under.65..einlplus120",
  "AUC.of.MAP.under.65..einlplus150",
  "AUC.of.MAP.under.65..OP",
  "Mean_Narkose",
  "STD_Narkose",
  "Entropy_Narkose",
  "Trend_Narkose",
  "Kurtosis_Narkose",
  "Skewness_Narkose",
  "Mean_einl",
  "STD_einl",
  "Entropy_einl",
  "Trend_einl",
  "Kurtosis_einl",
  "Skewness_einl",
  "Mean_einlplus30",
  "STD_einlplus30",
  "Entropy_einlplus30",
  "Trend_einlplus30",
  "Kurtosis_einlplus30",
  "Skewness_einlplus30",
  "Mean_einlplus60",
  "STD_einlplus60",
  "Entropy_einlplus60",
  "Trend_einlplus60",
  "Kurtosis_einlplus60",
  "Skewness_einlplus60",
  "Mean_einlplus90",
  "STD_einlplus90",
  "Entropy_einlplus90",
  "Trend_einlplus90",
  "Kurtosis_einlplus90",
  "Skewness_einlplus90",
  "Mean_einlplus120",
  "STD_einlplus120",
  "Entropy_einlplus120",
  "Trend_einlplus120",
  "Kurtosis_einlplus120",
  "Skewness_einlplus120",
  "Mean_einlplus150",
  "STD_einlplus150",
  "Entropy_einlplus150",
  "Trend_einlplus150",
  "Kurtosis_einlplus150",
  "Skewness_einlplus150",
  "Mean_OP",
  "STD_OP",
  "Entropy_OP",
  "Trend_OP",
  "Kurtosis_OP",
  "Skewness_OP"
)

# Neue Variablennamen (aus deiner Liste)
new_names <- c(
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
  "meanAnes",
  "stdAnes",
  "entropyAnes",
  "trendAnes",
  "kurtosisAnes",
  "skewnessAnes",
  "meanIndu",
  "stdIndu",
  "entropyIndu",
  "trendIndu",
  "kurtosisIndu",
  "skewnessIndu",
  "meanInduplus30",
  "stdInduplus30",
  "entropyInduplus30",
  "trendInduplus30",
  "kurtosisInduplus30",
  "skewnessInduplus30",
  "meanInduplus60",
  "stdInduplus60",
  "entropyInduplus60",
  "trendInduplus60",
  "kurtosisInduplus60",
  "skewnessInduplus60",
  "meanInduplus90",
  "stdInduplus90",
  "entropyInduplus90",
  "trendInduplus90",
  "kurtosisInduplus90",
  "skewnessInduplus90",
  "meanInduplus120",
  "stdInduplus120",
  "entropyInduplus120",
  "trendInduplus120",
  "kurtosisInduplus120",
  "skewnessInduplus120",
  "meanInduplus150",
  "stdInduplus150",
  "entropyInduplus150",
  "trendInduplus150",
  "kurtosisInduplus150",
  "skewnessInduplus150",
  "meanSurg",
  "stdSurg",
  "entropySurg",
  "trendSurg",
  "kurtosisSurg",
  "skewnessSurg"
)

# Überprüfen, ob die Längen gleich sind
if (length(old_names) != length(new_names)) {
  stop("Die Listen 'old_names' und 'new_names' haben unterschiedliche Längen!")
}

# Erstellen einer Zuordnung (Mapping)
name_mapping <- setNames(new_names, old_names)

# Umbenennen der Variablen in selected_map_features
colnames(selected_map_features) <- ifelse(colnames(selected_map_features) %in% names(name_mapping), 
                                          name_mapping[colnames(selected_map_features)], 
                                          colnames(selected_map_features))

# Überprüfen der neuen Variablennamen
print(colnames(selected_map_features))


# Anzahl der NA-Werte pro Variable berechnen
na_counts <- colSums(is.na(selected_map_features))

# Filtere nur Variablen mit NA-Werten
na_counts_filtered <- na_counts[na_counts > 0]

# Ausgabe der Variablen mit NA-Werten
print("Variablen mit NA-Werten und deren Anzahl:")
print(na_counts_filtered)



# Konvertieren aller Spalten in numerische Datentypen
full_data[] <- lapply(full_data, as.numeric)

#CAVE: Habe manuell überprüft dass die Datensätze in der gleichen Reihenfolge vorliegen und die Zeilen übereinstimmig sind
#Werde das später noch "100% korrekt" nacholen, wurden verschiedene Arten von IDs bzw. Reihenindizes verwendet von JL und HM
full_data_complete <- cbind(full_data, selected_map_features)



df_training_balanced_input <- readRDS('/your/path/devds_balanced_prep.rds')



sum(duplicated(rownames(df_training_balanced_input)))

# Anzahl der Zeilen mit fehlenden Zeilennamen
sum(is.na(rownames(df_training_balanced_input)))



# Behalte in df_training nur die Zeilen, die auch in df_training_balanced vorkommen
df_training_balanced <- full_data_complete[rownames(full_data_complete) %in% rownames(df_training_balanced_input), ]
df_validation <- full_data_complete[rownames(full_data_complete) %in% rownames(df_validation_input), ]


# Anzahl der Zeilen vor dem Entfernen von NA
training_rows_before <- nrow(df_training_balanced)
validation_rows_before <- nrow(df_validation)

# Entfernen der Zeilen mit NA
df_training_balanced <- na.omit(df_training_balanced)
df_validation <- na.omit(df_validation)

# Anzahl der Zeilen nach dem Entfernen von NA
training_rows_after <- nrow(df_training_balanced)
validation_rows_after <- nrow(df_validation)

# Berechnung der entfernten Zeilen
training_removed <- training_rows_before - training_rows_after
validation_removed <- validation_rows_before - validation_rows_after

# Ausgabe der Anzahl der entfernten Zeilen
cat("Anzahl der entfernten Zeilen aus df_training_balanced:", training_removed, "\n")
cat("Anzahl der entfernten Zeilen aus df_validation:", validation_removed, "\n")


# Function to count matching rows between two data frames
count_matching_rows <- function(df1, df2) {
  # Ensure both data frames have the same column order
  df1 <- df1[, colnames(df2), drop = FALSE]
  df2 <- df2[, colnames(df2), drop = FALSE]
  
  # Convert data frames to character rows to compare easily
  df1_as_char <- apply(df1, 1, paste, collapse = "|")
  df2_as_char <- apply(df2, 1, paste, collapse = "|")
  
  # Count how many rows in df1 appear in df2
  match_count <- sum(df1_as_char %in% df2_as_char)
  
  return(match_count)
}

# Call the function
matching_count <- count_matching_rows(df_training_balanced, df_validation)

# Print the result
print(paste("Number of matching rows:", matching_count))



# Function to count matching row names between two data frames
count_matching_rownames <- function(df1, df2) {
  # Extract row names from both data frames
  rownames_df1 <- rownames(df1)
  rownames_df2 <- rownames(df2)
  
  # Count matching row names
  match_count <- sum(rownames_df1 %in% rownames_df2)
  
  return(match_count)
}


# Call the function
matching_rownames_count <- count_matching_rownames(df_training_balanced, df_training_balanced_input)

# Print the result
print(paste("Number of matching row names:", matching_rownames_count))

colnames(df_training_balanced_input)



# ==============================================================================

# Transformation and standardization of timepoint specific MAP variables

# ==============================================================================


# remove duplicate features (minMAPs and AUCs, first instance removed respectively)
df_training_balanced <- df_training_balanced[, -c(60:111)]
df_validation <- df_validation[, -c(60:111)]
# isolate features

cont_dev <- df_training_balanced %>% select(- c(ID, sex, rcri, urgency, asa, nyha, chf, cad,
                                 cvd, pad, diab, decrease, diab_bin, liverCirrh, ekg, ltmACE,
                                 ltmSartan, ltmBB, ltmCCB, ltmBig, ltmIns, ltmDiu,
                                 ltmStat, iuc, stomTube, AKI, AKI_bin, clinic_cat_ACH,
                                 clinic_cat_GCH, clinic_cat_HCH, clinic_cat_HNO,
                                 clinic_cat_NCH, clinic_cat_PWC, clinic_cat_TCH,
                                 clinic_cat_UCH, clinic_cat_URO, clinic_cat_Other))

cont_val <- df_validation %>% select(- c(ID, sex, rcri, urgency, asa, nyha, chf, cad,
                                                cvd, pad, diab, decrease, diab_bin, liverCirrh, ekg, ltmACE,
                                                ltmSartan, ltmBB, ltmCCB, ltmBig, ltmIns, ltmDiu,
                                                ltmStat, iuc, stomTube, AKI, AKI_bin, clinic_cat_ACH,
                                                clinic_cat_GCH, clinic_cat_HCH, clinic_cat_HNO,
                                                clinic_cat_NCH, clinic_cat_PWC, clinic_cat_TCH,
                                                clinic_cat_UCH, clinic_cat_URO, clinic_cat_Other))

disc_dev <- df_training_balanced %>% select(  c( sex, rcri, urgency, asa, nyha, chf, cad,
                                                cvd, pad, diab, diab_bin, liverCirrh, ekg, ltmACE,
                                                ltmSartan, ltmBB, ltmCCB, ltmBig, ltmIns, ltmDiu,
                                                ltmStat, iuc, stomTube, AKI, AKI_bin, clinic_cat_ACH,
                                                clinic_cat_GCH, clinic_cat_HCH, clinic_cat_HNO,
                                                clinic_cat_NCH, clinic_cat_PWC, clinic_cat_TCH,
                                                clinic_cat_UCH, clinic_cat_URO, clinic_cat_Other))

disc_val <- df_validation %>% select(  c( sex, rcri, urgency, asa, nyha, chf, cad,
                                                  cvd, pad, diab, diab_bin, liverCirrh, ekg, ltmACE,
                                                  ltmSartan, ltmBB, ltmCCB, ltmBig, ltmIns, ltmDiu,
                                                  ltmStat, iuc, stomTube, AKI, AKI_bin, clinic_cat_ACH,
                                                  clinic_cat_GCH, clinic_cat_HCH, clinic_cat_HNO,
                                                  clinic_cat_NCH, clinic_cat_PWC, clinic_cat_TCH,
                                                  clinic_cat_UCH, clinic_cat_URO, clinic_cat_Other))


# log2 transformation (without skewness and trend)
summary(cont_dev)
summary(cont_val)
cont_dev[c(1:59,61,63:64,67,
           69:71,73,75:77,79,
           81:83,85,87, 89,91,
           93:95,97,99:101,103)] <- log(cont_dev[c(1:59,61,63:64,67,
                                                   69:71,73,75:77,79,
                                                   81:83,85,87, 89,91,
                                                   93:95,97,99:101,103)]+0.01, 2) # log 2 transformation after adding 0.01 to every value
cont_val[c(1:59,61,63:64,67,
           69:71,73,75:77,79,
           81:83,85,87, 89,91,
           93:95,97,99:101,103)] <- log(cont_val[c(1:59,61,63:64,67,
                                                   69:71,73,75:77,79,
                                                   81:83,85,87, 89,91,
                                                   93:95,97,99:101,103)]+0.01, 2) # log 2 transformation after adding 0.01 to every value
summary(cont_dev)
summary(cont_val)


# Create a function to scale devds using stored mean and standard deviation

scale_with_params <- function(devds, mean_vals, sd_vals) {
  scaled_devds <- scale(devds, center = mean_vals, scale = sd_vals)
  return(scaled_devds)
}

# standardization using the transformed train set means
means_dev <- colMeans(cont_dev[, c(1:104)], na.rm = FALSE) # creating vector of means for standardization of train and validation data
sds_dev <- apply(cont_dev[, c(1:104)], 2, sd, na.rm = FALSE) # creating vector of standard deviations for standardization of train and validation data

means_val <- colMeans(cont_val[, c(1:104)], na.rm = FALSE) # creating vector of means for standardization of train and validation data
sds_val <- apply(cont_val[, c(1:104)], 2, sd, na.rm = FALSE) # creating vector of standard deviations for standardization of train and validation data

cont_dev[, c(1:104)] <- scale_with_params(cont_dev[, c(1:104)], means_dev, sds_dev) # standardization of continuous train data 
cont_val[, c(1:104)] <- scale_with_params(cont_val[, c(1:104)], means_val, sds_val) # standardization of continuous train data 

summary(cont_dev)
summary(cont_val)


# visualization of transformed data
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

# create QQplots and Histograms of all transformed variables
qq_and_hist(cont_dev, "/your/path/dev_cont_TP_QQ_plots_and_Hist.pdf")
qq_and_hist(cont_val, "/your/path/val_cont_TP_QQ_plots_and_Hist.pdf")



# saving vectors for transformations
saveRDS(means_dev, "/your/path/means_dev_TP.rds")
saveRDS(sds_dev, "/your/path/sds_dev_TP.rds")
saveRDS(means_val, "/your/path/means_val_TP.rds")
saveRDS(sds_val, "/your/path/sds_val_TP.rds")

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

summary(disc_dev)
summary(disc_val)

#rejoin disc and cont variables 
df_training_balanced <- cbind(disc_dev, cont_dev)
df_validation <- cbind(disc_val, cont_val)

# Verwendung der Liste der Variablen aus df_training_balanced_input
variables_postop <- c(
  "inductDurat", "anestDurat", "age", "height", "bmi", "doseInduEtomidateBolus",
  "doseInduPropoBolus", "doseInduPropoPerfu", "doseInduRemifPerfu", "doseInduSufenBolus",
  "doseInduSufenPerfu", "doseInduThiopBolus", "doseSurgGelafInf", "doseSurgSteroInf",
  "maxSevoExp", "eGFRPreSurg", "minMAPcumu1MinAnes", "minMAPcumu5MinAnes", 
  "aucMAPunder65Anes", "meanAnes", "stdAnes", "entropyAnes", "trendAnes", 
  "kurtosisAnes", "skewnessAnes", "baselineMAP", "sex", "rcri", "urgency", 
  "asa", "nyha", "chf", "cad", "cvd", "pad", "diab_bin", 
  "liverCirrh", "ekg", "ltmACE", "ltmSartan", "ltmBB", "ltmCCB", "ltmBig", 
  "ltmIns", "ltmDiu", "ltmStat", "iuc", "stomTube", "medInduKateBolus_bin", 
  "medSurgAtroBolus_bin", "medSurgKateBolus_bin", "clinic_cat_ACH", "clinic_cat_GCH", 
  "clinic_cat_HCH", "clinic_cat_HNO", "clinic_cat_NCH", "clinic_cat_PWC", 
  "clinic_cat_TCH", "clinic_cat_UCH", "clinic_cat_URO", "clinic_cat_Other", 
  "AKI_bin"
)

#MAP-Variablen auf 150 angepasst + "anestDurat", "doseSurgGelafInf", "doseSurgSteroInf","medSurgAtroBolus_bin", "medSurgKateBolus_bin" entfernt
variables_einlplus150 <- c(
  "inductDurat", "age", "height", "bmi", "doseInduEtomidateBolus",
  "doseInduPropoBolus", "doseInduPropoPerfu", "doseInduRemifPerfu", "doseInduSufenBolus",
  "doseInduSufenPerfu", "doseInduThiopBolus", 
  "maxSevoExp", "eGFRPreSurg", "minMAPcumu1MinInduplus150", "minMAPcumu5MinInduplus150", 
  "aucMAPunder65Induplus150", "meanInduplus150", "stdInduplus150", "entropyInduplus150", "trendInduplus150", 
  "kurtosisInduplus150", "skewnessInduplus150", "baselineMAP", "sex", "rcri", "urgency", 
  "asa", "nyha", "chf", "cad", "cvd", "pad", "diab_bin", 
  "liverCirrh", "ekg", "ltmACE", "ltmSartan", "ltmBB", "ltmCCB", "ltmBig", 
  "ltmIns", "ltmDiu", "ltmStat", "iuc", "stomTube", "medInduKateBolus_bin", 
  "clinic_cat_ACH", "clinic_cat_GCH", 
  "clinic_cat_HCH", "clinic_cat_HNO", "clinic_cat_NCH", "clinic_cat_PWC", 
  "clinic_cat_TCH", "clinic_cat_UCH", "clinic_cat_URO", "clinic_cat_Other", 
  "AKI_bin"
)

#MAP-Variablen auf 120 angepasst
variables_einlplus120 <- c(
  "inductDurat", "age", "height", "bmi", "doseInduEtomidateBolus",
  "doseInduPropoBolus", "doseInduPropoPerfu", "doseInduRemifPerfu", "doseInduSufenBolus",
  "doseInduSufenPerfu", "doseInduThiopBolus", 
  "maxSevoExp", "eGFRPreSurg", "minMAPcumu1MinInduplus120", "minMAPcumu5MinInduplus120", 
  "aucMAPunder65Induplus120", "meanInduplus120", "stdInduplus120", "entropyInduplus120", "trendInduplus120", 
  "kurtosisInduplus120", "skewnessInduplus120", "baselineMAP", "sex", "rcri", "urgency", 
  "asa", "nyha", "chf", "cad", "cvd", "pad", "diab_bin", 
  "liverCirrh", "ekg", "ltmACE", "ltmSartan", "ltmBB", "ltmCCB", "ltmBig", 
  "ltmIns", "ltmDiu", "ltmStat", "iuc", "stomTube", "medInduKateBolus_bin", 
  "clinic_cat_ACH", "clinic_cat_GCH", 
  "clinic_cat_HCH", "clinic_cat_HNO", "clinic_cat_NCH", "clinic_cat_PWC", 
  "clinic_cat_TCH", "clinic_cat_UCH", "clinic_cat_URO", "clinic_cat_Other", 
  "AKI_bin"
)
#MAP-Variablen auf 90 angepasst
variables_einlplus90 <- c(
  "inductDurat", "age", "height", "bmi", "doseInduEtomidateBolus",
  "doseInduPropoBolus", "doseInduPropoPerfu", "doseInduRemifPerfu", "doseInduSufenBolus",
  "doseInduSufenPerfu", "doseInduThiopBolus", 
  "maxSevoExp", "eGFRPreSurg", "minMAPcumu1MinInduplus90", "minMAPcumu5MinInduplus90", 
  "aucMAPunder65Induplus90", "meanInduplus90", "stdInduplus90", "entropyInduplus90", "trendInduplus90", 
  "kurtosisInduplus90", "skewnessInduplus90", "baselineMAP", "sex", "rcri", "urgency", 
  "asa", "nyha", "chf", "cad", "cvd", "pad", "diab_bin", 
  "liverCirrh", "ekg", "ltmACE", "ltmSartan", "ltmBB", "ltmCCB", "ltmBig", 
  "ltmIns", "ltmDiu", "ltmStat", "iuc", "stomTube", "medInduKateBolus_bin", 
  "clinic_cat_ACH", "clinic_cat_GCH", 
  "clinic_cat_HCH", "clinic_cat_HNO", "clinic_cat_NCH", "clinic_cat_PWC", 
  "clinic_cat_TCH", "clinic_cat_UCH", "clinic_cat_URO", "clinic_cat_Other", 
  "AKI_bin"
)

#MAP-Variablen auf 60 angepasst
variables_einlplus60 <- c(
  "inductDurat", "age", "height", "bmi", "doseInduEtomidateBolus",
  "doseInduPropoBolus", "doseInduPropoPerfu", "doseInduRemifPerfu", "doseInduSufenBolus",
  "doseInduSufenPerfu", "doseInduThiopBolus", 
  "maxSevoExp", "eGFRPreSurg", "minMAPcumu1MinInduplus60", "minMAPcumu5MinInduplus60", 
  "aucMAPunder65Induplus60", "meanInduplus60", "stdInduplus60", "entropyInduplus60", "trendInduplus60", 
  "kurtosisInduplus60", "skewnessInduplus60", "baselineMAP", "sex", "rcri", "urgency", 
  "asa", "nyha", "chf", "cad", "cvd", "pad", "diab_bin", 
  "liverCirrh", "ekg", "ltmACE", "ltmSartan", "ltmBB", "ltmCCB", "ltmBig", 
  "ltmIns", "ltmDiu", "ltmStat", "iuc", "stomTube", "medInduKateBolus_bin", 
  "clinic_cat_ACH", "clinic_cat_GCH", 
  "clinic_cat_HCH", "clinic_cat_HNO", "clinic_cat_NCH", "clinic_cat_PWC", 
  "clinic_cat_TCH", "clinic_cat_UCH", "clinic_cat_URO", "clinic_cat_Other", 
  "AKI_bin"
)

#MAP-Variablen auf 30 angepasst
variables_einlplus30 <- c(
  "inductDurat", "age", "height", "bmi", "doseInduEtomidateBolus",
  "doseInduPropoBolus", "doseInduPropoPerfu", "doseInduRemifPerfu", "doseInduSufenBolus",
  "doseInduSufenPerfu", "doseInduThiopBolus", 
  "maxSevoExp", "eGFRPreSurg", "minMAPcumu1MinInduplus30", "minMAPcumu5MinInduplus30", 
  "aucMAPunder65Induplus30", "meanInduplus30", "stdInduplus30", "entropyInduplus30", "trendInduplus30", 
  "kurtosisInduplus30", "skewnessInduplus30", "baselineMAP", "sex", "rcri", "urgency", 
  "asa", "nyha", "chf", "cad", "cvd", "pad", "diab_bin", 
  "liverCirrh", "ekg", "ltmACE", "ltmSartan", "ltmBB", "ltmCCB", "ltmBig", 
  "ltmIns", "ltmDiu", "ltmStat", "iuc", "stomTube", "medInduKateBolus_bin", 
  "clinic_cat_ACH", "clinic_cat_GCH", 
  "clinic_cat_HCH", "clinic_cat_HNO", "clinic_cat_NCH", "clinic_cat_PWC", 
  "clinic_cat_TCH", "clinic_cat_UCH", "clinic_cat_URO", "clinic_cat_Other", 
  "AKI_bin"
)


variables_einleitung <- c(
  "inductDurat", "age", "height", "bmi", "doseInduEtomidateBolus",
  "doseInduPropoBolus", "doseInduPropoPerfu", "doseInduRemifPerfu", "doseInduSufenBolus",
  "doseInduSufenPerfu", "doseInduThiopBolus", 
  "maxSevoExp", "eGFRPreSurg", "minMAPcumu1MinIndu", "minMAPcumu5MinIndu", 
  "aucMAPunder65Indu", "meanIndu", "stdIndu", "entropyIndu", "trendIndu", 
  "kurtosisIndu", "skewnessIndu", "baselineMAP", "sex", "rcri", "urgency", 
  "asa", "nyha", "chf", "cad", "cvd", "pad", "diab_bin", 
  "liverCirrh", "ekg", "ltmACE", "ltmSartan", "ltmBB", "ltmCCB", "ltmBig", 
  "ltmIns", "ltmDiu", "ltmStat", "iuc", "stomTube", "medInduKateBolus_bin", 
  "clinic_cat_ACH", "clinic_cat_GCH", 
  "clinic_cat_HCH", "clinic_cat_HNO", "clinic_cat_NCH", "clinic_cat_PWC", 
  "clinic_cat_TCH", "clinic_cat_UCH", "clinic_cat_URO", "clinic_cat_Other", 
  "AKI_bin"
)

# Entfernen von "inductDurat","doseInduEtomidateBolus",
#"doseInduPropoBolus", "doseInduPropoPerfu", "doseInduRemifPerfu", "doseInduSufenBolus",
#"doseInduSufenPerfu", "doseInduThiopBolus", "maxSevoExp", "minMAPcumu1MinIndu", "minMAPcumu5MinIndu", 
#  "aucMAPunder65Indu", "meanIndu", "stdIndu", "entropyIndu", "trendIndu", 
#  "kurtosisIndu", "skewnessIndu","medInduKateBolus_bin"

variables_praeop <- c(
  "age", "height", "bmi",  "eGFRPreSurg",  "baselineMAP", "sex", "rcri", "urgency", 
  "decrease", "asa", "nyha", "chf", "cad", "cvd", "pad", "diab_bin", 
  "liverCirrh", "ekg", "ltmACE", "ltmSartan", "ltmBB", "ltmCCB", "ltmBig", 
  "ltmIns", "ltmDiu", "ltmStat", "iuc", "stomTube", "clinic_cat_ACH", "clinic_cat_GCH", 
  "clinic_cat_HCH", "clinic_cat_HNO", "clinic_cat_NCH", "clinic_cat_PWC", 
  "clinic_cat_TCH", "clinic_cat_UCH", "clinic_cat_URO", "clinic_cat_Other", 
  "AKI_bin"
)


# Anzahl der Variablen für jede Gruppe berechnen
num_variables <- list(
  variables_praeop = length(unique(variables_praeop)),
  variables_einleitung = length(unique(variables_einleitung)),
  variables_einlplus30 = length(unique(variables_einlplus30)),
  variables_einlplus60 = length(unique(variables_einlplus60)),
  variables_einlplus90 = length(unique(variables_einlplus90)),
  variables_einlplus120 = length(unique(variables_einlplus120)),
  variables_einlplus150 = length(unique(variables_einlplus150)),
  variables_postop = length(unique(variables_postop))
)

# Ergebnis anzeigen
print(num_variables)


# Überprüfung: Anzahl der Variablen nach der Entfernung
cat("Anzahl der Variablen nach Entfernung:\n")
cat("variables_praeop:", length(variables_praeop), "\n")
cat("variables_einleitung:", length(variables_einleitung), "\n")
cat("variables_einlplus30:", length(variables_einlplus30), "\n")
cat("variables_einlplus60:", length(variables_einlplus60), "\n")
cat("variables_einlplus90:", length(variables_einlplus90), "\n")
cat("variables_einlplus120:", length(variables_einlplus120), "\n")
cat("variables_einlplus150:", length(variables_einlplus150), "\n")
cat("variables_postop:", length(variables_postop), "\n")


# Funktion zur Erstellung neuer Datensätze basierend auf einer Variablenliste
create_dataset <- function(df, variable_list) {
  # Auswahl der Spalten, die in der Variablenliste enthalten sind
  df[, intersect(variable_list, colnames(df)), drop = FALSE]
}

# Erstellung neuer Datensätze für jeden Zeitpunkt
df_validation_praeop <- create_dataset(df_validation, variables_praeop)
df_training_balanced_praeop <- create_dataset(df_training_balanced, variables_praeop)

df_validation_einleitung <- create_dataset(df_validation, variables_einleitung)
df_training_balanced_einleitung <- create_dataset(df_training_balanced, variables_einleitung)

df_validation_einlplus30 <- create_dataset(df_validation, variables_einlplus30)
df_training_balanced_einlplus30 <- create_dataset(df_training_balanced, variables_einlplus30)

df_validation_einlplus60 <- create_dataset(df_validation, variables_einlplus60)
df_training_balanced_einlplus60 <- create_dataset(df_training_balanced, variables_einlplus60)

df_validation_einlplus90 <- create_dataset(df_validation, variables_einlplus90)
df_training_balanced_einlplus90 <- create_dataset(df_training_balanced, variables_einlplus90)

df_validation_einlplus120 <- create_dataset(df_validation, variables_einlplus120)
df_training_balanced_einlplus120 <- create_dataset(df_training_balanced, variables_einlplus120)

df_validation_einlplus150 <- create_dataset(df_validation, variables_einlplus150)
df_training_balanced_einlplus150 <- create_dataset(df_training_balanced, variables_einlplus150)

df_validation_postop <- create_dataset(df_validation, variables_postop)
df_training_balanced_postop <- create_dataset(df_training_balanced, variables_postop)


# Liste der Zeitpunkte und zugehörige Datensätze
time_points <- list("praeop" = list(train = df_training_balanced_praeop, val = df_validation_praeop), 
                    "einleitung" = list(train = df_training_balanced_einleitung, val = df_validation_einleitung), 
                    "einlplus30" = list(train = df_training_balanced_einlplus30, val = df_validation_einlplus30), 
                    "einlplus60" = list(train = df_training_balanced_einlplus60, val = df_validation_einlplus60), 
                    "einlplus90" = list(train = df_training_balanced_einlplus90, val = df_validation_einlplus90), 
                    "einlplus120" = list(train = df_training_balanced_einlplus120, val = df_validation_einlplus120), 
                    "einlplus150" = list(train = df_training_balanced_einlplus150, val = df_validation_einlplus150), 
                    "postop" = list(train = df_training_balanced_postop, val = df_validation_postop))



saveRDS(time_points, "/your/path/Timepoint_datasets.rds")


# ----
# end



