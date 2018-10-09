################## Clear the Global Environment ######################################
rm(list = ls(all.names = T))

#################### Set the working directory #######################################
getwd()
setwd("C:\\Users\\chait\\Documents\\Insofe\\MiTh\\data")

################## Load all the required libraries for preprocessing #############################
library(DMwR)
library(caret)
library(tidyverse)

#################### Read the train data file #############################################

pure_data_file = read.csv(file = "dataset.csv", header = T) #na.strings = c(":", "?")

pure_data = pure_data_file[pure_data_file$istrain == 1, ]

###################### Descriptive stats ############################################
#### Understanding the dataset like column statistics, no.of missing values,
#### quantiles of a column, proportion of target variable to determine class imbalance


str(pure_data)
head(pure_data, 5)
tail(pure_data, 5)

summary(pure_data)
colSums(is.na(pure_data))
sum(is.na(pure_data))
prop.table(table(pure_data$ExtraTime))

####################### Remove unnecessary columns #################################
### This involves removing columns that have most missing values and any column representing
### row id's


drop_cols = c("RowID", "istrain", "Over18", "EmployeeID", "EmployeeCount", "StandardHours") #datacollected
pure_data = pure_data[, !colnames(pure_data) %in% drop_cols]

pure_data$workExp <- as.numeric(as.Date(as.character(pure_data$datacollected), format="%m/%d/%Y")-
  as.Date(as.character(pure_data$FirstJobDate), format="%m/%d/%Y"))

pure_data$daysInCurrentCompany <- as.numeric(as.Date(as.character(pure_data$datacollected), format="%m/%d/%Y")-
  as.Date(as.character(pure_data$DateOfjoiningintheCurrentCompany), format="%m/%d/%Y"))

str(pure_data)

drop_date_cols = c("datacollected", "DateOfjoiningintheCurrentCompany", "FirstJobDate") #

pure_data = pure_data[, !colnames(pure_data) %in% drop_date_cols]

###### Type casting the columns into required datatypes ####################################
### Seperate numeric and categorical attr and convert into respective datatypes


col_names = colnames(pure_data)

## Include target variable as numeric (for timebeing) if it is classification problem
cat_attr = c("FrequencyofTravel", "Gender", "Division", "JobRole", "ExtraTime", 
             "MaritalStatus", "Specialization")
num_attr = setdiff(col_names, cat_attr)

num_dataframe = subset(x = pure_data, select = num_attr)

num_dataframe = data.frame(apply(X = num_dataframe, MARGIN = 2, FUN = function(x){as.numeric(x)}))
str(num_dataframe)

## Repeat last step for categorical attr if necessary

cat_dataframe = subset(pure_data, cat_attr)
cat_dataframe = data.frame(apply(X = cat_dataframe, MARGIN = 2, FUN = function(x){as.factor(x)}))

### Binding the columns back
##Tip: Bind such that target variable is the last column

pure_data = data.frame(cbind(cat_dataframe, num_dataframe))
str(pure_data)

##If necessary convert target variable as factor
pure_data$target = as.factor(pure_data$target)

### Clear all temp variables
rm(num_attr, col_names, num_dataframe, cat_dataframe)

################## Any extra steps needed for the dataset ############################



############################ Splitting train data into train and test #########################
### Split data into train and validation in 70:30 ratio respectively
### Should be done before any kind of data altering/modification steps are taken up, as we need
### validation data to replicate test data and keep it completely unaware of data in train dataset
### Doing stratified samping such that we manintain class proportionality in 
### both train and validation datasets


set.seed(143)
train_rows = createDataPartition(y = pure_data$ExtraTime, p = 0.7, list = F)
train_pure = pure_data[train_rows, ]
val_pure = pure_data[-train_rows, ]

rm(train_rows)


########################### Imputation ################################################
##1) Decide on when to do imputation. If it is only numeric data and if you feel that attributes
## are highly correlated, do imputation post correlation (as corelation can take NA's)
##2) If the attr is a mix bag of numeric and cat, please split into train and validation data
## first and do the imputation
##3) IMP: Always do split the data into train and validation and do the imputation


##Decide on central or knn imputation
train_imputed = centralImputation(data = train_pure)
##train_imputed = knnImputation(data = train_pure, k = 8, scale = T, meth = 'weighAvg')

val_imputed = centralImputation(data = val_pure)
##val_imputed = knnImputation(data = val_pure, k = 8, scale = T, meth = 'weighAvg')

sum(is.na(train_imputed))
sum(is.na(val_imputed))
## Remove any unnecessary variables
rm(pure_data)

str(train_imputed)
str(val_imputed)
