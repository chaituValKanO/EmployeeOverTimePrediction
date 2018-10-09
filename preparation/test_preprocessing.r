#################### Set the working directory #######################################
getwd()
setwd("C:\\Users\\chait\\Documents\\Insofe\\MiTh\\data")

################## Load all required libraries ######################################
library(DMwR)
library(caret)
library(tidyverse)

#################### Read the data file #############################################

#test_pure = read.csv(file = "test.csv", header = T) #na.strings = c(":", "?")

test_pure = pure_data_file[pure_data_file$istrain != 1, ]
###################### Descriptive stats ############################################

str(test_pure)
head(test_pure, 5)
tail(test_pure, 5)

summary(test_pure)
colSums(is.na(test_pure))

####################### Remove unnecessary columns #################################
ID = test_pure$RowID

test_pure$ExtraTime = NULL
test_pure = test_pure[, !colnames(test_pure) %in% drop_cols]

test_pure$workExp <- as.numeric(as.Date(as.character(test_pure$datacollected), format="%m/%d/%Y")-
                                  as.Date(as.character(test_pure$FirstJobDate), format="%m/%d/%Y"))

test_pure$daysInCurrentCompany <- as.numeric(as.Date(as.character(test_pure$datacollected), format="%m/%d/%Y")-
                                               as.Date(as.character(test_pure$DateOfjoiningintheCurrentCompany), format="%m/%d/%Y"))

str(test_pure)

drop_date_cols = c("datacollected", "DateOfjoiningintheCurrentCompany", "FirstJobDate") #

test_pure = test_pure[, !colnames(test_pure) %in% drop_date_cols]


###### Seperate numeric and categorical attr and convert into respective datatypes ##########

col_names = colnames(test_pure)

## Include target variable as numeric (for timebeing) if it is classification problem
cat_attr = c("xyz", "abc")
num_attr = setdiff(col_names, cat_attr)

num_dataframe = subset(x = test_pure, select = num_attr)

num_dataframe = data.frame(apply(X = num_dataframe, MARGIN = 2, FUN = function(x){as.numeric(x)}))
str(num_dataframe)

## Repeat last step for categorical attr if necessary

cat_dataframe = subset(test_pure, cat_attr)
cat_dataframe = data.frame(apply(X = cat_dataframe, MARGIN = 2, FUN = function(x){as.factor(x)}))

### Binding the columns back
##Tip: Bind such that target variable is the last column

test_pure = data.frame(cbind(cat_dataframe, num_dataframe))
str(test_pure)

##If necessary convert target variable as factor
#test_pure$target = as.factor(test_pure$target)

### Clear all temp variables
rm(num_attr, col_names, num_dataframe, cat_dataframe)

################## Any extra steps needed for the dataset ############################
test_pure = test_pure[, !colnames(test_pure) %in% high_cor_features]

test_pure$JobRole = gsub(pattern = "^Manufacturing Director", replacement = "Director", x = test_pure$JobRole)
test_pure$JobRole = gsub(pattern = "^Research Director", replacement = "Director", x = test_pure$JobRole)
test_pure$JobRole = gsub(pattern = "^Sales Executive", replacement = "Sales", x = test_pure$JobRole)
test_pure$JobRole = gsub(pattern = "^Sales Representative", replacement = "Sales", x = test_pure$JobRole)
test_pure$JobRole = as.factor(test_pure$JobRole)



test_pure$FrequencyofTravel = ifelse(test_pure$FrequencyofTravel == 'NoTravel', 0, 
                                     ifelse(test_pure$FrequencyofTravel == 'Less', 1, 3))

str(test_pure)
table(test_pure$FrequencyofTravel)

col_names = colnames(test_pure)
cat_attr = c("Gender", "Division", "JobRole", 
             "MaritalStatus", "Specialization")

num_attr = setdiff(col_names, cat_attr)
#num_attr = num_attr[-length(num_attr)]
num_df = test_pure[, colnames(test_pure) %in% num_attr]
#num_df$target1 = test_pure$target1
#str(num_df)


cat_dataframe = test_pure[, colnames(test_pure) %in% cat_attr]
dummy_df = dummyVars(~., data = cat_dataframe, sep = '.')
cat_dummy_df = as.data.frame(predict(object = dummy_df, cat_dataframe))
test_dummified = data.frame(cbind(cat_dummy_df,num_df))

str(train_dummified)


test_dummy = test_pure
str(test_dummy)

########################### Imputation ################################################
#########Based on steps performed on train data decide on imputation


##Decide on central or knn imputation
test_imputed = centralImputation(data = test_pure)
##test_imputed = knnImputation(data = test_pure, k = 8, scale = T, meth = 'weighAvg')
sum(is.na(test_imputed))
str(test_imputed)
## Remove any unnecessary variables
rm(test_pure)
