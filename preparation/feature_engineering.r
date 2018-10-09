################################# Feature Engineering #########################################

##If you want to consider already imputed dataset execute below lines 
#train_dummy = train_imputed
#val_dummy = val_imputed

##If unimputed dataset is to be taken into consideration
train_dummy = train_pure
val_dummy = val_pure

##################### Loading required libraries for feature engineering #######################

library(corrplot)
library(caret)

######################### Numerical Attr ##############################################
##### Finding correlation matrix among numeric attr

cor_mat = cor(train_dummy[, !colnames(train_dummy) %in% cat_attr], use = "complete.obs")
cor_mat = round(cor_mat, 2)

#Removed all features with >0.8
high_cor_features = findCorrelation(x = cor_mat, cutoff = 0.8, verbose = T, exact = T, names = T)
high_cor_features

## Dropping one of the features with correlation >0.9
## However we are retaining one feature in the pair

train_dummy = train_dummy[, !colnames(train_dummy) %in% high_cor_features]

## Recalculate the correlation matrix with truncated dataframe

cor_mat = cor(train_dummy[, !colnames(train_dummy) %in% cat_attr], use = "complete.obs")
cor_mat = round(cor_mat, 2)



train_dummy$target1 = train_dummy$ExtraTime
train_dummy$ExtraTime = NULL
str(train_dummy)

############# Grouping of levels under specific categorical attr and Dummification ############
## Sample code. Replace values accordingly
#Lets group/categorize the qualification variable. 
#First we categorize then we shall try to make ordinal

train_dummy$JobRole = gsub(pattern = "^Manufacturing Director", replacement = "Director", x = train_dummy$JobRole)
train_dummy$JobRole = gsub(pattern = "^Research Director", replacement = "Director", x = train_dummy$JobRole)
train_dummy$JobRole = gsub(pattern = "^Sales Executive", replacement = "Sales", x = train_dummy$JobRole)
train_dummy$JobRole = gsub(pattern = "^Sales Representative", replacement = "Sales", x = train_dummy$JobRole)
train_dummy$JobRole = as.factor(train_dummy$JobRole)


############# Ordinalizing Frequency of travel levels. Feature becomes numeric attr ###############
##Now lets dummify this qualification varibale with ordinality low as 1, medium as 2, high as 3, veryhigh as 4 

train_dummy$FrequencyofTravel = ifelse(train_dummy$FrequencyofTravel == 'NoTravel', 0, 
                                               ifelse(train_dummy$FrequencyofTravel == 'Less', 1, 3))

str(train_dummy)
table(train_dummy$FrequencyofTravel)

col_names = colnames(train_dummy)

## Include target variable as numeric (for timebeing) if it is classification problem
cat_attr = c("Gender", "Division", "JobRole", 
             "MaritalStatus", "Specialization")

num_attr = setdiff(col_names, cat_attr)
num_attr = num_attr[-length(num_attr)]
num_df = train_dummy[, colnames(train_dummy) %in% num_attr]
num_df$target1 = train_dummy$target1
str(num_df)


cat_dataframe = train_dummy[, colnames(train_dummy) %in% cat_attr]
dummy_df = dummyVars(~., data = cat_dataframe, sep = '.')
cat_dummy_df = as.data.frame(predict(object = dummy_df, cat_dataframe))
train_dummified = data.frame(cbind(cat_dummy_df,num_df))

str(train_dummified)

############## Repeat same feature engineering steps for validation data ######################
#### Note this section has steps only for numerical attr. Please add cat attr steps if any 

val_dummy = val_dummy[, !colnames(val_dummy) %in% high_cor_features]

val_dummy$target1 = val_dummy$ExtraTime
val_dummy$ExtraTime = NULL

val_dummy$JobRole = gsub(pattern = "^Manufacturing Director", replacement = "Director", x = val_dummy$JobRole)
val_dummy$JobRole = gsub(pattern = "^Research Director", replacement = "Director", x = val_dummy$JobRole)
val_dummy$JobRole = gsub(pattern = "^Sales Executive", replacement = "Sales", x = val_dummy$JobRole)
val_dummy$JobRole = gsub(pattern = "^Sales Representative", replacement = "Sales", x = val_dummy$JobRole)
val_dummy$JobRole = as.factor(val_dummy$JobRole)


val_dummy$FrequencyofTravel = ifelse(val_dummy$FrequencyofTravel == 'NoTravel', 0, 
                                     ifelse(val_dummy$FrequencyofTravel == 'Less', 1, 3))

str(val_dummy)
table(val_dummy$FrequencyofTravel)

num_df = val_dummy[, colnames(val_dummy) %in% num_attr]
num_df$target1 = val_dummy$target1
str(num_df)


cat_dataframe = val_dummy[, colnames(val_dummy) %in% cat_attr]
dummy_df = dummyVars(~., data = cat_dataframe, sep = '.')
cat_dummy_df = as.data.frame(predict(object = dummy_df, cat_dataframe))
val_dummified = data.frame(cbind(cat_dummy_df,num_df))

rm(cat_dummy_df, cor_mat, num_df, pure_data_file, dummy_df)

########################### Custom Functions #################################################

getF1Stat = function(preds, preds_ref){
  recall = sensitivity(data = preds, reference = preds_ref)
  prec = precision(data = preds, reference = preds_ref)
  f1score = 2 * recall * prec /(recall + prec)
  return (f1score)
}

########################### Imputation ################################################
##1) Decide on when to do imputation. If it is only numeric data and if you feel that attributes
## are highly correlated, do imputation post correlation
##2) If the attr is a mix bag of numeric and cat, please split into train and validation data
## first and do the imputation
##3) IMP: Always do split the data into train and validation and do the imputation


##Decide on central or knn imputation
train_dummy = centralImputation(data = train_dummy)
val_dummy = centralImputation(data = val_dummy)
##train_dummy = knnImputation(data = train_dummy, k = 8, scale = T, meth = 'weighAvg')
##val_dummy = knnImputation(data = val_dummy, k = 8, scale = T, meth = 'weighAvg')
sum(is.na(train_dummy))
sum(is.na(val_dummy))

##Check for integrity of data
str(train_dummy)
str(val_dummy)

## Remove any unnecessary variables
#rm(train_pure)
#rm(val_pure)




########################### Standardizing the numerical attrs #################################
########## IMP: Do not standardize unless you have distance based algorithms

std_obj = preProcess(train_dummy[, !colnames(train_dummy) %in% c("target1")], method = c('center', 'scale'))

train_std = predict(object = std_obj, newdata = train_dummy)
val_std = predict(object = std_obj, newdata = val_dummy)

summary(train_std)
summary(val_std)

#Use the same object for prediction on test data either

##Remove after std test data as well 
rm(std_obj)






