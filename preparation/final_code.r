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

## Splitting the entire dataset into train and test based on istrain attribute
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

##Observation: No class imbalance and no NA's available

###################### Feature Engineering ####################################################
#
#
####################### Remove unnecessary columns #################################
### This involves removing columns that have most missing values and any column representing
### row id's and columns with zero variance

## All the below columns are identified as zero variance  columns

drop_cols = c("RowID", "istrain", "Over18", "EmployeeID", "EmployeeCount", "StandardHours")
pure_data = pure_data[, !colnames(pure_data) %in% drop_cols]


### Decided to make use of date columns to get number of working days till date
### And also in the latest company.
### So subtracting dataCollected attr from firstjobdate and also DateOfjoiningintheCurrentCompany
### so new variables as workExp and daysInCurrentCompany are created

pure_data$workExp = as.numeric(as.Date(as.character(pure_data$datacollected), format="%m/%d/%Y")-
                                  as.Date(as.character(pure_data$FirstJobDate), format="%m/%d/%Y"))

pure_data$daysInCurrentCompany = as.numeric(as.Date(as.character(pure_data$datacollected), format="%m/%d/%Y")-
                                               as.Date(as.character(pure_data$DateOfjoiningintheCurrentCompany), format="%m/%d/%Y"))

str(pure_data)

#Droping of the parent classes that were feature engineered
drop_date_cols = c("datacollected", "DateOfjoiningintheCurrentCompany", "FirstJobDate")

############################ Splitting train data into train and test #########################
### Split data into train and validation in 70:30 ratio respectively
### Should be done before any kind of data altering/modification steps are taken up, as we need
### validation data to replicate test data and keep it completely unaware of data in train dataset
### Doing stratified samping such that we manintain class proportionality in 
### both train and validation datasets.
### All the operations till now are btween features and would effect purity of validation dataset


set.seed(143)
train_rows = createDataPartition(y = pure_data$ExtraTime, p = 0.7, list = F)
train_pure = pure_data[train_rows, ]
val_pure = pure_data[-train_rows, ]

rm(train_rows)


##Creating new dataframes as to be in sync with my scripts
train_dummy = train_pure
val_dummy = val_pure

################################# Plots #####################################################
####
####
####
###############################################################################################
library(tidyverse)

####Bivariate plots
##Acheieved by changing the varibales names and saving the graphs

train_dummy %>%
  group_by(target1) %>%
  ggplot() +
  geom_jitter(data = train_dummy %>% 
                group_by(target1),
              aes(x = target1, y = workExp, color = factor(target1)), 
              position = position_jitter(w = 0.3, h = 0),
              alpha = 0.5) +
  geom_point(data = train_dummy %>% 
               group_by(target1),
             aes(x = target1, y = workExp, color = factor(target1)),
             size = 5, alpha = 0.2) +
  labs(x = "Extra Hours(0/1)",
       y = "WorkExp (Days)",
       title = "WorkExp and Extra Hours")


### Univariate Plots
train_dummy %>%
  ggplot(aes(x = YearsInCurrentRole)) +
  geom_histogram()

##################### Loading required libraries for correlation #######################

library(corrplot)
library(caret)

######################### Numerical Attr ##############################################
##### Finding correlation matrix among numeric attr

cor_mat = cor(train_dummy[, !colnames(train_dummy) %in% cat_attr], use = "complete.obs")
cor_mat = round(cor_mat, 2)

### Checking for attr with correlation >= 0.9 and then changed to 0.8. Both cutoffs
### had same attribute Job_level
###However correlation > 0.7 has 4 attributes but keeping them as of now

## Finally Removed all features with correlation >=0.8
## Using findcorrelation from caret package to achieve the same. It uses an intelligent algorithm

high_cor_features = findCorrelation(x = cor_mat, cutoff = 0.8, verbose = T, exact = T, names = T)
high_cor_features

train_dummy = train_dummy[, !colnames(train_dummy) %in% high_cor_features]

## Recalculate the correlation matrix with truncated dataframe

cor_mat = cor(train_dummy[, !colnames(train_dummy) %in% cat_attr], use = "complete.obs")
cor_mat = round(cor_mat, 2)

### Observation: found nothing >=0.8 after droping the "JobLevel"

## Just changing target varibale to target1 to keep in sync with my scripts
train_dummy$target1 = train_dummy$ExtraTime
train_dummy$ExtraTime = NULL
str(train_dummy)

############# Grouping of levels under specific categorical attr and Dummification ############
#Lets group/categorize the qualification variable. 
#JobRole has some similar levels as director, so grouping them
#Also grouping sales related persons as one

train_dummy$JobRole = gsub(pattern = "^Manufacturing Director", replacement = "Director", x = train_dummy$JobRole)
train_dummy$JobRole = gsub(pattern = "^Research Director", replacement = "Director", x = train_dummy$JobRole)
train_dummy$JobRole = gsub(pattern = "^Sales Executive", replacement = "Sales", x = train_dummy$JobRole)
train_dummy$JobRole = gsub(pattern = "^Sales Representative", replacement = "Sales", x = train_dummy$JobRole)
train_dummy$JobRole = as.factor(train_dummy$JobRole)


######## Ordinalizing Frequency of travel levels. Feature becomes numeric attr ############
##Now lets dummify this qualification varibale with ordinality low as 1, medium as 2, high as 3, veryhigh as 4 

train_dummy$FrequencyofTravel = ifelse(train_dummy$FrequencyofTravel == 'NoTravel', 0, 
                                       ifelse(train_dummy$FrequencyofTravel == 'Less', 1, 3))

str(train_dummy)
table(train_dummy$FrequencyofTravel)

############################### Dummification process ###################################
### Most of classification models will work with categorical variables. 
### However native xgboost function requires a matrix, hence dummifying

##Listing all the categorical variables
cat_attr = c("Gender", "Division", "JobRole", 
             "MaritalStatus", "Specialization")

#Filtering the numeric attributes
num_attr = setdiff(col_names, cat_attr)
num_attr = num_attr[-length(num_attr)]

## Creating numeric and categorical data frames
num_df = train_dummy[, colnames(train_dummy) %in% num_attr]
num_df$target1 = train_dummy$target1
str(num_df)


cat_dataframe = train_dummy[, colnames(train_dummy) %in% cat_attr]
dummy_df = dummyVars(~., data = cat_dataframe, sep = '.')
cat_dummy_df = as.data.frame(predict(object = dummy_df, cat_dataframe))

## Merging the dataframes
train_dummified = data.frame(cbind(cat_dummy_df,num_df))

str(train_dummified)

############## Repeat same feature engineering steps for validation data ######################
####
####
####
###############################################################################################

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

########################## Logistic Regression ###########################################
####
####
########################## Setting up right data ########################################

## Switching variables to keep sync with my scripts
train_data = train_dummy
val_data = val_dummy


########################## Model #######################################################
log_reg = glm(formula = target1 ~., data = train_data, family = 'binomial')
log_reg ###Returns log values and not probabilities

log_train_proba = predict(object = log_reg, type = 'response')
library(ROCR)
train_log_preds = prediction(log_train_proba, train_data$target1)
#class(train_log_preds)
train_log_perf = performance(prediction.obj = train_log_preds, measure = 'tpr', x.measure = 'fpr')
plot(train_log_perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))

#Extracting AUC from performance object

train_auc_obj = performance(train_log_preds, measure = 'auc')
basic_auc = train_auc_obj@y.values[[1]]
basic_auc
#0.702

########## Train - Test data ##################################
##Predicting probabilities on train data with 0.5 and 0.4
##Observed that accuracy changes significantly between these two values
####Tune between 0.44 and 0.45


log_train_proba = predict(object = log_reg, newdata = train_data, type='response')
log_train_class = ifelse(log_train_proba > 0.44, "Yes", "No")
caret::confusionMatrix(data = as.factor(log_train_class), reference = as.factor(train_data$target1), positive = 'Yes')
##Accuracy is 0.6525 sensitivity is 0.7337

#Predicting probabilities on test data with 0.44
log_val_proba = predict(object = log_reg, newdata = val_data, type = 'response')
log_val_class = ifelse(log_val_proba > 0.44, "Yes", "No")


#Accuracy test using caret package
caret::confusionMatrix(data = as.factor(log_val_class), reference = as.factor(val_data$target1), 
                       positive = 'Yes')
##Accuracy is 0.6561 sensitivity is 0.7357

########################## Decision Trees ###############################################
library(C50)
library(rpart)

########################## Setting up right data ########################################

## If non-standardized data to be used
train_data = train_dummy
val_data = val_dummy

#If Standardized data to be used
train_data = train_std
val_data = val_std

########################## Model #######################################################

########################## C50 ##########################################################
c50_model = C5.0(target1 ~., data = train_data, rules = T) 
C5imp(object = c50_model, metric = 'usage')

c50_train_class = predict(object = c50_model, 
                          newdata = train_data[, !colnames(train_data) %in% c('target1')])
c50_val_class = predict(object = c50_model, 
                        newdata = val_data[, !colnames(val_data) %in% c('target1')])

caret::confusionMatrix(c50_val_class, val_data$target1, positive = "Yes")
#Accuracy 0.7539 Sensitivity 0.7330

####################### Rpart ##########################################################
rpart_model = rpart(target1 ~., data = train_data, method = 'class')
rpart_model$variable.importance

rpart_train_class = predict(object = rpart_model, newdata = train_data, type='class')
rpart_val_class = predict(object = rpart_model, newdata = val_data, type = 'class')

confusionMatrix(rpart_val_class, val_data$target1, positive = "Yes")
##Accuracy 0.7218 sens 0.6407

#################### Random forests and tuning #############################################
library(randomForest)
rf_model = randomForest(target1 ~., data = train_data, keep.forest = T, ntree = 100)
rf_model

########## Determining the important variables ##################
varImpPlot(rf_model)
rf_imp_attr = data.frame(round(rf_model$importance, 2))
rf_imp_attr = data.frame(row.names(rf_imp_attr), rf_imp_attr[, 1])
colnames(rf_imp_attr) = c("Attribute", "Importance")
rf_imp_attr = rf_imp_attr[order(rf_imp_attr$Importance, decreasing = T), ]

head(rf_imp_attr, 20)

##############################Predictions using the randomeforest #########################

rf_train_class = predict(object = rf_model, 
                         newdata = train_data[, !colnames(train_data) %in% c('target1')],
                         norm.votes = T, type='response')

rf_val_class = predict(object = rf_model, 
                       newdata = val_data[, !colnames(val_data) %in% c('target1')], 
                       norm.votes = T, type = 'response')

confusionMatrix(rf_val_class, val_data$target1, positive = 'Yes')
#Accuracy 0.7519 Sensitivity 0.7252

####### Tuning RF

rf_tune_model = tuneRF(x = train_data[, !colnames(train_data) %in% c("target1")], y = train_data$target1, 
                       stepFactor = 1.5, improve = 0.01, trace = T, plot = T, ntreeTry = 500)

######### Fetching the best number of features
print(rf_tune_model)
best.m <- rf_tune_model[rf_tune_model[, 2] == min(rf_tune_model[, 2]), 1]
best.m
#6 or 7

############ Using best tuned metric and building randomeforest

rf_besttune_model = randomForest(target1~., data = train_data, mtry = best.m, 
                                 importance = T, ntree = 500)

#############################Important features for random forest tuned model###################

importance(rf_besttune_model)
rf_tuned_imp_attr = data.frame(round(rf_besttune_model$importance[, 4], 2))
rf_tuned_imp_attr = data.frame(rownames(rf_tuned_imp_attr), rf_tuned_imp_attr[, 1])
colnames(rf_tuned_imp_attr) = c("Attribute", "Importance")

rf_tuned_imp_attr = rf_tuned_imp_attr[order(rf_tuned_imp_attr$Importance, decreasing = T), ]
head(rf_tuned_imp_attr, 20)


####################### Predictions using the tuned randome forest ########################
rf_tuned_train_class = predict(object = rf_besttune_model, train_data[, !colnames(train_data) %in% c('target1')],
                               type = 'response', norm.votes = T)

rf_tuned_val_class = predict(object = rf_besttune_model, newdata = val_data[, !colnames(val_data) %in% c('target1')],
                             type = 'response', norm.votes = T)


confusionMatrix(rf_tuned_val_class, val_data$target1, positive = "Yes")
#Accuracy 0.7545 sens = 0.7373


######## Using top 20 variables from tuned random forest model ##############################

rf_tuned_imp_attr30 = as.character(rf_tuned_imp_attr[1:20, "Attribute"])
rf_tuned_imp_attr30 = append(rf_tuned_imp_attr30, 'target1')
rf_tuned_imp_attr30

################Building model using top20 varibales#####################################

rf_besttune_topattr_model = randomForest(target1 ~., data = train_data[, rf_tuned_imp_attr30], 
                                         mtry = best.m, importance = T, ntree = 100)

rf_tuned_topattr_train_class = predict(object = rf_besttune_topattr_model, 
                                       newdata = train_data[, rf_tuned_imp_attr30[-length(rf_tuned_imp_attr30)]],
                                       type = 'response', norm.votes = T)
rf_tuned_topattr_val_class = predict(object = rf_besttune_topattr_model, 
                                     newdata = val_data[, rf_tuned_imp_attr30[-length(rf_tuned_imp_attr30)]],
                                     type = 'response', norm.votes = T)

confusionMatrix(rf_tuned_topattr_val_class, val_data$target1, positive = 'Yes')
#Accuracy= 0.7493 sens = 0.7203
## Obs: Same accuracy and F1 score as before

rm(rf_tuned_imp_attr, rf_tuned_imp_attr30)

######### Random forest using mlr package
library(mlr)

##### step1: Creating task required for mlr package

train_task = makeClassifTask(data = train_data, target = 'target1', positive = 'Yes')
val_task = makeClassifTask(data = val_data, target = 'target1', positive = 'Yes')

###Get all tunable params for randomeforest under mlr package
getParamSet("classif.randomForest")

##### step2: Creating learner required for mlr package
rf.learner = makeLearner('classif.randomForest', predict.type = 'response')
rf.param <- makeParamSet(
  makeIntegerParam("ntree",lower = 100, upper = 300),
  makeIntegerParam("mtry", lower = 5, upper = 12)
)

#### Creating a random walkover the grid of params
rancontrol <- makeTuneControlRandom(maxit = 5L)

#### Creating a sampling stratgies using stratify such that class balance is maintained
set_cv = makeResampleDesc(method = 'CV', stratify = T, iters = 20)

rf.tune = tuneParams(learner = rf.learner, task = train_task, resampling = set_cv, 
                     measures = tpr, 
                     par.set = rf.param, control = rancontrol)

rf.tune$x
rf.learner = setHyperPars(learner = rf.learner, par.vals = rf.tune$x)
rf.train = train(learner = rf.learner, task = train_task)

rf.train.class = predict(rf.train, train_task)
rf.val.class = predict(rf.train, val_task)

caret::confusionMatrix(as.factor(rf.val.class), val_data$target1)

############################ Boosting Algorithms ############################################
library(xgboost)
########################## Setting up right data ########################################

## Using dummified dataframe
train_data = train_dummified
val_data = val_dummified


########################## Model #######################################################

##To get target varibale as o and 1 instead of 1 and 2
train_data$target1 = as.numeric(train_data$target1) - 1
val_data$target1 = as.numeric(val_data$target1) -1

train_data$target1 = as.factor(train_data$target1)
val_data$target1 = as.factor(val_data$target1)

########Creating matrices #########################

train_matrix = xgb.DMatrix(data = as.matrix(train_data[, !colnames(train_data) %in% c("target1")]),
                           label = as.matrix(train_data[, colnames(train_data) %in% c('target1')]))

val_matrix = xgb.DMatrix(data = as.matrix(val_data[, !colnames(val_data) %in% c("target1")]), 
                         label = as.matrix(val_data[, colnames(val_data) %in% c("target1")]))

params_list <- list("objective" = "binary:logistic",
                    "eta" = 0.1,
                    "early_stopping_rounds" = 10,
                    "max_depth" = 6,
                    "gamma" = 0.5,
                    "colsample_bytree" = 0.6,
                    "subsample" = 0.65,
                    "eval_metric" = "auc",
                    "silent" = 1)

xgb_model_with_params <- xgboost(data = train_matrix, params = params_list, nrounds = 500
                                 , early_stopping_rounds = 20)


###train. Decide on cutoff from AUC curve
xgb_train_proba <- predict(xgb_model_with_params, train_matrix, type="response", norm.votes=TRUE) 

#check auc 

xgb_train_pred_obj <- prediction(xgb_train_proba, train_data$target1)
perf_train <- ROCR::performance(xgb_train_pred_obj, measure = "tpr", x.measure ="fpr")

plot(perf_train,col = rainbow(10),colorize = T,print.cutoffs.at= seq(0,1,0.25))
auc_train <- ROCR::performance(xgb_train_pred_obj,measure = "auc")
auc_train@y.values[[1]]
xgb_train_pred <- ifelse(xgb_train_proba > 0.16 , 1 , 0)

caret::confusionMatrix(as.factor(xgb_train_pred), train_data$target1, positive="1")
#Accuracy 0.6592 Sens = 0.9817


## Predictions on Validation Data
xgb_val_proba <- predict(xgb_model_with_params, val_matrix, type="response", norm.votes=TRUE) 

#check auc 

xgb_val_pred_obj <- prediction(xgb_val_proba, val_data$target1)
perf <- performance(xgb_val_pred_obj, measure = "tpr", x.measure ="fpr")

plot(perf,col = rainbow(10),colorize = T,print.cutoffs.at= seq(0,1,0.05))
auc <- performance(xgb_val_pred_obj,measure = "auc")
auc@y.values[[1]]
xgb_val_pred <- ifelse(xgb_val_proba > 0.16 , 1 , 0)

caret::confusionMatrix(as.factor(xgb_val_pred), val_data$target1, positive="1")
#Accuracy = 0.6036 Sensi = 0.9171


##################### Working on important variables ###############################################
xgb_varimp_matrix <- xgb.importance(feature_names = colnames(train_matrix), 
                                    model = xgb_model_with_params)

xgb.plot.importance(xgb_varimp_matrix)
xgb_imp_attr_30 <- data.frame(xgb_varimp_matrix[1:20,1])[, 'Feature']
xgb_imp_attr_30 =  append(xgb_imp_attr_30, 'target1')
xgb_imp_attr_30

train_data_xgb_impattr = train_data[, xgb_imp_attr_30]
val_data_xgb_impattr = val_data[, xgb_imp_attr_30]

train_matrix_impattr = xgb.DMatrix(data = as.matrix(train_data_xgb_impattr[, !colnames(train_data_xgb_impattr) %in% c("target1")]),
                                   label = as.matrix(train_data_xgb_impattr[, colnames(train_data_xgb_impattr) %in% c('target1')]))

val_matrix_impattr = xgb.DMatrix(data = as.matrix(val_data_xgb_impattr[, !colnames(val_data_xgb_impattr) %in% c("target1")]), 
                                 label = as.matrix(val_data_xgb_impattr[, colnames(val_data_xgb_impattr) %in% c("target1")]))


params_list <- list("objective" = "binary:logistic",
                    "eta" = 0.1,
                    "early_stopping_rounds" = 10,
                    "max_depth" = 6,
                    "gamma" = 0.5,
                    "colsample_bytree" = 0.6,
                    "subsample" = 0.65,
                    "eval_metric" = "auc",
                    "silent" = 1)

xgb_model_with_params_impattr <- xgboost(data = train_matrix_impattr, params = params_list, 
                                         nrounds = 1500, early_stopping_rounds = 20)


###train. Decide on cutoff from AUC curve
xgb_train_proba <- predict(xgb_model_with_params_impattr, train_matrix_impattr, 
                           type="response", norm.votes=TRUE)

#check auc 

xgb_train_pred_obj <- prediction(xgb_train_proba, train_data_xgb_impattr$target1)
perf_train <- ROCR::performance(xgb_train_pred_obj, measure = "tpr", x.measure ="fpr")

plot(perf_train,col = rainbow(10),colorize = T,print.cutoffs.at= seq(0,1,0.25))
auc_train <- ROCR::performance(xgb_train_pred_obj,measure = "auc")
auc_train@y.values[[1]]
#.9563
xgb_train_pred <- ifelse(xgb_train_proba > 0.2 , "1" , "0")

confusionMatrix(as.factor(xgb_train_pred), as.factor(train_data_xgb_impattr$target), positive = "1")
#0.9986



## Validation Data
xgb_val_proba <- predict(xgb_model_with_params_impattr, val_matrix_impattr, 
                         type="response", norm.votes=TRUE) 

#check auc 

xgb_val_pred_obj <- prediction(xgb_val_proba, val_data_xgb_impattr$target1)
perf <- ROCR::performance(xgb_val_pred_obj, measure = "tpr", x.measure ="fpr")

plot(perf,col = rainbow(10),colorize = T,print.cutoffs.at= seq(0,1,0.05))
auc <- ROCR::performance(xgb_val_pred_obj,measure = "auc")
auc@y.values[[1]]
xgb_val_pred <- ifelse(xgb_val_proba > 0.2 , 1 , 0)

confusionMatrix(as.factor(xgb_val_pred), as.factor(val_data_xgb_impattr$target1), positive = "1")
#0.9572


##################### tune the xgboost ################################################
train_data$target1 = as.factor(as.numeric(train_data$target1) - 1)
val_data$target1 = as.factor(as.numeric(val_data$target1) -1)

sampling_strategy = trainControl(method = 'repeatedcv', number = 8, repeats = 4, 
                                 verboseIter = T)

param_grid <- expand.grid(.nrounds = 30, .max_depth = c(4, 6, 8), .eta = c(0.1, 0.3),
                          .gamma = c(0.6, 0.5, 0.4), .colsample_bytree = c(0.6, 0.4),
                          .min_child_weight = 1, .subsample = c(0.6, 0.5 ,0.9))

xgb_tuned_model <- caret::train(x = train_data[ , !(names(train_data) %in% c("target1"))], 
                                y = train_data[ , names(train_data) %in% c("target1")], 
                                method = "xgbTree",
                                trControl = sampling_strategy,
                                tuneGrid = param_grid)

#### Got error

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


#################### Random forests and tuning #############################################
## If non-standardized data to be used
test_data = test_dummy

##If Standardized data to be used
test_data = test_std

rf_test_class = predict(object = rf_model, newdata = test_data, type='response', norm.votes = T)
table(rf_test_class)
test_result = data.frame(cbind(ID, rf_test_class))
colnames(test_result) = c("RowID", "ExtraTime")
test_result$ExtraTime = ifelse(test_result$ExtraTime == 1, "No", "Yes")
write.csv(test_result, file = '..\\submission\\rf.csv', row.names = F)

####Tuned RF model
rf_tuned_test_class = predict(object = rf_besttune_model, newdata = test_data, type = 'response', norm.votes = T)
table(rf_tuned_test_class)
test_result = data.frame(cbind(ID, rf_tuned_test_class))
colnames(test_result) = c("RowID", "ExtraTime")
test_result$ExtraTime = ifelse(test_result$ExtraTime == 1, "No", "Yes")
write.csv(test_result, file="..\\submission\\rf_tuned.csv", row.names = F)

################################# Boosting #################################################
## If non-standardized data to be used
test_data = test_dummified

##If Standardized data to be used
test_data = test_std

xg_params_test_proba = predict(object = xgb_model_with_params, newdata = as.matrix(test_data), 
                               type='response', norm.votes = T)


xg_params_test_class = ifelse(xg_params_test_proba > 0.2, "1", "0")
table(xg_params_test_class)
test_result = data.frame(cbind(ID, xg_params_test_class))
colnames(test_result) = c("RowID", "ExtraTime")
test_result$ExtraTime = ifelse(test_result$ExtraTime == 1, "Yes", "No")
write.csv(test_result, file="..\\submission\\xg_params_new.csv", row.names = F)

#### Imp attr

test_data_impattr = test_data[, xgb_imp_attr_30[-length(xgb_imp_attr_30)]]
xg_params_varimp_test_proba = predict(object = xgb_model_with_params_impattr, 
                                      newdata = as.matrix(test_data_impattr), 
                                      type='response', norm.votes = T)


xg_params_test_class = ifelse(xg_params_varimp_test_proba > 0.2, "1", "0")
table(xg_params_test_class)
test_result = data.frame(cbind(ID, xg_params_test_class))
colnames(test_result) = c("RowID", "ExtraTime")
test_result$ExtraTime = ifelse(test_result$ExtraTime == 1, "Yes", "No")
write.csv(test_result, file="..\\submission\\xg_params_varimp_20.csv", row.names = F)


