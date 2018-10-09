########################## Logistic Regression ###########################################
####
####
########################## Setting up right data ########################################

## If non-standardized data to be used
train_data = train_dummy
val_data = val_dummy

#If Standardized data to be used
train_data = train_std
val_data = val_std

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
#Predicting probabilities on train data with 0.5
log_train_proba = predict(object = log_reg, newdata = train_data, type='response')
log_train_class = ifelse(log_train_proba > 0.44, "Yes", "No")
caret::confusionMatrix(data = as.factor(log_train_class), reference = as.factor(train_data$target1), positive = 'Yes')
##Tune between 0.44 and 0.45
##Accuracy is 0.6525 sensitivity is 0.7337

#Predicting probabilities on test data with 0.5
log_val_proba = predict(object = log_reg, newdata = val_data, type = 'response')
log_val_class = ifelse(log_val_proba > 0.44, "Yes", "No")


#Accuracy test using caret package
caret::confusionMatrix(data = as.factor(log_val_class), reference = as.factor(val_data$target1), positive = 'Yes')
##Accuracy is 0.6561 sensitivity is 0.7357


#getF1Stat(as.factor(log_val_class),val_data$target1)
#0.9746

rm(train_log_preds, train_log_perf, train_auc_obj, basic_auc)

####################### Using MLR package for logistic regressions #############################
library(mlr)
train_task = makeClassifTask(data = train_data, target = 'target1', positive = 'Yes')
val_task = makeClassifTask(data = val_data, target = 'target1', positive = 'Yes')

#train_task

#log.learner = makeLearner("classif.logreg", predict.type = 'response')
#cv.log = crossval(learner = log.learner, task = train_task, iters = 5, stratify = T, measures = f1, show.info = T)
#cv.log$aggr

#log.model = train(learner = log.learner, task = train_task)
#getLearnerModel(log.model)

#log.train.class = predict(log.model, train_task)
##log.train.class$data$response
#log.val.class = predict(log.model, val_task)
#confusionMatrix(as.factor(log.val.class$data$response), val_data$target1)
#getF1Stat(as.factor(log.val.class$data$response), val_data$target1)
#.97384

