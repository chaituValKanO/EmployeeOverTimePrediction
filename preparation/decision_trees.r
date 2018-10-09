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

#getF1Stat(c50_val_class, val_data$target1)
#0.980

########## Using MLR package for c50 ##################
c50.learner = makeLearner("classif.C50", predict.type = 'response')
getParamSet("classif.C50")

set_cv = makeResampleDesc(method = 'CV', stratify = T, iters = 20) #reps = 10, folds = 4
c50.params = makeParamSet(
  makeNumericParam("CF", lower = 0.4, upper = 0.7))
gs_control = makeTuneControlGrid()
c50.tune = tuneParams(learner = c50.learner, task = train_task, resampling = set_cv,
                      measures = , par.set = c50.params, control = gs_control)
c50.tune$x
c50.tune$y

c50.learner = setHyperPars(c50.learner, par.vals = c50.tune$x)

c50.train = mlr::train(c50.learner, train_task)
c50.train.class = predict(c50.train, train_task)
c50.val.class = predict(object = c50.train, val_task)

caret::confusionMatrix(as.factor(c50.val.class$data$response), val_data$target1)
#0.9557
getF1Stat(as.factor(c50.val.class$data$response), val_data$target1)
#0.9771

####################### Rpart ##########################################################
rpart_model = rpart(target1 ~., data = train_data, method = 'class')
rpart_model$variable.importance

rpart_train_class = predict(object = rpart_model, newdata = train_data, type='class')
rpart_val_class = predict(object = rpart_model, newdata = val_data, type = 'class')

confusionMatrix(rpart_val_class, val_data$target1, positive = "Yes")
##Accuracy 0.7218 sens 0.6407

#getF1Stat(rpart_val_class, val_data$target1)
#0.9758

##### Using Rpart with mlr tuning
rpart.learner = makeLearner("classif.rpart", predict.type = 'response')
set_cv = makeResampleDesc(method = 'CV', stratify = T, iters = 20) #for RepCV reps = 10, folds = 4
rpart.params = makeParamSet(makeIntegerParam("minsplit",lower = 30, upper = 50),
                            makeNumericParam("cp", lower = 0.0002, upper = 0.0003))

#makeIntegerParam("minbucket", lower = 10, upper = 30),
raprt.tune = tuneParams(learner = rpart.learner, task = train_task, resampling = set_cv,
                        par.set = rpart.params, measures = f1, control = gs_control)


rpart.learner = setHyperPars(learner = rpart.learner, par.vals = raprt.tune$x)
rpart.train = mlr::train(learner = rpart.learner, task = train_task)
#getLearnerModel(rpart.train)

rpart.train.class = predict(rpart.train, train_task)
rpart.val.class = predict(rpart.train, val_task)
caret::confusionMatrix(as.factor(rpart.val.class$data$response), val_data$target1)
#0.9495
getF1Stat(as.factor(rpart.val.class$data$response), val_data$target1)
#0.9739

#################### Random forests and tuning #############################################
library(randomForest)
rf_model = randomForest(target1 ~., data = train_data, keep.forest = T, ntree = 100)
rf_model
varImpPlot(rf_model)
rf_imp_attr = data.frame(round(rf_model$importance, 2))
rf_imp_attr = data.frame(row.names(rf_imp_attr), rf_imp_attr[, 1])
colnames(rf_imp_attr) = c("Attribute", "Importance")
rf_imp_attr = rf_imp_attr[order(rf_imp_attr$Importance, decreasing = T), ]

head(rf_imp_attr, 20)

rf_train_class = predict(object = rf_model, 
                         newdata = train_data[, !colnames(train_data) %in% c('target1')],
                         norm.votes = T, type='response')

rf_val_class = predict(object = rf_model, 
                       newdata = val_data[, !colnames(val_data) %in% c('target1')], 
                       norm.votes = T, type = 'response')

confusionMatrix(rf_val_class, val_data$target1, positive = 'Yes')
#Accuracy 0.7519 Sensitivity 0.7252
#getF1Stat(rf_val_class, val_data$target1)
#0.97511

####### Tuning RF

rf_tune_model = tuneRF(x = train_data[, !colnames(train_data) %in% c("target1")], y = train_data$target1, 
                       stepFactor = 1.5, improve = 0.01, trace = T, plot = T, ntreeTry = 500)

print(rf_tune_model)
best.m <- rf_tune_model[rf_tune_model[, 2] == min(rf_tune_model[, 2]), 1]
best.m
#28

rf_besttune_model = randomForest(target1~., data = train_data, mtry = best.m, importance = T, ntree = 500)

importance(rf_besttune_model)
rf_tuned_imp_attr = data.frame(round(rf_besttune_model$importance[, 4], 2))
rf_tuned_imp_attr = data.frame(rownames(rf_tuned_imp_attr), rf_tuned_imp_attr[, 1])
colnames(rf_tuned_imp_attr) = c("Attribute", "Importance")

rf_tuned_imp_attr = rf_tuned_imp_attr[order(rf_tuned_imp_attr$Importance, decreasing = T), ]
head(rf_tuned_imp_attr, 20)

rf_tuned_train_class = predict(object = rf_besttune_model, train_data[, !colnames(train_data) %in% c('target1')],
                               type = 'response', norm.votes = T)

rf_tuned_val_class = predict(object = rf_besttune_model, newdata = val_data[, !colnames(val_data) %in% c('target1')],
                             type = 'response', norm.votes = T)


confusionMatrix(rf_tuned_val_class, val_data$target1, positive = "Yes")
#Accuracy 0.7545 sens = 0.7373

#getF1Stat(rf_tuned_val_class, val_data$target1)
#0.9760

######## Using top 30 variables out of 41 variables from rf tuned model
rf_tuned_imp_attr30 = as.character(rf_tuned_imp_attr[1:20, "Attribute"])
rf_tuned_imp_attr30 = append(rf_tuned_imp_attr30, 'target1')
rf_tuned_imp_attr30

rf_besttune_topattr_model = randomForest(target1 ~., data = train_data[, rf_tuned_imp_attr30], mtry = best.m, importance = T, ntree = 100)

rf_tuned_topattr_train_class = predict(object = rf_besttune_topattr_model, 
                                       newdata = train_data[, rf_tuned_imp_attr30[-length(rf_tuned_imp_attr30)]],
                                       type = 'response', norm.votes = T)
rf_tuned_topattr_val_class = predict(object = rf_besttune_topattr_model, 
                                  newdata = val_data[, rf_tuned_imp_attr30[-length(rf_tuned_imp_attr30)]],
                                  type = 'response', norm.votes = T)
confusionMatrix(rf_tuned_topattr_val_class, val_data$target1, positive = 'Yes')
#Accuracy= 0.7493 sens = 0.7203
#getF1Stat(rf_tuned_val_class, val_data$target1)
#0.9760

## Obs: Same accuracy and F1 score as before

rm(rf_tuned_imp_attr, rf_tuned_imp_attr30)

######### Random forest using mlr package
getParamSet("classif.randomForest")

rf.learner = makeLearner('classif.randomForest', predict.type = 'response')
rf.param <- makeParamSet(
  makeIntegerParam("ntree",lower = 100, upper = 300),
  makeIntegerParam("mtry", lower = 5, upper = 12)
)

rancontrol <- makeTuneControlRandom(maxit = 5L)
rf.tune = tuneParams(learner = rf.learner, task = train_task, resampling = set_cv, measures = tpr, 
                     par.set = rf.param, control = rancontrol)

rf.tune$x
rf.learner = setHyperPars(learner = rf.learner, par.vals = rf.tune$x)
rf.train = train(learner = rf.learner, task = train_task)

rf.train.class = predict(rf.train, train_task)
rf.val.class = predict(rf.train, val_task)

caret::confusionMatrix(as.factor(rf.val.class), val_data$target1)
#getF1Stat(as.factor(rf.val.class), val_data$target1)



