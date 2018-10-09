########################## Decision Trees ###############################################
library(e1071)
library(kernlab)
library(caret)

########################## Setting up right data ########################################
##Better to use std datasets for svm

## If non-standardized data to be used
train_data = train_dummy
val_data = val_dummy

#If Standardized data to be used
train_data = train_std
val_data = val_std


########################## Model #######################################################
################## SVM linear ##########################################################
svm_linear_model = svm(target1 ~., train_data, kernel = 'linear')

svm_linear_train_class = predict(object = svm_linear_model, train_data)
svm_linear_val_class = predict(svm_linear_model, val_data)
caret::confusionMatrix(svm_linear_val_class, val_data$target1)
getF1Stat(svm_linear_val_class, val_data$target1)

### Tuning for SVM ########
##If the metric to be used is starightforward like accuracy then use caret train
## else if it requires uncomventional metric as F1 score use mlr

##I am using mlr here to train and caret code is in week10 lab session
## with mlr you can train all the kernels in single shot with respective tuning params 

getParamSet("classif.ksvm")
svm.learner = makeLearner("classif.ksvm", predict.type = 'response')

tune_ctrl = makeTuneControlGrid()
svm.params = makeParamSet(makeDiscreteParam("C", values = 2^c(-8,-4,-2, 0)),
                makeDiscreteParam("kernel", values = c("polydot", "rbfdot")),
                makeDiscreteParam("sigma", values = 2^c(-8,-4,0, 4), requires = quote(kernel == "rbfdot")),
                makeIntegerParam("degree", lower = 2L, upper = 5L,
                                       requires = quote(kernel == "polydot")),
                makeNumericParam("scale", lower = 0.15, upper = 0.25, 
                                 requires = quote(kernel == "polydot")))


svm.tune = tuneParams(learner = svm.learner, task = train_task, resampling = set_cv, 
                      measures = f1, par.set = svm.params, control = tune_ctrl)

svm.tune$x

svm.learner = setHyperPars(learner = svm.learner, par.vals = svm.tune$x)
svm.train = train(learner = svm.learner, task = train_task)
svm.train.class = predict(object = svm.train, train_task)
svm.val.class = predict(object = svm.train, val_task)
caret::confusionMatrix(as.factor(svm.val.class), val_data$target1)
getF1Stat(as.factor(svm.val.class), val_data$target1)




