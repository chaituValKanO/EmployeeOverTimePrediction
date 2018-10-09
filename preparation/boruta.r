setwd("C:\\Users\\chait\\Documents\\Insofe\\MiTh\\data")
boruta_data = read.csv(file = "train.csv", header = T)

boruta_data$ID = NULL
boruta_data$Attr37 = NULL

set.seed(143)
train_rows = createDataPartition(y = boruta_data$target, p = 0.7, list = F)
boruta_train = boruta_data[train_rows, ]
boruta_val = boruta_data[-train_rows, ]

boruta_train = centralImputation(data = boruta_train)
sum(is.na(boruta_train))

library(DMwR)
library(Boruta)

boruta.train = Boruta(target~., data = boruta_train, doTrace = 2)
print(boruta.train)

findCorrelation(x = cor_mat, cutoff = 0.9, verbose = T, exact = T, names = T)
