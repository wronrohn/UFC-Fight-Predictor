# Libraries
rm(list = ls())
dev.off()
library("ggplot2") # Visualization
library("caret") # Classification and regression training
library("dplyr") # Data manipulation
library("ggthemes") # Plot themes
library("corrplot") # Correlation visualizations
library("gbm") # ML
library("nnet") # ML 
library("randomForest") # ML
library(e1071)
library(class)
library(neuralnet)
library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)



# Data PrePRocessing
data.raw <- read.csv(file.choose())
summary(data.raw)

# Features Extracted
df.essen.raw <- data.raw[,c(1:3, 5:12, 37,38, 66:79,104,105, 133:145)]
str(df.essen.raw)

# Replacing NA Values with the mean value of the column

for(i in 1:ncol(df.essen.raw)){
  df.essen.raw[is.na(df.essen.raw[,i]), i] <- mean(df.essen.raw[,i], na.rm = TRUE)
}

# Removing Fighter Names along with Refree Names
df.rf <- df.essen.raw[,c(-1:-4)]

df.rf <- na.omit(df.rf)
# Random Forest

set.seed(123)
index <- sort(sample(nrow(df.rf), round (.25*nrow(df.rf))))
training <- df.rf[-index,]
test <- df.rf[index,]


fit<-randomForest(Winner ~., data = training,nodesize=20, importance=FALSE, mtry=5 ,ntree=500)
varImpPlot(fit)
Prediction <- predict(fit, test)
table(actual=test[,1], Prediction)


wrong<- (test[,1]!=Prediction)
Prediction_rate <- sum(wrong)/ length(test[,1])
accuracy_rf <- 1 - Prediction_rate


# KNN Data preperation

df.knn <- df.rf

df.knn$Winner <- as.numeric(df.knn$Winner)
df.knn$B_Stance <- as.numeric(df.knn$B_Stance)
df.knn$R_Stance <- as.numeric(df.knn$R_Stance)
df.knn$weight_class <- as.numeric(df.knn$weight_class)
df.knn$title_bout <- as.numeric(df.knn$title_bout)

med <- function(x) {
  z <- median(x, na.rm = TRUE)
  x[is.na(x)] <- z
  return(x)
}
df <- sapply(df.rf, function(x){
  if(is.numeric(x) & any(is.na(x))){
    med(x)
  } else {
    x
  }
}
)
df.knn <- as.data.frame(df)


#Converting response variable to factor
df.knn$Winner <- as.factor(df.knn$Winner)

normalize <- function(x){
  (x - min(x))/(max(x) - min(x))
}

df.knn <- na.omit(df.knn)
df.knn[] <- lapply(df.knn, function(x) as.numeric(x))

df_normalized <- as.data.frame(lapply(df.knn, normalize))
df_normalized$B_draw<- NULL
df_normalized$R_draw<- NULL
summary(df_normalized)

# Creating training and sample
set.seed(123) 

sample <- sample.int(n = nrow(df_normalized), size = floor(.75*nrow(df_normalized)), replace = F)
train <- df_normalized[sample, ]
train <- train[, -1]

test  <- df_normalized[-sample, ]
test <- test[,-1]
#class category
class_train <- df.rf[sample, 1]
class_train <- as.numeric(class_train)

class_test <- df.rf[-sample, 1]
class_test <- as.numeric(class_test)


summary(train)

# Function to calculate accuracy
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x))))}

## Applying KNN with Value of K as 10
predict_k10 <- knn(train, test, cl=class_train, k = 10)
table_k10 <- table(predict_k10, class_test)
table_k10
accuracy_k10 <- accuracy(table_k10)
accuracy_k10





## Applying KNN with Value of K as 15
predict_k15 <- knn(train, test, cl=class_train, k = 15)
table_k15 <- table(predict_k15, class_test)
table_k15
accuracy_k15 <- accuracy(table_k15)
accuracy_k15



## Applying KNN with Value of K as 20
predict_k20 <- knn(train, test, cl=class_train, k = 20)
table_k20 <- table(predict_k20, class_test)
table_k20
accuracy_k20 <- accuracy(table_k20)
accuracy_k20




# GBM Model Prep

df.gbm <- df.rf
sample <- sample.int(n = nrow(df.gbm), size = floor(.75*nrow(df.gbm)), replace = F)
train <- df.gbm[sample, ]
train <- train[, -1 ]
test  <- df.gbm[-sample, ]
test <- test[, -1 ]
#class category
class_train <- df.gbm[sample, 1]


class_test <- df.gbm[-sample, 1]
length(class_test)


## GBM

fitControl <- trainControl(method = "repeatedcv",
                           number = 5, # number of resampling iterations
                           repeats = 5) # the number of complete sets of folds to compute
summary(train)



gbmFit <- train(train, class_train,
                method = "gbm",
                trControl = fitControl,
                verbose = FALSE,
                )

gbmplot = plot(varImp(gbmFit, scale = T ),main="GBM")

grid.arrange(gbmplot ,  nrow =1)
gbmpred = predict(gbmFit, newdata=test)

accuracy_gbm <- confusionMatrix(gbmpred, class_test)
accuracy_gbm <- as.numeric(accuracy_gbm$overall[1])


# CART

df.cart <- df.rf


set.seed(123) 
sample <- sample.int(n = nrow(df.cart), size = floor(.75*nrow(df.cart)), replace = F)
train <- df.cart[sample, ]
test  <- df.cart[-sample, ]
str(df.cart)

summary(df.cart)



CART_class <- rpart(Winner ~., data = train)
rpart.plot(CART_class)
CART_predict <- predict(CART_class, test, type='class')
table(Actual = test[, 1], CART = CART_predict)
summary(CART_class)




CART_wrong <- sum(test[,1] != CART_predict)
CART_error_rate <- CART_wrong/ length(test[,1])
accuracy_cart <- 1-CART_error_rate
accuracy_cart












