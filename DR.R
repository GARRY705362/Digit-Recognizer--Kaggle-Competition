#RF
library(stats)
library(randomForest)
library(caret)

#reading the data
train<-read.csv("D:/Multivariate/Digit Recognizer/train.csv")
test<-read.csv("D:/Multivariate/Digit Recognizer/test.csv")

#converting labels to factors
train$label<-as.factor(train$label)

#Scaling the training and the testing data set
trainS<-scale(train[,c(2:785)])
testS<-scale(test)

#Applying PCA on both the data
train_pca<-prcomp(trainS,scale.=TRUE)
test_pca<-prcomp(testS,scale.=TRUE)
NewTrain <- data.frame(label=train$label,train_pca$x)
NewTest<-data.frame(test_pca$x)

#using the most prominent variables
NewTrain <- NewTrain[, c(1:41)]
NewTest<-NewTest[, c(1:41)] 

rf.model <- randomForest(label ~ .,data = NewTrain,method='class',mtry=9,ntree=1500)
rf.prediction <- predict(rf.model,NewTest)
Alpha<- data.frame(ImageId=seq(1,30000),label=rf.prediction)
write.csv(Alpha,'D:/Multivariate/Digit Recognizer/file.csv',row.names=FALSE)

#SVM
library(caret)
library(stats)
library(e1071)

train<-read.csv("D:/Multivariate/Digit Recognizer/train.csv")
test<-read.csv("D:/Multivariate/Digit Recognizer/test.csv")

#Converting the label column in training set to factors.
train$label<-as.factor(train$label)

#Scaling the training and the testing data set
TrainS<-scale(train[,c(2:785)])
TestS<-scale(test)


#applying PCA
trainPCA<-prcomp(TrainS,scale.=TRUE)
testPCA<-prcomp(TestS,scale.=TRUE)
Alpha <- data.frame(label=train$label,trainPCA$x)
Beta<-data.frame(testPCA$x)

#Taking the prominent variables
Alpha<- Alpha[, c(1:41)]
Beta<-Beta[, c(1:41)] 


svm_model <- svm(label~., data = Alpha, 
                 cost = 3, 
                 kernel = "linear") 
svm.prediction <- predict(svm_model,Beta)

#Submit the values to the output file
Gamma<- data.frame(ImageId=seq(1,30000),label=svm.prediction)
write.csv(Gamma,'D:/Multivariate/Digit Recognizer/file1.csv',row.names=FALSE)



#H2O
library(readr)
library(h2o)


train<-read_csv("D:/Multivariate/Digit Recognizer/train.csv")
test<-read_csv("D:/Multivariate/Digit Recognizer/test.csv")

#Proving the size
Alpha = h2o.init(max_mem_size = '10g', nthreads = -1) 

#converting to factors
train$label<-as.factor(train$label)
trainModified = as.h2o(train)
testModified = as.h2o(test)

#running deeplearning with 1000 epochs and 2 layers
Alpha =  h2o.deeplearning(x = 2:785,  y = 1,  training_frame = trainModified,
                          activation = "RectifierWithDropout", input_dropout_ratio = 0.2,
                          hidden_dropout_ratios = c(0.5,0.5), balance_classes = TRUE,
                          hidden = c(2000,2000),  momentum_stable = 0.99,
                          nesterov_accelerated_gradient = T, epochs = 1000) 
h2o.confusionMatrix(Alpha)

#predicting
testPredict <- h2o.predict(Alpha, testModified)
testFrame = as.Alpha.frame(testPredict)
testFrame = Alpha.frame(ImageId = seq(1,length(testFrame$predict)), Label = testFrame$predict)
write.csv(testFrame, file = "D:/Multivariate/Digit Recognizer/submissionh2o.csv", row.names=F)



