remove(list = ls())

library(mxnet)

# Kaggle用テストデータ設定
test.Kaggle <- read.csv('data/test.csv', header=TRUE)
test.Kaggle <- data.matrix(test.Kaggle)
test.Kaggle <- t(test.Kaggle/255)
test.array.Kaggle <- test.Kaggle
dim(test.array.Kaggle) <- c(28, 28, 1, ncol(test.Kaggle))

train <- read.csv('./data/train.csv', header = TRUE)

train <- data.matrix(train)

train.x <- train[, -1]
train.y <- train[, 1]

train.x <- t(train.x/255)

data <- mx.symbol.Variable("data")

conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2,2), stride=c(2,2))
drop1 <- mx.symbol.Dropout(data=pool1, p=0.5)

conv2 <- mx.symbol.Convolution(data=drop1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2,2), stride=c(2,2))
drop2 <- mx.symbol.Dropout(data=pool2, p=0.5)

flatten <- mx.symbol.Flatten(data=drop2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=50)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
drop3 <- mx.symbol.Dropout(data=tanh3, p=0.5)

fc2 <- mx.symbol.FullyConnected(data=drop3, num_hidden=10)

lenet <- mx.symbol.SoftmaxOutput(data=fc2)

train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))

mx.set.seed(0)
devices = mx.cpu()
tic <- proc.time()

model.Kaggle <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y, ctx=devices, num.round=30, array.batch.size=100, learning.rate=0.05, momentum=0.9, wd=0.00001, eval.metric=mx.metric.accuracy, batch.end.callback=mx.callback.log.train.metric(100))

print(proc.time() - tic)

# 識別結果の作成
preds.Kaggle <- predict(model.Kaggle, test.array.Kaggle)
pred.label.Kaggle <- max.col(t(preds.Kaggle)) - 1
# 投稿用 csv ファイルの作成
submission <- data.frame(ImageId=1:ncol(test.Kaggle), Label=pred.label.Kaggle)
write.csv(submission, file='submission.csv', row.names=FALSE, quote=FALSE)