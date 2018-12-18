remove(list = ls())

train <- read.csv('./data/short_prac_train.csv', header = TRUE)
test <- read.csv('data/short_prac_test.csv', header = TRUE)

train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[, -1]
train.y <- train[, 1]
test_org <- test
test <- test[, -1]

train.x <- t(train.x/255)
test <- t(test/255)

data <- mx.symbol.Variable("data")

conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=c(2,2), stride=c(2,2))
# drop1 <- mx.symbol.Dropout(data=pool1, p=0.5)

conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=c(2,2), stride=c(2,2))
# drop2 <- mx.symbol.Dropout(data=pool2, p=0.5)

flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=50)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="relu")
# drop3 <- mx.symbol.Dropout(data=tanh3, p=0.5)

fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)

lenet <- mx.symbol.SoftmaxOutput(data=fc2)

train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))

test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))

mx.set.seed(0)
devices = mx.cpu()
tic <- proc.time()

model.CNNtanhDrop <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y, ctx=devices, num.round=30, array.batch.size=100, learning.rate=0.05, momentum=0.9, wd=0.00001, eval.metric=mx.metric.accuracy, batch.end.callback=mx.callback.log.train.metric(100))

print(proc.time() - tic)

# テストデータによる正答率の出力
preds <- predict(model.CNNtanhDrop, test.array, ctx=devices)
pred.label <- max.col(t(preds)) - 1
sum(diag(table(test_org[, 1], pred.label))) / 1000