remove(list = ls())

library(mxnet)

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
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(data, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)

softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

devices <- mx.cpu()
mx.set.seed(200)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y, initializer=mx.init.uniform(0.07), ctx=devices, num.round=10, array.batch.size=100, learning.rate=0.05, momentum=0.9, wd=0.00001, eval.metric=mx.metric.accuracy, epoch.end.callback=mx.callback.log.train.metric(100))

preds <- predict(model, test, ctx=devices)
pred.label <- max.col(t(preds)) - 1
sum(diag(table(test_org[, 1], pred.label))) / 1000

table(test_org[,1], pred.label)