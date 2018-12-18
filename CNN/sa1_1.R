rm(list=ls())
source("http://hosho.ees.hokudai.ac.jp/~kubo/log/2007/img07/plot.nn.txt")
library(nnet)
library(MASS)

x1 <- c(0, 0)
x2 <- c(0, 1)
x3 <- c(1, 0)
x4 <- c(1, 1)

x <- rbind(x1, x2, x3, x4)
y <- c("-1", "1", "1", "-1")
xor <- data.frame(x, classes=y)

mlp <- nnet(classes~., data=xor, size=5, rang=0.5, decay=0, maxit=100)

predict(mlp, xor, type="class")

plot.nn(mlp)