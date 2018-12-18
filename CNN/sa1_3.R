remove(list=ls())

library(nnet)
library(MASS)

ir <- data.frame(rbind(iris3[,,1], iris3[,,2], iris3[,,3]), species=factor(c(rep("sv", 50), rep("c", 50), rep("sv", 50))))
stdsamp <- c(sample(1:50, 25), sample(51:100, 25), sample(101:150, 25))
tstsamp <- c(sample(1:50, 25), sample(51:100, 25), sample(101:150, 25))

mlpir <- nnet(species~., data=ir[stdsamp,], size=2, rang=0.5, decay=0, maxit=200)
table(ir$species[stdsamp], predict(mlpir, ir[stdsamp,], type="class"))
table(ir$species[tstsamp], predict(mlpir, ir[tstsamp,], type="class"))

