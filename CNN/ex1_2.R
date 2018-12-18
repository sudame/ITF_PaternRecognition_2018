remove(list=ls())

library(nnet)
library(MASS)

# source("http://hosho.ees.hokudai.ac.jp/~kubo/log/2007/img07/plot.nn.txt")

ir <- data.frame(rbind(iris3[,,1], iris3[,,2], iris3[,,3]), species=factor(c(rep("sv", 50), rep("c", 50), rep("sv", 50))))
stdsamp <- c(sample(1:50, 25), sample(51:100, 25), sample(101:150, 25))
tstsamp <- c(sample(1:50, 25), sample(51:100, 25), sample(101:150, 25))

hidden = c(1:10)
iter = c(1:10)
stdtrave = rep(0, length(hidden))
tsttrave = rep(0, length(hidden))
stdtr = matrix(0, length(iter), length(hidden))
tsttr = matrix(0, length(iter), length(hidden))

wtss <- as.list(NULL)

for(i in 1: length(hidden)){
  for(j in 1:length(iter)){
    mlpir <- nnet(species~., data=ir[stdsamp,], size=hidden[i], rang=0.5, decay=0.00, maxit=200)
    stdout = predict(mlpir, ir[stdsamp,], type="class")
    tstout = predict(mlpir, ir[-stdsamp,], type="class")
    stdtr[j, i] = mean(stdout != ir[stdsamp,]$species)
    tsttr[j, i] = mean(tstout != ir[-stdsamp,]$species)
    if(i > length(wtss)){
      wtss[i] <- mlpir$wts
    } else {
      wtss[[i]] <- c(mlpir$wts, wtss[[i]])
    }
  }
  stdtrave[i] = mean(stdtr[, i])
  tsttrave[i] = mean(tsttr[, i])
}

# 再代入誤り識別率のグラフの表示
plot(hidden, stdtrave, ylim = c(0, 0.4), type = "l", main = "再代入誤り識別率の遷移", xlab = "隠れ素子の数", ylab = "誤識別率")

# 汎化誤差率のグラフの表示
plot(hidden, tsttrave, ylim = c(0, 0.4), type = "l", main = "汎化誤差の遷移", xlab = "隠れ素子の数", ylab = "誤識別率")


# table(ir$species[stdsamp], predict(mlpir, ir[stdsamp,], type="class"))
# table(ir$species[tstsamp], predict(mlpir, ir[tstsamp,], type="class"))

stdminidx <- match(min(stdtrave), stdtrave)[1]
tstminidx <- match(min(tsttrave), tsttrave)[1]

hist(wtss[[stdminidx]], freq=TRUE, main = "再代入誤り率最小場合の結合係数", xlab = "重み", ylab = "頻度")
hist(wtss[[tstminidx]], freq=TRUE, main = "汎化誤差最小場合の結合係数", xlab = "重み", ylab = "頻度")