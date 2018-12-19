rm(list=ls())

library(nnet)
library(MASS)

x1 <- c(0, 0)
x2 <- c(0, 1)
x3 <- c(1, 0)
x4 <- c(1, 1)

x <- rbind(x1, x2, x3, x4)
y <- c("-1", "1", "1", "-1")
xor <- data.frame(x, classes=y)

hidden = c(1:10)
iter = c(1:10)
trave = rep(0, length(hidden))
tr = matrix(0, length(iter), length(hidden))

wtss <- as.list(NULL)

for(i in 1: length(hidden)){
  for(j in 1:length(iter)){
    res = nnet(classes~., data=xor, size=hidden[i], range=0.1)
    out = predict(res, xor, type="class")
    tr[j, i] = mean(out != xor$classes)
    
    if(i > length(wtss)){
      wtss[i] <- res$wts
    } else {
      wtss[[i]] <- c(res$wts, wtss[[i]])
    }
  }
  trave[i] = mean(tr[, i])
  if(i == 1) {
    hist(res$wts,breaks=seq(-100, 100, 5), freq=TRUE, main = "隠れ素子数1の場合の結合係数", xlab = "重み", ylab = "頻度")
  }
}

# 誤識別率のグラフの表示
plot(iter, trave, type = "l", main = "誤識別率の遷移", xlab = "隠れ素子の数", ylab = "誤識別率")

hist(wtss[[length(hidden)]],breaks=seq(-100, 100, 5),  freq=TRUE, main = "隠れ素子数10の場合の結合係数", xlab = "重み", ylab = "頻度")