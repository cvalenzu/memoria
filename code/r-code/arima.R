data <- read.csv("../../data/lota_r_filtered.csv")
data_len <- length(data[,2])
train_len <- as.integer(0.8*data_len)

train <- data[1:train_len,2]
test_len <- data_len - train_len
test <- data[train_len+1:test_len,2]

model <- arima(train, order = c(3,1,3))

data <- train
params <- c(3,1,3)
y_test  <- c()
y_approx <- c()
for(i in 1:(test_len-12)){
    print(paste("Training ",i))
    model <-arima(data, order = params)
    p <- predict(model,n.ahead = 12)
    
    y_approx <- rbind(y_approx, p$pred)
    y_test <- rbind(y_test,test[seq(i+1,i+12)])
    data <- c(data,test[i])
}

write.csv(y_test,"y_test.csv", row.names= F)
write.csv(y_approx,"y_approx.csv", row.names=F)
