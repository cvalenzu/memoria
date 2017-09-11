library(forecast)

args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args) < 4) {
  print(args)
  stop("Usage: arima.R inputData.csv p q d", call.=FALSE)
}

data <- read.csv(args[1])
data_len <- length(data[,2])
train_len <- as.integer(0.8*data_len)

train <- data[1:train_len,2]
test_len <- data_len - train_len
test <- data[train_len+1:test_len,2]

data <- train
params <- c(as.numeric(args[2]),as.numeric(args[3]),as.numeric(args[4]))
y_test  <- c()
y_approx <- c()
model <- Arima(data,order=params)
for(i in 1:(test_len-12)){
    print(paste("Training ",i))
    p <- predict(model,n.ahead = 12)

    y_approx <- rbind(y_approx, p$pred)
    y_test <- rbind(y_test,test[seq(i+1,i+12)])
    data <- c(data,test[i])
    model <-Arima(data, model = model)
}
name <- strsplit(basename(args[1]),".csv")

write.csv(y_test,paste(name,"y_test.csv",sep="_"), row.names= F)
write.csv(y_approx,paste(name,"y_approx.csv",sep="_"), row.names=F)
