library(forecast)

args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args) < 7) {
  print(args)
  stop("Usage: arima.R inputData.csv p d q P D Q", call.=FALSE)
}

data <- read.csv(args[1])
data_len <- length(data[,2])
train_len <- as.integer(0.8*data_len)

train <- data[1:train_len,2]
test_len <- data_len - train_len
test <- data[train_len+1:test_len,2]

print(length(train))
print(length(test))

data <- train
params <- c(as.numeric(args[2]),as.numeric(args[3]),as.numeric(args[4]))
seasonal.params <- c(as.numeric(args[5]),as.numeric(args[6]),as.numeric(args[7]))
y_test  <- c()
y_approx <- c()
print(paste("Training Model with ","p:",as.numeric(args[2])," q:",as.numeric(args[3])," d:",as.numeric(args[4]), " P:",as.numeric(args[5])," Q:",as.numeric(args[6])," D:",as.numeric(args[7])))

t0 <- proc.time()
model <- Arima(data,order=params, seasonal=list(order=seasonal.params,period=12))
time <- proc.time() - t0
aic.model <- AIC(model)
residuals.model <- residuals(model)

print("Saving Data")
name <- strsplit(basename(args[1]),".csv")
p <- args[2]
d <- args[3]
q <- args[4]
P <- args[5]
D <- args[6]
Q <- args[7]
write.csv(residuals.model,  paste(name,p,d,q,P,D,Q,"residuals.csv", sep="_"))
write.csv(c(model$aic,model$aicc,model$bic), paste(name,p,d,q,P,D,Q, "aic.csv", sep="_"))
write.csv(time[3], paste(name,p,d,q,P,D,Q, "time.csv", sep="_"))
