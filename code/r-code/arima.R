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
model <- Arima(data,order=params, seasonal=list(order=seasonal.params,period=12))
aic.model <- AIC(model)
residuals.model <- residuals(model)

#print("Forecasting testing set")
#for(i in 1:(test_len-12)){
#    print(paste("Training ",i))
#    p <- predict(model,n.ahead = 12)
#
#    y_approx <- rbind(y_approx, p$pred)
#    y_test <- rbind(y_test,test[seq(i+1,i+12)])
#    data <- c(data,test[i])
#    model <-Arima(data, model = model)
#}
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
#write.csv(y_test,paste(name,p,d,q,P,D,Q,"y_test.csv",sep="_"), row.names= F)
#write.csv(y_approx,paste(name,args[2],args[3], args[4],args[5],args[6],args[7],"y_approx.csv",sep="_"), row.names=F)
