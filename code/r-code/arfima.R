library(forecast)

args = commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args) < 1) {
  print(args)
  stop("Usage: arfima.R inputData.csv", call.=FALSE)
}

print(paste("Reading data from", args[1]))
data <- read.csv(args[1])
data_len <- length(data[,2])
train_len <- as.integer(0.8*data_len)

train <- ts(data[1:train_len,2], frequency=12)
test_len <- data_len - train_len
test <- ts(data[train_len+1:test_len,2],frequency=12)


print("Training ARFIMA model")
data <- train
#Training ARFIMA model automatically
t0 <- proc.time()
model <- arfima(data,seasonal=T,approximation=F, stepwise=F, parallel=T, num.cores=NULL)
time <- proc.time() - t0
aic.model <- AIC(model)
residuals.model <- residuals(model)

name <- strsplit(basename(args[1]),".csv")
name <- paste(name, "arfima", sep="_")

print("Saving model")
save(model, file=paste(name,"model.Rdata", sep="_"))

print("Forecasting")
#Forecasting on test set and getting 
# confidence intervals
y_test  <- c()
y_approx <- c()
y_approx_25_lo <- c()
y_approx_25_hi <- c()
y_approx_50_lo <- c()
y_approx_50_hi <- c()
y_approx_75_lo <- c()
y_approx_75_hi <- c()

print("Forecasting testing set")
for(i in 1:(test_len-12)){
    print(paste("Training ",i))
    p <- forecast(model,h = 12, level=c(.25,.50,.75))

    y_approx <- rbind(y_approx, p$mean)

    y_approx_25_lo <- rbind(y_approx_25_lo,p$lower[,1])
    y_approx_25_hi <- rbind(y_approx_25_hi,p$upper[,1])
    y_approx_50_lo <- rbind(y_approx_50_lo,p$lower[,2])
    y_approx_50_hi <- rbind(y_approx_50_hi,p$upper[,2])
    y_approx_75_lo <- rbind(y_approx_75_lo,p$lower[,3])
    y_approx_75_hi <- rbind(y_approx_75_hi,p$upper[,3])

    y_test <- rbind(y_test,test[seq(i+1,i+12)])

    data <- c(data,test[i])
    model <-arfima(data, model = model)
}

print("Saving Results")
#Saving data
write.csv(y_test,paste(name,"y_test.csv",sep="_"), row.names= F)
write.csv(y_approx,paste(name,"y_approx.csv",sep="_"), row.names=F)

write.csv(y_approx_25_lo,paste(name,"y_approx_25_lo.csv",sep="_"), row.names=F)
write.csv(y_approx_25_hi,paste(name,"y_approx_25_hi.csv",sep="_"), row.names=F)
write.csv(y_approx_50_lo,paste(name,"y_approx_50_lo.csv",sep="_"), row.names=F)
write.csv(y_approx_50_hi,paste(name,"y_approx_50_hi.csv",sep="_"), row.names=F)
write.csv(y_approx_75_lo,paste(name,"y_approx_75_lo.csv",sep="_"), row.names=F)
write.csv(y_approx_75_hi,paste(name,"y_approx_75_hi.csv",sep="_"), row.names=F)
