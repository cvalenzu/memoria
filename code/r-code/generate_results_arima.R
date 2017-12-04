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


data <- train
params <- c(as.numeric(args[2]),as.numeric(args[3]),as.numeric(args[4]))
seasonal.params <- c(as.numeric(args[5]),as.numeric(args[6]),as.numeric(args[7]))
print(paste("Training Model with ","p:",as.numeric(args[2])," q:",as.numeric(args[3])," d:",as.numeric(args[4]), " P:",as.numeric(args[5])," Q:",as.numeric(args[6])," D:",as.numeric(args[7])))

t0 <- proc.time()
model <- Arima(data,order=params, seasonal=list(order=seasonal.params,period=12))
time <- proc.time() - t0
aic.model <- AIC(model)
residuals.model <- residuals(model)

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
    model <-Arima(data, model = model)
#    break
}

name <- strsplit(basename(args[1]),".csv")
p <- args[2]
d <- args[3]
q <- args[4]
P <- args[5]
D <- args[6]
Q <- args[7]
write.csv(y_test,paste(name,p,d,q,P,D,Q,"y_test.csv",sep="_"), row.names= F)
write.csv(y_approx,paste(name,args[2],args[3], args[4],args[5],args[6],args[7],"y_approx.csv",sep="_"), row.names=F)

write.csv(y_approx_25_lo,paste(name,args[2],args[3], args[4],args[5],args[6],args[7],"y_approx_25_lo.csv",sep="_"), row.names=F)
write.csv(y_approx_25_hi,paste(name,args[2],args[3], args[4],args[5],args[6],args[7],"y_approx_25_hi.csv",sep="_"), row.names=F)
write.csv(y_approx_50_lo,paste(name,args[2],args[3], args[4],args[5],args[6],args[7],"y_approx_50_lo.csv",sep="_"), row.names=F)
write.csv(y_approx_50_hi,paste(name,args[2],args[3], args[4],args[5],args[6],args[7],"y_approx_50_hi.csv",sep="_"), row.names=F)
write.csv(y_approx_75_lo,paste(name,args[2],args[3], args[4],args[5],args[6],args[7],"y_approx_75_lo.csv",sep="_"), row.names=F)
write.csv(y_approx_75_hi,paste(name,args[2],args[3], args[4],args[5],args[6],args[7],"y_approx_75_hi.csv",sep="_"), row.names=F)
