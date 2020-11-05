library(tidyverse)
library(forecast)
library(urca)
library(ggfortify)
library(grid)
library(gridExtra)

correl <- function(y){
  p1 <- ggAcf(y,36) + scale_y_continuous(limits=c(-1,1)) + 
    theme(axis.title.x = element_blank(), plot.title = element_blank())
  p2 <- ggPacf(y, 36) + scale_y_continuous(limits=c(-1,1)) +
    theme(axis.title.x = element_blank(), plot.title = element_blank())
  grid.arrange(p1,p2,ncol=1)
}

load_dx <- function(filename, start,end=NA) {
  df <- read_csv(filename,
                 col_names=c("Date","Exports"),
                 col_types=list(col_date("%b-%y"), col_double()))
  if (is.na(end))
    v = ts(df$Exports,start=start,frequency=12)
  else
    v = ts(df$Exports,start=start,end=end,frequency=12)
}

do_plot <- function(timeSeries) {
  autoplot(timeSeries) +
    scale_color_manual(values=rep("black", 4)) +
    ylab("") + xlab("") +
    aes(linetype=series) +
    scale_linetype_manual(values=c("solid", "dashed", rep("dotted",2))) +
    theme(legend.position="none")
}

filename <- "C:/Users/Matthew/Desktop/school/econ233 project/nodx_2017.csv"
filename2018 <- "C:/Users/Matthew/Desktop/school/econ233 project/dx_2018.csv"
y <- load_dx(filename,start=c(1976,1))
dx_2018 <- load_dx(filename2018,start=c(2018,1))
tail(y,6)

autoplot(y) + theme(aspect.ratio=1/2)
correl(y)

uy <- ur.df(y, type="none", selectlags="BIC")
summary(uy)
uy <- ur.kpss(y, type="tau", lags="short")
summary(uy)

# Seasonal plot & monthly subseries plot
ggseasonplot(y)
ggsubseriesplot(y)

# Run diagnostics on differenced log data
dlog_y = diff(log(y))
autoplot(dlog_y) + theme(aspect.ratio=1/2)
correl(dlog_y)      # Resembles an MA(1) with SAR(1)

# our selected model: SARIMA(0,1,1) x (1,0,0)
model.fit <- function(data) {
  fit <- Arima(data, order=c(0,1,1),
               seasonal=list(order=c(0,0,1),period=12),
               include.constant = T, #can include seasonal dummy here
               lambda=0) # box-cox transformation, lambda=0 is log-transformation
}

# run model fit on all training data
fit <- model.fit(y)
summary(fit)
autoplot(fit)
checkresiduals(fit)
correl(residuals(fit))

# Rolling 1-step forecast
fcstStart=c(2013,1)
numForecasts <- length(window(y, start=fcstStart))
windowSize <- length(window(y))-numForecasts
fcst1step<-ts(matrix(rep(NA,numForecasts*3),ncol=3),
              start=fcstStart, frequency=12) # to store forecasts
colnames(fcst1step) <- c("mean", "lower", "upper")
for (i in 1:numForecasts){
  fit1step <- model.fit(y[1:(windowSize+i-1)])
  temp <- forecast(fit1step, h=1)
  fcst1step[i,]<-cbind(temp$mean, temp$lower[,"95%"], temp$upper[,"95%"])
}
ts.joint = ts.union(Actual=window(y,start=fcstStart),
                    Fcst=fcst1step)
autoplot(ts.joint) +
  scale_color_manual(values=rep("black", 4)) +
  ylab("") + xlab("") +
  aes(linetype=series) +
  scale_linetype_manual(values=c("solid", "dashed", rep("dotted",2))) +
  theme(legend.position="none")

# Run OOS fit diagnostics
sse <- sum((ts.joint[,"Actual"]-ts.joint[,"Fcst.mean"])^2)
sst <- sum((ts.joint[,"Actual"]-mean(ts.joint[,"Actual"]))^2)
OOSR2 <- 1-sse/sst
print(paste0("Out-of-sample RMSE is ",as.character(round(sqrt(sse/numForecasts),2))))
print(paste0("Out-of-sample R-sqr is ",as.character(round(OOSR2,2))))

# use data only until 2017 to forecast 2018
model <- model.fit(y)
model.pred <- forecast(model, h=12)
oosFcst <- cbind(model.pred$mean, model.pred$lower[,"95%"], model.pred$upper[,"95%"])
dx_2018 <- load_dx(filename2018,start=c(2018,1))

# visualize only forecast
pred.joint = ts.union(Actual=dx_2018,
                      Fcst=oosFcst)
sse <- sum((pred.joint[,"Actual"]-pred.joint[,"Fcst.model.pred$mean"])^2)
sst <- sum((pred.joint[,"Actual"]-mean(pred.joint[,"Actual"]))^2)
OOSR2 <- 1-sse/sst
print(paste0("Out-of-sample RMSE is ",as.character(round(sqrt(sse/numForecasts),2))))
print(paste0("Out-of-sample R-sqr is ",as.character(round(OOSR2,2))))

autoplot(pred.joint) +
  scale_color_manual(values=rep("black", 4)) +
  ylab("") + xlab("") +
  aes(linetype=series) +
  scale_linetype_manual(values=c("solid", "dashed", rep("dotted",2))) +
  theme(legend.position="none")

# visualize all data
all = ts(c(y,dx_2018),start=c(1976,1),frequency=12)
pred.joint = ts.union(Actual=all,
                      Fcst=oosFcst)
colnames(pred.joint) = c("Actual","Forecast","Lower","Upper")
autoplot(pred.joint) +
  scale_color_manual(values=rep("black", 4)) +
  ylab("") + xlab("") +
  aes(linetype=series) +
  scale_linetype_manual(values=c("solid", "dashed", rep("dotted",2))) +
  theme(legend.position="none")

# NOTE: THIS IS FOR PREDICTION!! #
# Rolling 1-step forecast
dataStart=c(2013,1)
fcstStart=c(2018,1)
dataEnd=c(2018,12)
all = window(ts(c(y,dx_2018),start=c(1976,1),frequency=12),
             start=dataStart,
             end=dataEnd)
numForecasts <- length(window(all, start=fcstStart))
windowSize <- length(window(all))-numForecasts
fcst1step<-ts(matrix(rep(NA,numForecasts*3),ncol=3),
              start=fcstStart, frequency=12) # to store forecasts
colnames(fcst1step) <- c("mean", "lower", "upper")
for (i in 1:numForecasts){
  fit1step <- model.fit(all[1:(windowSize+i-1)])
  temp <- forecast(fit1step, h=1)
  fcst1step[i,]<-cbind(temp$mean,
                       temp$lower[,"95%"],
                       temp$upper[,"95%"])
}
ts.joint = ts.union(Actual=window(all,start=fcstStart),
                    Fcst=fcst1step)
do_plot(ts.joint)

# Run OOS fit diagnostics
sse <- sum((ts.joint[,"Actual"]-ts.joint[,"Fcst.mean"])^2)
sst <- sum((ts.joint[,"Actual"]-mean(ts.joint[,"Actual"]))^2)
OOSR2 <- 1-sse/sst
print(paste0("Out-of-sample RMSE is ",as.character(round(sqrt(sse/numForecasts),2))))
print(paste0("Out-of-sample R-sqr is ",as.character(round(OOSR2,2))))

# baseline model
naiveFcst = naive(y,h=12)
naive = cbind(dx_2018,
              naiveFcst$mean,
              naiveFcst$lower[,"95%"],
              naiveFcst$upper[,"95%"])
colnames(naive)=c("pred","lower","upper","actual")
sse <- sum((naive[,"actual"]-naive[,"pred"])^2)
sst <- sum((naive[,"actual"]-mean(naive[,"actual"]))^2)

OOSR2 <- 1-sse/sst
print(paste0("Out-of-sample RMSE is ",as.character(round(sqrt(sse/12),2))))
print(paste0("Out-of-sample R-sqr is ",as.character(round(OOSR2,2))))

do_plot(naive)