library(quantmod)
library(tseries)
library(xts)
library(zoo)
library(PerformanceAnalytics)
library(knitr)
library(dplyr)

options(scipen = 999)

stock1 <- "KO"
stock2 <- "PEP"

##get data from Yahoo Finance for stock1
getSymbols(stock1, src = "yahoo")

##get data from Yahoo Finance for stock1
getSymbols(stock2, src = "yahoo")

stock1 <- KO
stock2 <- PEP

kable(head(stock1))
kable(head(stock2))

##keeping only Adjusted close data (which accounts for splits/dividends)

stock1 <- stock1[, grep("Adjusted", colnames(stock1))]
stock2 <- stock2[, grep("Adjusted", colnames(stock2))]

cut_off_date <- as.Date("2015-01-01")
stock1 <- stock1[index(stock1) >= cut_off_date]
stock2 <- stock2[index(stock2) >= cut_off_date]

stock1[is.na(stock1),]
stock2[is.na(stock2),]

##back filling missing data
stock1 <- na.locf(stock1)
stock2 <- na.locf(stock2)

##calculate daily returns
ret_stock1 <- Delt(stock1)
ret_stock2 <- Delt(stock2)

##combining two time series into one dataframe
data <- data.frame(matrix(NA, dim(ret_stock1)[1],2))
data[, 1] <- ret_stock1
data[, 2] <- ret_stock2
data <- xts(data, index(ret_stock1))
head(data)

##getting rolling correlation across a certain period
correlation <- function(x) {
  result <- cor(x[, 1], x[, 2])
  return(result)
}
corr <- rollapply(data, 100, correlation, by.column = FALSE)
plot(corr)
mtext(paste0('Mean Corr: ', mean(corr, na.rm=TRUE) %>% format.default(digits=3)))

##calculate the hedge ratio
hedge_ratio <- stock1/stock2

##generate trading signals
n_period <- 14
roll_mean <- rollapply(hedge_ratio, n_period, mean)
roll_std <- rollapply(hedge_ratio, n_period, sd)

n <- 1
roll_ub <- roll_mean + roll_std*n
roll_lb <- roll_mean - roll_std*n

plot(cbind(hedge_ratio, roll_ub, roll_lb))

##define trading signals
signal <- NULL
signal <- ifelse(
  hedge_ratio > roll_ub, -1, ifelse(
    hedge_ratio < roll_lb, 1, 0
  )
)

##we can only trade on the next trading day after a signal is generated
##hence we lag our trading signal by 1 day
signal <- lag(signal, k=-1)

##calculate returns generated
spread_return <- ret_stock1 - ret_stock2*hedge_ratio
trade_ratrun <- spread_return*signal

##analyze the return performances and print out key stats
charts.PerformanceSummary(trade_ratrun)

print(paste0("Cumulative Returns -- ", Return.cumulative(trade_ratrun)))
print(paste0("Annualized Returns -- ", Return.annualized(trade_ratrun)))
print(paste0("Maximum Drawdown -- ", maxDrawdown(trade_ratrun)))
print(paste0("Sharpe Ratio -- ", SharpeRatio(as.ts(trade_ratrun), Rf = 0, p = 0.95, FUN = "StdDev")))


# improved strategy
stock1 <- KO[, grep("Adjusted", colnames(KO))]
stock2 <- PEP[, grep("Adjusted", colnames(PEP))]

window <- 14 # always use 100 observations prior as an analysis period

cut_off_date <- as.Date("2015-01-01")
ix <- which.min(ifelse(
  (index(stock1)-cut_off_date)<0,999,
   index(stock1)-cut_off_date))[1]-1
z_thresh <- 1.5

spreads <- vector()
num_signals <- nrow(stock1)-ix
signals <- xts(matrix(rep(NA,num_signals*2),ncol=2),
               order.by=index(stock1[(ix+1):nrow(stock1)]))
colnames(signals) <- c("slope", "signal")

for (i in (ix-window+1):(nrow(stock1)-1)) {
  y <- stock1[(i-window+1):i]
  x <- stock2[(i-window+1):i]
  
  # run linear regression on lookback window
  # parameters are refreshed every 100 days
  days_no_regress <- window
  if (days_no_regress==window) {
    model <- lm(y ~ x)
    slope     <- model$coefficients[[2]]
    intercept <- model$coefficients[[1]]
    days_no_regress = 0
  }
  
  # derive z-score of the price spread
  spread <- stock1[[i]]-(slope*stock2[[i]])+intercept
  spreads <- c(spreads, spread)
  wind_spreads <- tail(spreads, window)
  
  # decide to trade only if 100 historical spreads available
  if (length(wind_spreads) == window) {
    z_score <- (spread-mean(wind_spreads)) / sd(wind_spreads)
    
    # compute trade signal (if z-score exceeds threshold)
    trade_signal <- ifelse(
      z_score > z_thresh, -1, ifelse(
        z_score < -z_thresh, 1, 0
      )
    )
    
    date_trade <- index(stock1[i+1])
    signals[date_trade,] <- cbind(slope, trade_signal)
  }
  
  days_no_regress <- days_no_regress+1
}

trade_ratrun <- signals$signal*(ret_stock1-ret_stock2)
charts.PerformanceSummary(trade_ratrun)

print(paste0("Cumulative Returns -- ", Return.cumulative(trade_ratrun)))
print(paste0("Annualized Returns -- ", Return.annualized(trade_ratrun)))
print(paste0("Maximum Drawdown -- ", maxDrawdown(trade_ratrun)))
print(paste0("Sharpe Ratio -- ", SharpeRatio(as.ts(trade_ratrun), Rf = 0, p = 0.95, FUN = "StdDev")))





# Matt
cut_off_date <- as.Date("2014-01-01")

ret_stock1 <- KO [, grep("Adjusted", colnames(stock1))] %>% Delt()
ret_stock2 <- PEP[, grep("Adjusted", colnames(stock2))] %>% Delt()

# filter to later than 1st Jan 2014
ret_stock1 <- ret_stock1[index(ret_stock1) >= cut_off_date]
ret_stock2 <- ret_stock2[index(ret_stock2) >= cut_off_date]

window <- 14
sd_thresh <- 1

num_signals <- nrow(ret_stock1)-window
signals <- xts(matrix(rep(NA,num_signals*2), ncol=2),
               order.by=index(ret_stock1[(window+1):nrow(ret_stock1)]))
colnames(signals) <- c("hedge_ratio", "signal")

# use a rolling window to generate trading signal
for (i in 1:(nrow(ret_stock1)-window)) {
  y <- ret_stock1[i:(i+window-1)]
  x <- ret_stock2[i:(i+window-1)]
  
  model <- lm(y ~ x)
  
  errors <- resid(model)
  error_z <- (errors[[length(errors)]]-mean(errors))/sd(errors)

  # signal is defined as the latest error deviating materially from the avg error
  hedge_ratio <- model$coefficients[[2]]
  if (abs(error_z) < .5) {
    signal <- 0
  } else if (error_z > sd_thresh) {
    signal <- -1
  } else if (error_z < sd_thresh) {
    signal <- 1
  }
  
  signals[i,] <- cbind(hedge_ratio, signal)
}

trade_ret_stock1 <- ret_stock1[(window+1):nrow(ret_stock1)]
trade_ret_stock2 <- ret_stock2[(window+1):nrow(ret_stock2)]
trade_ratrun <- signals$signal*(trade_ret_stock1-signals$hedge_ratio*trade_ret_stock2)

charts.PerformanceSummary(trade_ratrun)
print(paste0("Cumulative Returns -- ", Return.cumulative(trade_ratrun)))
print(paste0("Annualized Returns -- ", Return.annualized(trade_ratrun)))
print(paste0("Maximum Drawdown -- ", maxDrawdown(trade_ratrun)))
print(paste0("Sharpe Ratio -- ", SharpeRatio(as.ts(trade_ratrun), Rf = 0, p = 0.95, FUN = "StdDev")))





# Long's strategy
cut_off_date <- as.Date("2015-01-01") 
start_date <- as.Date("2014-01-01")

stock1_in <- KO [index(KO)  < cut_off_date & index(KO)  >= start_date]
stock2_in <- PEP[index(PEP) < cut_off_date & index(PEP) >= start_date]

ret_stock1 <- ret_stock1[index(ret_stock1) >= cut_off_date]
ret_stock2 <- ret_stock2[index(ret_stock2) >= cut_off_date]

## Linear regression to calculate the intercept
lm <- lm(stock1_in ~ stock2_in)
intercept <- lm$coefficients[1]

##calculate the new hedge ratio
hedge_ratio <- (stock1-intercept)/stock2

##generate trading signals
n_period <- 14
roll_mean <- rollapply(hedge_ratio, n_period, mean)
roll_std <- rollapply(hedge_ratio, n_period, sd)

roll_ub <- roll_mean + roll_std
roll_lb <- roll_mean - roll_std

##define trading signals
signal <- NULL
signal <- ifelse(
  hedge_ratio > roll_ub, -1, ifelse(
    hedge_ratio < roll_lb, 1, 0
  )
)

##we can only trade on the next trading day after a signal is generated
##hence we lag our trading signal by 1 day
signal <- lag(signal)

##calculate returns generated
spread_return <- ret_stock1 - ret_stock2*hedge_ratio
trade_ratrun <- spread_return*signal

##analyze the return performances and print out key stats
charts.PerformanceSummary(trade_ratrun)

print(paste0("Cumulative Returns -- ", Return.cumulative(trade_ratrun)))
print(paste0("Annualized Returns -- ", Return.annualized(trade_ratrun)))
print(paste0("Maximum Drawdown -- ", maxDrawdown(trade_ratrun)))
print(paste0("Sharpe Ratio -- ", SharpeRatio(as.ts(trade_ratrun), Rf = 0, p = 0.95, FUN = "StdDev")))

