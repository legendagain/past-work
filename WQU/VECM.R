library(Quandl)
library(dplyr)
library(tsDyn)
library(urca)
library(vars)
library(tseries)

# retrieve data from Quandl
quandl_codes = c("FRED/EXJPUS",                                     # USD/JPY monthly rate
                 "RATEINF/CPI_JPN", "RATEINF/CPI_USA",              # JP and US CPI
                 "MOFJ/INTEREST_RATE_JAPAN_5Y", "FRED/GS5")          # JGB and UST 5Y yield

quandl_data <- Quandl(quandl_codes, start_date="1999-12-31", type="xts")
quandl_data[,2:3] <- lag(quandl_data[,2:3])

# format data and take first differences
data <- apply.monthly(quandl_data, colMeans, na.rm=TRUE) %>%
          na.omit()
names(data) <- c("usdjpy", "jp_cpi", "us_cpi", "jp_ir", "us_ir")
plot(data, multi.panel=TRUE, yaxis.same=FALSE)
sapply(data, adf.test, k=12)   # here, ADF test shows all have unit root

# take first differences
data.d <- data %>% diff() %>% na.omit()
cor(data.d)

# check for number of lags using VAR
VARselect(data.d, lag.max=12)

# Johansen Test for # of cointegrating relationships
jotest1=ca.jo(data, type="eigen", K=2, ecdet="const", spec="longrun")
summary(jotest1)
jotest2=ca.jo(data, type="trace", K=2, ecdet="const", spec="longrun")
summary(jotest2)

VECM_fit = VECM(data, 2, r=1, include="const", estim="ML", LRinclude="none")
summary(VECM_fit)

# plot fitted vs. actuals
fitted_vals <- fitted(VECM_fit, level="original")$usdjpy
values <- cbind(data$usdjpy, fitted_vals$usdjpy)
plot(values)

fitted_diffs <- fitted(VECM_fit)[,"usdjpy"] %>% as.xts()
index(fitted_diffs) <- index(data.d)[-1:-2]
diff_values <- merge.xts(data.d$usdjpy, fitted_diffs) %>% na.omit()
plot(diff_values)