library(zoo)
library(xts)
library(TTR)
library(fracdiff)
library(quadprog)
library(Rcpp)
library(RcppArmadillo)
library(tseries)
library(timeDate)
library(colorspace)
library(forecast)

###########################################################

# reading data in
crime <- scan("ArmedRobbery.txt")
# convert it to a timeseries
crimetimeseries <- ts(crime, start=c(2000,1), frequency=12)
plot.ts(crimetimeseries)

# decompose into trend, seasonality, random
crimecomponents <- decompose(crimetimeseries)
plot(crimecomponents)
# crimescomponents$trend, crimescomponents$seasonal and crimescomponents$random 


###########################################################

# read rainfall
rainfall <- scan("LondonRainfall.txt", skip=1)
# Convert it to a time series, and plot it
rainseries <-ts(rainfall, start=c(1813))
plot.ts(rainseries)

# simple moving average with a window of 3
rainSMA3 <- SMA(rainseries,n=3)
plot(rainSMA3)
# Plot sma3 along with original timeseries
ts.plot(rainseries, rainSMA3, col = c("black", "red"))

# Apply HoltWinters smoothing, setting beta and gamma to false, and plot the result:
rainseriesSmoothing <- HoltWinters(rainseries, beta=FALSE, gamma=FALSE)
plot(rainseriesSmoothing)
# forecast 
rainForecast <- forecast(rainseriesSmoothing, h=5)
plot(rainForecast)

# Estimate Alpha, beta and gamma
crimeSmoothing <- HoltWinters(crimetimeseries)
plot(crimeSmoothing) 
# forecast crime
crimeForecast <- forecast(crimeSmoothing, h=7)
plot(crimeForecast)

# leave gamma out
crimeSmoothing2 <- HoltWinters(crimetimeseries, gamma=FALSE)
crimeForecast2 <- forecast(crimeSmoothing2, h=7)
plot(crimeForecast2)

##############################################################
# autocorrelation - crime 
acf(crimetimeseries, lag.max=20)
acf(crimetimeseries, lag.max=120)
# A correlogram of the seasonal component is more informative 
acf(crimecomponents$seasonal, lag.max = 24)



