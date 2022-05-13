library(reshape)
library(astsa)
library(forecast)
library(stargazer)
library(tseries)


data=read.csv('temperatures.csv') #read data CET
data[data==-99.9]=NA #set future items to NA
data$yearmean=rowMeans(data[,2:5]) #yearly means
data=subset(data,Year>=1752 & Year<2017) #subset due to recordings change. From 1752 switched to gregorian calendar and same measuring techniques
#data$yearmean=ts(data$yearmean,start=1752)
data[2:6]=ts(data[2:6]) #turn to time series

####plot the data by season and by year for initial analysis
#seasonal
par(mfrow=c(2,2))
plot.ts(data$Year,data$DJF,type='l',main='Winter',xlab='Year',ylab='Temperature') #winter
hist(data$DJF,main='Winter',xlab='Temperature') #make histogram for winter
qqnorm(data$DJF);qqline(data$DJF,col=2)
shapiro.test(data$DJF)
plot.ts(data$Year,data$MAM,type='l',main='Spring',xlab='Year',ylab='Temperature') #spring
hist(data$MAM,main='Spring',xlab='Temperature') #make historgram for spring
qqnorm(data$MAM);qqline(data$MAM,col=2)
shapiro.test(data$MAM)
plot.ts(data$Year,data$JJA,type='l',main='Summer',xlab='Year',ylab='Temperature') #summer
hist(data$JJA,main='Summer',xlab='Temperature') #summer historgram
qqnorm(data$JJA);qqline(data$JJA,col=2)
shapiro.test(data$JJA)
plot.ts(data$Year,data$SON,type='l',main='Fall',xlab='Year',ylab='Temperature') #fall
hist(data$SON,main='Fall',xlab='Temperature') #fall histogram
qqnorm(data$SON);qqline(data$SON,col=2)
shapiro.test(data$SON)

#yearly
par(mfrow=c(2,2))
plot.ts(data$Year,data$yearmean,type='l',main='Yearly CET',xlab='Year',ylab='Temperature') # yearly CET results
hist(data$yearmean,main='',xlab='Temperature')
qqnorm(data$yearmean);qqline(data$yearmean,col=2)
shapiro.test(data$yearmean)

#### smooth data with MA and with Kernals
#the function ksmooth is used which uses a Nadaraya-Watson regression estimate to smooth
#year long ma smoothing
par(mfrow=c(1,1))
ma10=filter(data$yearmean,filter=rep(1,10)/10,sides=1) #run moving average filter... 9 year lag 
plot(data$Year,data$yearmean,type='l',main='Yearly CET with MA Smoothing',xlab='Year',ylab='Temperature')
lines(data$Year,ma10,col='blue')

#kernel smoothing yearlong
plot(data$Year,data$yearmean,type='l',main='Yearly CET with Kernel Smoothing',xlab='Year',ylab='Temperature')
lines(ksmooth(data$Year,data$yearmean,'normal',bandwidth=35),type='l',col='red',main='Smoothed Yearly Temperature',xlab='Year',ylab='Temperature')
lines(ksmooth(data$Year,data$yearmean,'normal',bandwidth=5),type='l',col='green',main='Smoothed Yearly Temperature',xlab='Year',ylab='Temperature')
lines(ksmooth(data$Year,data$yearmean,'normal',bandwidth=100),type='l',col='blue',main='Smoothed Yearly Temperature',xlab='Year',ylab='Temperature')


#### Cross validation
window=35
y=ksmooth(data$Year,data$yearmean,'normal',bandwidth=window)$y
x=data$yearmean


mse=function(data,window){
  x=data$yearmean
  y=ksmooth(data$Year,data$yearmean,'normal',bandwidth=window)$y
  return (mean((x-y)^2))
}

bic_calc=function(data,window){
  lm1 <- lm(data$yearmean ~ ksmooth(data$Year,data$yearmean,'normal',bandwidth=window)$y)
  return (AIC(lm1,k=length(data$yearmean)-window/2))}


window=seq(5,60,by=1)
cv=data.frame(window=window)
cv$mse=NaN
cv$mae=NaN
cv$mamse=NaN
cv$bic=NaN
#data[200:250, ]

for (i in 1:nrow(cv)){
  print(i)
  print(window[i])
  #cv[i,'mae']=mae(data[100:225, ],window[i])
  cv[i,'mse']=mse(data[165:225, ],window[i])
#  cv[i,'mamse']=mamse(data,window[i])
  cv[i,'bic']=-bic_calc(data[1:225,],window[i])
  
}
#plot(cv[c('window','mae')])
plot(cv[c('window','bic')])

plot(cv[c('window','mse')])



# kernel smoothing seasons
par(mfrow=c(2,2)) #2x2
plot(ksmooth(data$Year,data$DJF,'normal',bandwidth=10),type='l',col='red',main='Smoothed Winter Temperature',xlab='Year',ylab='Temperature')
lines(ksmooth(data$Year,data$DJF,'normal',bandwidth=20),type='l',col='blue',main='Smoothed Winter Temperature',xlab='Year',ylab='Temperature')
plot(ksmooth(data$Year,data$MAM,'normal',bandwidth=10),type='l',col='red',main='Smoothed Spring Temperature',xlab='Year',ylab='Temperature')
lines(ksmooth(data$Year,data$MAM,'normal',bandwidth=20),type='l',col='blue',main='Smoothed Spring Temperature',xlab='Year',ylab='Temperature')
plot(ksmooth(data$Year,data$JJA,'normal',bandwidth=10),type='l',col='red',main='Smoothed Summer Temperature',xlab='Year',ylab='Temperature')
lines(ksmooth(data$Year,data$JJA,'normal',bandwidth=20),type='l',col='blue',main='Smoothed Spring Temperature',xlab='Year',ylab='Temperature')
plot(ksmooth(data$Year,data$SON,'normal',bandwidth=10),type='l',col='red',main='Smoothed Fall Temperature',xlab='Year',ylab='Temperature')
lines(ksmooth(data$Year,data$SON,'normal',bandwidth=20),type='l',col='blue',main='Smoothed Spring Temperature',xlab='Year',ylab='Temperature')

#### spectral analysis for different years and for different seasons
#calculate periodogram using a fast Fourier transform and smooth with Daniell and modified Daniell smoothers (moving averages giving half weight to the end values). Default is detrend
#modified daniell and daniel give similar results
#using spec.pgram will output a frequency,spectrum,kernel,degrees of freedom,bandwith
par(mfrow=c(1,1))
regular_spectrum=spec.pgram(data$yearmean,taper=0,log='no',main='Yearly Temperature Periodogram') #plot regular spectrum
#nonparametric periodogram
the_kernel=kernel(coef='daniell',m=3) #set kernel to modified daniell,and run with dimensions 3
nonp.spec=spec.pgram(data$yearmean,kernel=the_kernel,taper=0,log='no',main='Smoothed Periodogram') #do not taper,nor detrend and log
nonparametric_frame=data.frame(nonp.spec$freq,nonp.spec$spec)
colnames(nonparametric_frame)=c('Frequency','Spectrum')
plot(nonparametric_frame,col=ifelse((nonparametric_frame$Spectrum>=.6)&(nonparametric_frame$Frequency>=.03),'red','black')) #There are four distinct local maximum
abline(h =.6,col='red')
which((nonparametric_frame$Spectrum>=.6)&(nonparametric_frame$Frequency>=.03)) #these are the frequencies at which the local minimums exist,10,22,39,141

#dominante frequency index at 9,21,35,and 127... turn them into a matrix
frequency_matrix=matrix(c(nonp.spec$freq[9],1/nonp.spec$freq[9],nonp.spec$spec[9],
               nonp.spec$freq[21],1/nonp.spec$freq[21],nonp.spec$spec[21],
               nonp.spec$freq[35],1/nonp.spec$freq[35],nonp.spec$spec[35],
               nonp.spec$freq[127],1/nonp.spec$freq[127],nonp.spec$spec[127]),nrow=4,ncol=3,byrow=T)
colnames(frequency_matrix)=c("Frequency","Years per Cycle","Spectrum")
frequency_matrix #output matrix
#paste('Cycle of',toString(round(1/nonp.spec$freq[9]),4),'years and spectrum of',toString(round(nonp.spec$spec[9],3)))
#paste('Cycle of',toString(round(1/nonp.spec$freq[21]),4),'years and spectrum of',toString(round(nonp.spec$spec[21],3)))
#paste('Cycle of',toString(round(1/nonp.spec$freq[35]),4),'years and spectrum of',toString(round(nonp.spec$spec[35],3)))
#paste('Cycle of',toString(round(1/nonp.spec$freq[127]),4),'years and spectrum of',toString(round(nonp.spec$spec[127],3)))
#U = qchisq(.025,2)
#L = qchisq(.975,2)

#seasonal frequency with periodograms... year has similarities which each season
the_kernel=kernel('daniell',2)
par(mfrow=c(2,2))
spring_spectrum=spec.pgram(data$MAM,the_kernel,taper=0,log='no',main='Smoothed Spring Periodogram')
summer_spectrum=spec.pgram(data$JJA,the_kernel,taper=0,log='no',main='Smoothed Summer Periodogram')
fall_spectrum=spec.pgram(data$SON,the_kernel,taper=0,log='no',main='Smoothed Fall Periodogram')
winter_spectrum=spec.pgram(data$DJF,the_kernel,taper=0,log='no',main='Smoothed Winter Periodogram')
par(mfrow=c(1,1))
plot(winter_spectrum$freq,winter_spectrum$spec,col='red',type='l',main='Combined Seasonal Periodogram',xlab='Spectrum',ylab='frequency')
lines(spring_spectrum$freq,spring_spectrum$spec,col='blue')
lines(summer_spectrum$freq,summer_spectrum$spec,col='green')
lines(fall_spectrum$freq,fall_spectrum$spec,col='black')
legend('topright',cex=.5,legend=c('Winter','Spring','Summer','Fall'),col=c('red','blue','green','black'),ncol=2,pch='l')
#spring
spring_frame=data.frame(spring_spectrum$freq,spring_spectrum$spec)
colnames(spring_frame)=c('Frequency','Spectrum')
plot(spring_frame,col=ifelse((spring_frame$Spectrum>=1.5)&(spring_frame$Frequency>=.03),'red','black')) #There are four distinct local maximum
abline(h =1.5,col='red')
which((spring_frame$Spectrum>=1.5)&(spring_frame$Frequency>=.03)) #these are the frequencies at which the local minimums exist,10,22,39,141
# index 18 and 127 are significant
spring_matrix=matrix(c(summer_spectrum$freq[18],1/summer_spectrum$freq[18],summer_spectrum$spec[18],
                       summer_spectrum$freq[127],1/summer_spectrum$freq[127],summer_spectrum$spec[127]),nrow=2,ncol=3,byrow=T)
colnames(spring_matrix)=c("Frequency","Years per Cycle","Spectrum")
spring_matrix #output matrix

#### search for a trend (MA2 + quadratic trend)
##quadractic trend
data$year_2=data$Year^2
quadratic_trend_reg=lm(yearmean~Year+year_2,data=data)
summary(quadratic_trend_reg)
#stargazer(quadratic_trend_reg)
resid=quadratic_trend_reg$residuals
#plot quadratic trend only
par(mfrow=c(1,1))
plot(data$Year,data$yearmean,type='l',main='Data with Quadratic Trend',xlab='Year',ylab='Average Yearly Temperature') #plot data
lines(data$Year,fitted(quadratic_trend_reg),col='blue') #add fitted quadratic trend
trend_upper_bound=predict(quadratic_trend_reg,interval='predict')[,2] #in sample upper bound
trend_lower_bound=predict(quadratic_trend_reg,interval='predict')[,3] #in sample lower pount
lines(data$Year,trend_upper_bound,col='blue',lty='dotted')
lines(data$Year,trend_lower_bound,col='blue',lty='dotted')
##add arma component to residuals
#unit root test for residuals
adf.test(resid) #p-values less than .05,reject null hypothesis
# ACF of trend residuals
par(mfrow=c(1,2))
Acf(resid,35,main='ACF')
Pacf(resid,35,main='PACF') #2nd has large spike,try AR(2)
# fit ARMA to trend resid
arma_2_0=arima(resid,c(2,0,0))
summary(arma_2_0) #ran cross validation... arma(2,0) is the best fit for residuals
stargazer(arma_2_0)
# residual diag of ARMA
arma_2_0_residuals=arma_2_0$residuals
par(mfrow=c(1,2))
Acf(arma_2_0_residuals,main='ACF Plot on residuals')
Pacf(arma_2_0_residuals,main='PACF Plot on residuals')
#plot fitted arma and residuals are AR(2)
par(mfrow=c(1,1))
plot(data$Year,data$yearmean,type='l',main='Data with Quadratic Trend and AR(2)',xlab='Year',ylab='Average Yearly Temperature')
lines(data$Year,fitted(quadratic_trend_reg)+fitted(arma_2_0),col='blue') #add ar(2)
lines(data$Year,trend_upper_bound+fitted(arma_2_0),col='blue',lty='dotted')
lines(data$Year,trend_lower_bound+fitted(arma_2_0),col='blue',lty='dotted')

#### Forecast into the future... go from 2000 to 2025 out of sample. 
data_train=subset(data,Year<2000) #training set is pre-2000 data
data_train$year_2=data_train$Year^2 #add quadratic trend
quadratic_trend_reg_2=lm(yearmean~Year+year_2,data=data_train)
summary(quadratic_trend_reg_2)
stargazer(quadratic_trend_reg_2)
resid_2=quadratic_trend_reg_2$residuals
#unit root test for resid
adf.test(resid_2)
# ACF of trend residuals
par(mfrow=c(1,2))
Acf(resid_2,35,main='ACF')
Pacf(resid_2,35,main='PACF') #stil ar(2)
arma2_2_0=arima(resid_2,c(2,0,0))
summary(arma2_2_0)
stargazer(arma2_2_0)
# residual diag of ARMA
resid2_2_0=arma2_2_0$residuals
par(mfrow=c(2,2))
Acf(resid2_2_0); Pacf(resid2_2_0)
Acf(resid2_2_0^2); Pacf(resid2_2_0^2)

par(mfrow=c(1,1))
prediction=quadratic_trend_reg_2$coefficients[1]+quadratic_trend_reg_2$coefficients[2]*(2000:2025)+quadratic_trend_reg_2$coefficients[3]*(2000:2025)^2
forecasts=forecast(arma2_2_0,26)$mean+prediction
#plot(1989:2016,c(data_train$yearmean[(nrow(data_train)-10):nrow(data_train)],forecasts),main='Forecast vs. Actual',xlab='Years',ylab='Temperature')
trend_upper=prediction+1.96*0.5996
trend_lower=prediction-1.96*0.5996
upper=forecast(arma2_2_0,26)$upper[,2]+trend_upper #95% ci lines
lower=forecast(arma2_2_0,26)$lower[,2]+trend_lower

plot(1752:2025,c(data_train$yearmean,forecasts),main='Forecast vs. Actual',xlab='Years',ylab='Temperature',ylim=c(7.5,12.5))
lines(1752:2016,data$yearmean,col='Blue')
abline(v=2000,lty=2,col='red')
lines(2000:2025,upper,col='blue',lty='dotted') #plot 95% ci lines
lines(2000:2025,lower,col='blue',lty='dotted')






