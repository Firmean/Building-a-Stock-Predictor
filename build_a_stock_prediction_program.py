


#This program predicts stock prices by using machine learning models
!pip install quandl

import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

#Getting stocks data
df=quandl.get("WIKI/FB")
print(df.head())
#take a look at the data

#Get the Adjusted Close Price
df=df[['Adj. Close']]
#Taking a look in the new data
print(df.head())

#considering how many days we will try to predict
#a variable for predicting "n" days out into the future

forecast_out=30
 #creating another column (target or dependant variable) shifted "n" units up
df['Prediction']=df[['Adj. Close']].shift(-forecast_out)
#print the new data set
print(df.tail())

#creating the independant data set (x)
#then we are converting the dataframe to numpy array
x=np.array(df.drop(['Prediction'],axis=1))
#removing the last "n" rows
x=x[:-forecast_out]
print(x)

# creating the dependant data set (y)
# converting the datafram to a numpy array(All of the values including the NaN's)
y=np.array(df['Prediction'])
#Getting all the y values except the last "n" rows
y=y[:-forecast_out]
print(y)

#spliting the data into 80 percent into training and 20 percent in testing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#creating and training the support vector machine (Regressor)
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)
svr_rbf.fit(x_train,y_train)

#Testing Model: Scores returns the coefficient of the determination R^2 of the prediction
#the best possible score is 1.0
svm_confidence=svr_rbf.score(x_test,y_test)
print("svm confidence: ",svm_confidence)

#this shows the confidence of the prediction. The more closer to 1 , the better.

#create and train the Linear Regression Model
lr=LinearRegression()
#training the model
#fit is used for training
lr.fit(x_train,y_train)

#testing the linear regression model
lr_confidence=lr.score(x_test,y_test)
print("svm confidence: ",lr_confidence)

#support is bettter than linear regression model

#set x_forecast equal to the last 30 rows of the original data set from the Adj. Close column
x_forecast=np.array(df.drop(['Prediction'],axis=1))[-forecast_out:]
print(x_forecast)

#Print the predictions for the next "n" days. For linear regression
lr_prediction=lr.predict(x_forecast)
print('Linear regression vector=',lr_prediction)

#Print the predictions for the next "n" days. For support vector regression
svm_prediction=svr_rbf.predict(x_forecast)
print('Support vector=',svm_prediction)