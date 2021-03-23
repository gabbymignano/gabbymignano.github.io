Imports:

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split

----
Q1 and Q2:
import numpy as np
import pandas as pd
df = pd.read_csv('drive/My Drive/Data Sci/weatherHistory.csv')
df

df.shape

---
Q3:- incorrect
X = df[["Temperature (C)"]]
y = df[["Humidity"]]

def RMSE(y,yhat):
  return np.sqrt(MSE(y,yhat))
RMSE(X,y)

--- 
Q4:
X = df["Temperature (C)"].values
y = df["Humidity"].values

kf = KFold(n_splits=20, random_state=2020,shuffle=True)
model = Ridge(alpha=0.1)
scale = StandardScaler()
pipe = Pipeline([('Scale',scale),('Regressor', model)])

def DoKFold1(X,y,model):
  PE = []
  for idxtrain,idxtest in kf.split(X):
    Xtrain = X[idxtrain]
    Xtest = X[idxtest]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    pipe.fit(Xtrain,ytrain)
    yhat = pipe.predict(Xtest)
    PE.append(RMSE(ytest,yhat))
  return np.mean(PE)
  
DoKFold1(X.reshape((-1,1)),y,model)

--- 
Q5:
Xf = df["Apparent Temperature (C)"].values
yf = df["Humidity"].values
Xr= Xf.reshape((-1,1))

model = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=1693)

def DoKFold2(X,y,model,k):
  PE = []
  kf= KFold(n_splits=k, shuffle=True,random_state=1693)
  for idxtrain,idxtest in kf.split(X):
    Xtrain = X[idxtrain]
    Xtest = X[idxtest]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    model.fit(Xtrain,ytrain)
    yhat = model.predict(Xtest)
    PE.append(RMSE(ytest,yhat))
  return np.mean(PE)
  
DoKFold2(Xr,yf,model,10)

---
Q6:
Xf = df["Apparent Temperature (C)"].values
yf = df["Humidity"].values

kf = KFold(n_splits=10, random_state=1693,shuffle=True)
model = LinearRegression()
polynomial_features= PolynomialFeatures(degree=6)

polynomial_features= PolynomialFeatures(degree=6)
RMSE_test = []
for idxtrain, idxtest in kf.split(Xf):
  x_train = Xf[idxtrain]
  x_test = Xf[idxtest]
  y_train = yf[idxtrain]
  y_test  = yf[idxtest]
  x_poly_train = polynomial_features.fit_transform(np.array(x_train).reshape(-1,1))
  x_poly_test = polynomial_features.fit_transform(np.array(x_test).reshape(-1,1))
  model.fit(x_poly_train,y_train)
  yhat_train = model.predict(x_poly_train)
  yhat_test = model.predict(x_poly_test)
  RMSE_test.append(RMSE(y_test,yhat_test))

print('The avg RMSE on the test sets is : '+str(np.mean(RMSE_test)))

---
Q7:
X = df["Temperature (C)"].values
y = df["Humidity"].values

kf = KFold(n_splits=10, shuffle=True, random_state=1234)
model = Ridge(alpha=0.2)
scale = StandardScaler()
pipe = Pipeline([('Scale',scale),('Regressor', model)])

def DoKFold3(X,y,model):
  PE = []
  for idxtrain,idxtest in kf.split(X):
    Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    pipe.fit(Xtrain,ytrain)
    yhat = pipe.predict(Xtest)
    PE.append(RMSE(ytest,yhat))
  return np.mean(PE)
  
DoKFold3(X.reshape((-1,1)),y,model)

---
Q8:
y = df['Temperature (C)'].values
cols = ['Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)']
X = df[cols].values

kf = KFold(n_splits=10, random_state=1234,shuffle=True)
model = LinearRegression()
scale = StandardScaler()
polynomial_features= PolynomialFeatures(degree=6)
pipe = Pipeline([('Scale',scale),('Regressor', model)])

# SCALED W PIPELINE IN CODE:
polynomial_features= PolynomialFeatures(degree=6)
RMSE_test = []
for idxtrain, idxtest in kf.split(X):
  x_train = X[idxtrain,:]
  x_test = X[idxtest,:]
  y_train = y[idxtrain]
  y_test  = y[idxtest]
  x_poly_train = polynomial_features.fit_transform(np.array(x_train))
  x_poly_test = polynomial_features.fit_transform(np.array(x_test))
  pipe.fit(x_poly_train,y_train)
  yhat_train = pipe.predict(x_poly_train)
  yhat_test = pipe.predict(x_poly_test)
  RMSE_test.append(RMSE(y_test,yhat_test))
  
print('The avg RMSE on the test sets is : '+str(np.mean(RMSE_test)))

---
Q9:
y1 = df['Temperature (C)'].values
cols = ['Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)']
X1 = df[cols].values

model = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=1234)
scale = StandardScaler()
pipe = Pipeline([('Scale',scale),('Regressor', model)])

def DoKFold2(X,y,model,k):
  PE = []
  kf= KFold(n_splits=k, shuffle=True,random_state=1234)
  for idxtrain,idxtest in kf.split(X):
    Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    pipe.fit(Xtrain,ytrain)
    yhat = pipe.predict(Xtest)
    PE.append(RMSE(ytest,yhat))
  return np.mean(PE)
  
 DoKFold2(X1,y1,model,10)
 
 ---
 Q10:
 import matplotlib.pyplot as plt
plt.scatter(df['Temperature (C)'], df['Humidity'])
plt.xlabel('Temperature (C)')
plt.ylabel('Humidity')
plt.show()
