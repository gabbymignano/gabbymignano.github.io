Lab 4 Work Markdown:

QUESTION 7:
------------
X = df.values
kf = KFold(n_splits=10, shuffle=True, random_state=1234)
model = Lasso(alpha = 0.03)
scale = StandardScaler()
pipe = Pipeline([('Scale',scale),('Regressor', model)])

def DoKFold(X,y,model):
  PE = []
  for idxtrain,idxtest in kf.split(X):
    # index from X the features in idxtest
    Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    pipe.fit(Xtrain,ytrain)
    yhat = pipe.predict(Xtest)
    PE.append(MAE(ytest,yhat))

  return np.mean(PE)
  
  DoKFold(X,y,model)
  
  3.380687054118185


QUESTION 8:
------------
X = df.values
kf = KFold(n_splits=10, shuffle=True, random_state=1234)
model = ElasticNet(alpha= 0.05, l1_ratio=0.9)
scale = StandardScaler()
pipe = Pipeline([('Scale',scale),('Regressor', model)])

def DoKFold(X,y,model):
  PE = []
  for idxtrain,idxtest in kf.split(X):
    # index from X the features in idxtest
    Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    pipe.fit(Xtrain,ytrain)
    yhat = pipe.predict(Xtest)
    PE.append(MAE(ytest,yhat))

  return np.mean(PE)
  
  DoKFold(X,y,model)
  
  3.364476136359214
