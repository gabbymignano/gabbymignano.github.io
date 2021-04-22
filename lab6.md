## Lab 6 Work

# Imports:
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score as acc
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load Data:
data = load_breast_cancer()
X = data.data
y= data.target
X = X[:,0:2]
y = (y == 0) + 0 #label 1 for malignant, 0 for benign

Xtrain, Xtest, ytrain, ytest = tts(X,y,test_size=0.30,random_state=1693)

scale = StandardScaler()
Xstrain = scale.fit_transform(Xtrain)
Xstest  = scale.transform(Xtest)

# Question 4: which model predicted the most false negatives?
y_names = data.target_names

def CompareClasses(actual, predicted, names=None):
    import pandas as pd
    accuracy = sum(actual==predicted)/actual.shape[0]
    classes = pd.DataFrame(columns=['Actual','Predicted'])
    classes['Actual'] = actual
    classes['Predicted'] = predicted
    conf_mat = pd.crosstab(classes['Predicted'],classes['Actual'])
    # Relabel the rows/columns if names was provided
    if type(names) != type(None):
        conf_mat.index=y_names
        conf_mat.index.name='Predicted'
        conf_mat.columns=y_names
        conf_mat.columns.name = 'Actual'
    print('Accuracy = ' + format(accuracy, '.2f'))
    return conf_mat, accuracy
    
# Bayesian
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(Xtrain,ytrain)
y_pred = model.predict(Xtest)
print(model.score(Xtrain,ytrain))
CompareClasses(ytest,y_pred,y_names)
---------
1.0
Accuracy = 0.88
(Actual     malignant  benign
 Predicted                   
 malignant         32       4
 benign             3      17, 0.875)


# Classification
model = LogisticRegression(solver='lbfgs', random_state=1693)
model.fit(Xtrain,ytrain)
y_pred = model.predict(Xtest)
print(model.score(Xtrain,ytrain))
CompareClasses(ytest,y_pred,y_names)
---------
0.8888888888888888
Accuracy = 0.91
(Actual     malignant  benign
 Predicted                   
 malignant         33       3
 benign             2      18, 0.9107142857142857)
 
 
# Naive Bayes
model = GaussianNB()
model.fit(Xtrain,ytrain)
y_pred = model.predict(Xtest)
print(model.score(Xtrain,ytrain))
CompareClasses(ytest,y_pred,y_names)
---------
0.8791423001949318
Accuracy = 0.89
(Actual     malignant  benign
 Predicted                   
 malignant         34       5
 benign             1      16, 0.8928571428571429)
 
 
# Random Forest
model = RandomForestClassifier(random_state=1693, max_depth=5, n_estimators = 1000)
model.fit(Xtrain,ytrain)
y_pred = model.predict(Xtest)
print(model.score(Xtrain,ytrain))
CompareClasses(ytest,y_pred,y_names)
---------
0.935672514619883
Accuracy = 0.88
(Actual     malignant  benign
 Predicted                   
 malignant         32       4
 benign             3      17, 0.875)
 
 
# Question 14; Guassian Naive Bayes model, average accuracy on 10-fold cross validation w random state 1693
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
predicted_classes = model.predict(X)
accuracy = acc(y,predicted_classes)
print(accuracy)
0.8857644991212654

kf = KFold(n_splits=10,shuffle=True,random_state=1693)
AC1 = []
for idxtrain, idxtest in kf.split(X):
  Xtrain = X[idxtrain,:]
  Xtest  = X[idxtest,:]
  ytrain = y[idxtrain]
  ytest  = y[idxtest]
  #Xstrain = scale.fit_transform(Xtrain)
  #Xstest  = scale.transform(Xtest)
  model.fit(X, y)
  AC1.append(acc(ytest,model.predict(Xtest)))
  
 np.mean(AC1)
 0.8858082706766919


# Question 15; Random Forest Classifier w 100 trees, max depth 7, random state 1693; average accuracy on 10-fold cross validation?
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=1693, max_depth=7, n_estimators = 100)
model.fit(X, y)
predicted_classes = model.predict(X)
accuracy = acc(y,predicted_classes)
print(accuracy)

kf = KFold(n_splits=10,shuffle=True,random_state=1693)
AC2 = []
for idxtrain, idxtest in kf.split(X):
  Xtrain = X[idxtrain,:]
  Xtest  = X[idxtest,:]
  ytrain = y[idxtrain]
  ytest  = y[idxtest]
  #Xstrain = scale.fit_transform(Xtrain)
  #Xstest  = scale.transform(Xtest)
  model.fit(X, y)
  AC2.append(acc(ytest,model.predict(Xtest)))
  
np.mean(AC2)
0.9666040100250625


# Question 16; average accuracy determined on a 10-fold cross validation (random state 1693)
model = Sequential()
model.add(Dense(16,kernel_initializer='random_normal', input_dim=2, activation='relu'))
model.add(Dense(8,kernel_initializer='random_normal', activation='relu'))
model.add(Dense(4,kernel_initializer='random_normal', activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(Xstrain, ytrain, epochs=150, verbose=0,validation_split=0.25,batch_size=10, shuffle=False)
_, accuracy = model.evaluate(Xstrain, ytrain)
print('Accuracy on the Train Data: %.2f' % (accuracy*100))
100*acc(ytest,model.predict_classes(Xstest))

from sklearn.model_selection import KFold
kf = KFold(n_splits=10,shuffle=True,random_state=1693)

AC = []
for idxtrain, idxtest in kf.split(X):
  Xtrain = X[idxtrain,:]
  Xtest  = X[idxtest,:]
  ytrain = y[idxtrain]
  ytest  = y[idxtest]
  Xstrain = scale.fit_transform(Xtrain)
  Xstest  = scale.transform(Xtest)
  model.fit(Xstrain,ytrain,epochs=150, verbose=0,validation_split=0.25,batch_size=10,shuffle=False)
  AC.append(acc(ytest,model.predict_classes(Xstest)))
  print(acc(ytest,model.predict_classes(Xstest)))
  
  np.mean(AC)
  0.8840538847117795
