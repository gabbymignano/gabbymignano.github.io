## LAB 7 Work:

#imports:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV

#Question 1: If we use RandomForest (random_state=310) max_depth=10 and 1000 trees for ranking the importance of the input features the top three features are (in decreasing order)
columns = 'age gender bmi map tc ldl hdl tch ltg glu'.split()
diabetes = datasets.load_diabetes()
df = pd.DataFrame(diabetes.data, columns=columns)
y = diabetes.target

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=310, max_depth=10,n_estimators=1000)
df=pd.get_dummies(df)
model.fit(df,y)

features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-9:]  # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

