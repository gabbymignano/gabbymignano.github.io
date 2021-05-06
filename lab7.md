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

![Screen Shot 2021-05-06 at 5 38 40 PM](https://user-images.githubusercontent.com/78487402/117368733-eb884780-ae91-11eb-85f8-ef49e059c737.png)

#Question 2: For the diabetes dataset you worked on the previous question, apply stepwise regression with add/drop p-values both set to 0.001. The model selected has the following input variables:
# Implementation of stepwise regression
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
    
 result = stepwise_selection(df,y,[],0.001,0.001)
 /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:25: DeprecationWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
Add  bmi                            with p-value 3.46601e-42
Add  ltg                            with p-value 3.03968e-20
Add  map                            with p-value 3.74192e-05

#Question 3: For the diabetes dataset scale the input features by z-scores and then apply the ElasticNet model with alpha=0.1 and l1_ratio=0.5. If we rank the variables in the decreasing order of the absolute value of the coefficients the top three variables (in order) are
