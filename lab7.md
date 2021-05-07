## LAB 7 Work:

# imports:
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

# Question 1: If we use RandomForest (random_state=310) max_depth=10 and 1000 trees for ranking the importance of the input features the top three features are (in decreasing order)
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

# Question 2: For the diabetes dataset you worked on the previous question, apply stepwise regression with add/drop p-values both set to 0.001. The model selected has the following input variables:
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
 -----
 /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:25: DeprecationWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
Add  bmi                            with p-value 3.46601e-42
Add  ltg                            with p-value 3.03968e-20
Add  map                            with p-value 3.74192e-05

# Question 3: For the diabetes dataset scale the input features by z-scores and then apply the ElasticNet model with alpha=0.1 and l1_ratio=0.5. If we rank the variables in the decreasing order of the absolute value of the coefficients the top three variables (in order) are:
from sklearn import linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
scale = StandardScaler()
dfs = scale.fit_transform(df)

model = lm.ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(dfs,y)
-----
ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.5,
           max_iter=1000, normalize=False, positive=False, precompute=False,
           random_state=None, selection='cyclic', tol=0.0001, warm_start=False)

model.coef_
-----
array([ -0.06450966, -10.44169752,  24.131291  ,  14.75251583,
        -6.39435385,  -1.73433665,  -8.41018002,   5.20382877,
        22.94137958,   3.72439119])

list(df.columns[np.abs(model.coef_)>1e-1])
v = -np.sort(-np.abs(model.coef_))
for i in range(df.shape[1]):
  print(df.columns[np.abs(model.coef_)==v[i]])
-----
Index(['bmi'], dtype='object')
Index(['ltg'], dtype='object')
Index(['map'], dtype='object')
Index(['gender'], dtype='object')
Index(['hdl'], dtype='object')
Index(['tc'], dtype='object')
Index(['tch'], dtype='object')
Index(['glu'], dtype='object')
Index(['ldl'], dtype='object')
Index(['age'], dtype='object')


# Question 5: In this problem consider 10-fold cross-validations and random_state=1693 for cross-validations and the decision tree.
 # # # The optimal pair of hyper-parameters (such as max depth and min leaf samples) is?

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target
X = X[:,0:2]
y = (y == 0) + 0 #label 1 for malignant, 0 for benign

df1 = pd.DataFrame(data=data.data, columns=data.feature_names)

cols = ['mean radius', 'mean texture']
X = df1[cols].values
df = pd.DataFrame(data=X, columns=cols)

model = DecisionTreeClassifier(random_state=1693)
params = [{'max_depth':np.arange(1,101), 'min_samples_leaf':np.arange(1,25)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='accuracy', param_grid=params)
gs_results = gs.fit(df,y)
print(gs_results.best_params_)
print('The best accuracy is achieved when: ', np.abs(gs_results.best_score_))
-----
{'max_depth': 5, 'min_samples_leaf': 5}
The best accuracy is achieved when:  0.8962406015037594

# Question 6: In this problem consider 10-fold cross-validations and random_state=12345 for cross-validations and the decision tree. Number of False Positives?

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
    
 y_names = data.target_names
 
model = DecisionTreeClassifier(max_depth=19, min_samples_leaf=5, random_state=12345)
model.fit(Xtrain,ytrain)
y_pred = model.predict(Xtest)
print(model.score(Xtrain,ytrain))
CompareClasses(ytest,y_pred,y_names)
-----
0.9698492462311558
Accuracy = 0.94
(Actual     malignant  benign
 Predicted                   
 malignant         99       5
 benign             6      61, 0.935672514619883)
 
 # Question 7:  In this problem consider 10-fold cross-validations and random_state=1693 for cross-validations and the decision tree. Accuracy?
 from sklearn.model_selection import cross_val_score

tree = DecisionTreeClassifier(max_depth=19,min_samples_leaf=5)
scores = cross_val_score(tree, X, y, cv=10)
acc_r=scores.mean()
acc_r
-----
0.8803571428571428


# Question 12: In this problem the input features will be scaled by the z-scores and consider a use a random_state=1234. If you analyze the data with benign/malign tumors from breast cancer data, consider a decision tree with max_depth=10,min_samples_leaf=20 and fit on 9 principal components the number of true positives is

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
dfs = scale.fit_transform(df1)

from sklearn.model_selection import train_test_split as tts
Xtrain, Xtest, ytrain, ytest = tts(df1,y,test_size=0.30,random_state=1234)

from sklearn.decomposition import PCA
pca = PCA(n_components=9)
principalComponents = pca.fit_transform(dfs)
principalDf = pd.DataFrame(data = principalComponents,
             columns = ['Principal Component 1','Principal Component 2','Principal Component 3','Principal Component 4','Principal Component 5','Principal Component 6','Principal Component 7','Principal Component 8','Principal Component 9'])
principalDf

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
    
y_names = data.target_names
y_names
-----
array(['malignant', 'benign'], dtype='<U9')

model= DecisionTreeClassifier(max_depth=10,min_samples_leaf=20, random_state=1234)
model.fit(Xtrain,ytrain)
y_pred = model.predict(Xtest)
print(model.score(Xtrain,ytrain))
CompareClasses(ytest,y_pred,y_names)
-----
0.9396984924623115
Accuracy = 0.88
(Actual     malignant  benign
 Predicted                   
 malignant        102      18
 benign             3      48, 0.8771929824561403)
