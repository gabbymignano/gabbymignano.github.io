# LAB 5 Work:

# Load Imports:
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

# Load Data:
from sklearn.datasets import load_breast_cancer
dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
df

y = dat.target
x = df.values
# here 0 means malignant and 1 means benign
dat.target_names

# Question 4: how many observations are in the training set?
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1963)
x_train.shape
----------
(426,30)

# Index first two features, radius and texture, for future questions
y = dat.target
cols = ['mean radius', 'mean texture']
X = df[cols].values

# Question 6: Using your Kernel SVM model with a radial basis function kernel, predict the classification of a tumor if it has a radius mean of 16.78 and a texture mean of 17.89.
svc = SVC(kernel='rbf', C=1,gamma='auto',probability=True).fit(X, y)
Z = svc.predict(X)
svc.predict([[16.78,17.89]])
----------
array([0])

# Question 7: Using your logistic model, predict the probability a tumor is malignant if it has a radius mean of 15.78 and a texture mean of 17.89.
model = LogisticRegression()
scale = StandardScaler()
pipe = Pipeline([('Scale',scale),('Regressor', model)])
pipe.fit(X, y)
predicted_classes = pipe.predict(X)

rad = 15.78
text = 17.89
proba = pipe.predict_proba([[rad,text]])
proba
----------
array([[0.64134136, 0.35865864]])


# Question 8: Using your nearest neighbor classifier with k=5 and weights='uniform', predict if a tumor is benign or malignant if the Radius Mean is 17.18, and the Texture Mean is 8.65
model = neighbors.KNeighborsClassifier(5, weights='uniform')
model.fit(X,y)

model.predict([[17.18,8.65]])
----------
array([1])
![Screen Shot 2021-04-22 at 11 43 48 AM](https://user-images.githubusercontent.com/78487402/115743829-06ba6980-a360-11eb-97d1-81aec750fe5c.png)
![Screen Shot 2021-04-22 at 11 44 19 AM](https://user-images.githubusercontent.com/78487402/115743931-1a65d000-a360-11eb-930f-6a7e8e340744.png)


# Question 9: Random Forest classifier, 100 trees, max depth 5, random state 1234. Input features: mean radius, mean texture. Apply 10 fold stratified cross-validation, estimate mean AUC
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=1234, max_depth=5, n_estimators = 100)
model.fit(X, y)

from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

# #############################################################################
# Data IO and generation

n_samples, n_features = X.shape

# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=10)
classifier = RandomForestClassifier(random_state=1234, max_depth=5, n_estimators = 100)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10,8))
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = plot_roc_curve(classifier, X[test], y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.show()
