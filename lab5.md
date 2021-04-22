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
(426,30)
