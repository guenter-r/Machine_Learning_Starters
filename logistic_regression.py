import numpy as np, pandas as pd
import sklearn.datasets
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, mean_squared_error as mse

# load data
data = sklearn.datasets.load_breast_cancer()

x,y = data.data, data.target
x.shape, y.shape

# xy = np.concatenate((x,y.reshape(-1,1)), axis=1)
X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size=0.2)

## Build pipeline

pipeline = Pipeline([ ('scaler', StandardScaler()), ('LR', LogisticRegression(random_state=42)) ])
pipeline.fit(X_train, y_train)
pipeline.predict(X_test)
pipeline.score(X_test, y_test)
pipeline['LR'].coef_



## NON pipeline:

# 1 StandardScaler (fit to train data and transform. do that seperately so that
#   scaler can be used later to .transform(X_test)).
# 2 fit_transform  data
# 3 fit model
# 4 predict

scaler = StandardScaler()
x_train_scaled = scaler.fit(X_train).transform(X_train)

logr = LogisticRegression(random_state=42)
logr.fit(x_train_scaled, y_train)

x_test_scaled = scaler.transform(X_test)
1-mse(logr.predict(x_test_scaled), y_test)
logr.score(x_test_scaled, y_test)

preds = logr.predict_proba(x_test_scaled)
1-mse((preds[:,1] > .5)*1, y_test)

for i in np.linspace(0,1,11):
    error = 1-mse((preds[:,0] < i)*1, y_test)
    print(np.round(i,4), np.round(error,4))


tpr, tpr, thresholds = roc_curve(y_test,preds[:,1])
np.round(thresholds, 3)
