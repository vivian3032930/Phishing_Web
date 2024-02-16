import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE as RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from feature_engine.selection import RecursiveFeatureElimination

# Kaggle dataset url = https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset?resource=download
# The dataset from this url above


df=pd.read_csv("dataset_phishing.csv")
# 將 status 從 string 轉為 int / legitimate -> 0; phishing -> 1
classes={'legitimate':0, 'phishing':1}
df['status'] = df['status'].map(classes)

X = df.select_dtypes(include=['number']).copy()
X = X.drop(["status"], axis=1)
y = df["status"]

# find the elbow index for deciding estimator value
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

cv_scores = []

# run through the n_estimators values
for n_estimators in range(1, 87):
    model = GradientBoostingRegressor(n_estimators = n_estimators, random_state = 42)
    scores = cross_val_score(model, X, y, cv=2, scoring='r2')

    # Store the mean R2 score of the cross-validation
    cv_scores.append(np.mean(scores))

# find the highest R2 score
sorted_indices = np.argsort(cv_scores)[::-1]

# Calculate the first differences of the R2 scores
diff = np.diff(cv_scores)

# Calculate the second differences of the R2 scores
diff2 = np.diff(diff)

# Find the index of the elbow point
elbow_index = np.argmax(diff2) + 2

# gradient boosting machine for regression, use it to evaluate the features during the search.
model = GradientBoostingRegressor(
    n_estimators = 30,
    random_state = 42,
)
# feature elimination search, uses the previous gradient boosting machine and the R2 to evaluate the feature subsets using 2-fold cross-validation
RFE_model = RecursiveFeatureElimination(
    estimator = model, # evaluate the feature subsets
    scoring = 'r2',
    threshold = 0.001,  # minimum acceptable decrease
    cv = 2, # number of cross-validation folds
)

# X_RFE = RFE_model.transform(X_train)
X_RFE = RFE_model.transform(X)

corr_matrix = df.corr()
corr_matrix['status']
status_corr = corr_matrix['status']

def feature_selector_correlation(cmatrix, threshold):
    selected_features = []
    feature_score = []
    i = 0
    for score in cmatrix:
        if abs(score) > threshold:
            selected_features.append(cmatrix.index[i])
            feature_score.append( ['{:3f}'.format(score)])
        i += 1
    result = list(zip(selected_features, feature_score))
    return result

features_selected = feature_selector_correlation(status_corr, 0.25)
selected_features = [i for (i,j) in features_selected if i != 'status']
corr_selected = df[selected_features]

# set columns for compare purpose
corr_selected_features = set(corr_selected.columns)
x_rfe_features = set(X_RFE.columns)

# find different column
features_only_in_ranfor = corr_selected_features - x_rfe_features
features_only_in_x_rfe = x_rfe_features - corr_selected_features

# Random forest
X = corr_selected
y = df['status']
X_train_corr, X_test_corr, y_train_corr, y_test_corr = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Create a Random Forest model
model = RandomForestClassifier(n_estimators = 100, random_state = 42)

# Fit the model to the training data
model.fit(X_train_corr, y_train_corr)

from sklearn.metrics import accuracy_score

# Make predictions on the test data
y_pred_corr = model.predict(X_test_corr)

# Evaluate the model's performance
accuracy_rf_corr = accuracy_score(y_test_corr, y_pred_corr)

# random forest using X_RFE feature selection
X = X_RFE
y = df['status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Create a Random Forest model
model = RandomForestClassifier(n_estimators = 100, random_state = 42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy_rf_rfe = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_rf_rfe)

# xgboost using corr_selected
import xgboost as xgb

model = xgb.XGBClassifier()
model.fit(X_train_corr, y_train_corr)
y_pred_corr = model.predict(X_test_corr)
accuracy_xgb_corr = accuracy_score(y_test_corr, y_pred_corr)
print("Accuracy:", accuracy_xgb_corr)

# xgboost using X_RFE feature selection
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy_xgb_rfe = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_xgb_rfe)

# SVM using corr_selected
from sklearn.svm import SVC

model = SVC()
model.fit(X_train_corr, y_train_corr)
y_pred_corr = model.predict(X_test_corr)
accuracy_svm_corr = accuracy_score(y_test_corr, y_pred_corr)
print("Accuracy:", accuracy_svm_corr)

# SVM using X_RFE feature selection
model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy_svm_rfe = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_svm_rfe)

# Logistic regression using corr_selected
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_corr, y_train_corr)
y_pred_corr = model.predict(X_test_corr)
accuracy_lr_corr = accuracy_score(y_test_corr, y_pred)
print("Accuracy:", accuracy_lr_corr)

# Logistic regression using X_RFE feature selection
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy_lr_rfe = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy_lr_rfe)

# Conclusi1on
print("Accuracy of correlation pre-process and random forest:\n", accuracy_rf_corr)
print("Accuracy of RFE pre-process and random forest:\n", accuracy_rf_rfe)
print("Accuracy of correlation pre-process and xgboost:\n", accuracy_xgb_corr)
print("Accuracy of RFE pre-process and xgboost:\n", accuracy_xgb_rfe)
print("Accuracy of correlation pre-process and SVM:\n", accuracy_svm_corr)
print("Accuracy of RFE pre-process and SVM:\n", accuracy_svm_rfe)
print("Accuracy of correlation pre-process and logistic regression:\n", accuracy_lr_corr)
print("Accuracy of RFE pre-process and logistic regression:\n", accuracy_lr_rfe)