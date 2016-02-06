# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:51:11 2015

Blood Donations Competition via DrivenData.org

@author: Nate Bitting
Twitter: @nbitting
Email: nate.bitting@gmail.com
LinkedIn: https://www.linkedin.com/in/natebitting
"""

import pandas as pd
import numpy as np
from ggplot import *
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn import linear_model # logistic regression
from sklearn import svm # support vector machines
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn import tree # decision tree
from sklearn.ensemble import RandomForestClassifier # random forest classifier
from sklearn.neighbors import KNeighborsClassifier # knn classfier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier


train = pd.read_csv('train.csv')

#explore the data looking at histograms
ggplot(aes(x='Months since Last Donation'), data=train) + geom_histogram(binwidth=1)
ggplot(aes(x='Number of Donations'), data=train) + geom_histogram(binwidth=1)
ggplot(aes(x='Total Volume Donated (c.c.)'), data=train) + geom_histogram(binwidth=100)
ggplot(aes(x='Months since First Donation'), data=train) + geom_histogram(binwidth=1)

#explore the percentiles to let us know if trimming is required
train['Months since Last Donation'].describe(percentiles=[.5,.25,.5,.75,.95])
train['Number of Donations'].describe(percentiles=[.5,.25,.5,.75,.95])
train['Total Volume Donated (c.c.)'].describe(percentiles=[.5,.25,.5,.75,.95])
train['Months since First Donation'].describe(percentiles=[.5,.25,.5,.75,.95])

#trim at the 95% level
train['Months since Last Donation'] = np.where(train['Months since Last Donation']>23, 23, train['Months since Last Donation'])
train['Number of Donations'] = np.where(train['Number of Donations']>15, 15, train['Number of Donations'])
train['Total Volume Donated (c.c.)'] = np.where(train['Total Volume Donated (c.c.)']>3750, 3750, train['Total Volume Donated (c.c.)'])

#create a new variable using the log10 of Total Volume Donated (c.c.)
train['log_Total_Vol_Donated'] = np.log10(train['Total Volume Donated (c.c.)']+1)

X = np.array([np.array(train['Months since Last Donation']), np.array(train['Number of Donations']),
             np.array(train['log_Total_Vol_Donated']), np.array(train['Months since First Donation'])]).T

y = np.array(train['Made Donation in March 2007']).T

# create your training and test data sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9999)

# --------------------------------------------------------
# Model A: Logistic Regression
# --------------------------------------------------------
logreg = linear_model.LogisticRegression(C=1e5, class_weight='auto')
log_model_fit = logreg.fit(x_train, y_train)

 # predicted class in training data only
log_reg_y_pred = log_model_fit.predict(x_test)

print 'Logistic Regression Accuracy Score:', round(accuracy_score(y_test, log_reg_y_pred), 6)
cv_results_log = cross_val_score(log_model_fit, x_train, y_train, cv=10)
print 'Logistic Regression CV Results:',round(cv_results_log.mean(), 6), '\n'

# Logistic Regression Accuracy Score: 0.706897
# Logistic Regression CV Results: 0.663063


# --------------------------------------------------------
# Model B: SVM
# --------------------------------------------------------

# fit the linear svc model
SVM = svm.SVC(probability=True)
svm_model_fit = SVM.fit(X, y)

svm_y_pred = svm_model_fit.predict(x_test)
svm_y_pred_prob = svm_model_fit.predict_proba(x_test)

print 'SVM Accuracy Score:', round(accuracy_score(y_test, svm_y_pred), 6)
cv_results_svm = cross_val_score(svm_model_fit, x_train, y_train, cv=10)
print 'SVM CV Results:',round(cv_results_svm.mean(), 6), '\n'

# SVM Accuracy Score: 0.896552
# SVM CV Results: 0.756543


# --------------------------------------------------------
# Model C: Random Forest
# --------------------------------------------------------
clf = RandomForestClassifier(n_estimators=800, criterion='gini', random_state=8888)
rf = clf.fit(x_train, y_train)

rf_y_pred =  rf.predict(x_test)

print 'Random Forest Accuracy Score:', round(accuracy_score(y_test, rf_y_pred), 6)
cv_results_rf = cross_val_score(rf, x_train, y_train, cv=10)
print 'Random Forest CV Results:',round(cv_results_rf.mean(), 6), '\n'

# Random Forest Accuracy Score: 0.758621
# Random Forest CV Results: 0.747665


# --------------------------------------------------------
# Model D: Naive Bayes
# --------------------------------------------------------

gnb = GaussianNB()
gnb_model_fit = gnb.fit(x_train, y_train)

# # predicted class in training data only
gnb_y_pred = gnb_model_fit.predict(x_test)

print 'Naive Bayes Accuracy Score:', round(accuracy_score(y_test, gnb_y_pred), 6)
cv_results_gnb = cross_val_score(gnb_model_fit, x_train, y_train, cv=10)
print 'Naive Bayes CV Results:',round(cv_results_gnb.mean(), 6), '\n'

# Naive Bayes Accuracy Score: 0.775862
# Naive Bayes CV Results: 0.743409


# --------------------------------------------------------
# Model E: K-Nearest Neighbors
# --------------------------------------------------------

knn = KNeighborsClassifier()
knn_model = knn.fit(x_train, y_train)

# predicted class in training data only
knn_y_pred = knn_model.predict(x_test)

print 'KNN Accuracy Score:', round(accuracy_score(y_test, knn_y_pred), 6)
cv_results_knn = cross_val_score(knn_model, x_train, y_train, cv=10)
print 'KNN CV Results:',round(cv_results_knn.mean(), 6), '\n'

# KNN Accuracy Score: 0.784483
# KNN CV Results: 0.741233

# --------------------------------------------------------
# Model E: Gradient Boosting
# --------------------------------------------------------

# fit the GridSearch model
grad = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
grad_model = grad.fit(x_train, y_train)

# predicted class in training data only
grad_y_pred = grad_model.predict(x_test)

print 'Gradient Boosting Accuracy Score:', round(accuracy_score(y_test, grad_y_pred), 6)
cv_results_grad = cross_val_score(grad_model, x_train, y_train, cv=10)
print 'Gradient Boosting CV Results:',round(cv_results_grad.mean(), 6), '\n'

# Gradient Boosting Accuracy Score: 0.767241
# Gradient Boosting CV Results: 0.76283

# --------------------------------------------------------
# Model E: Extreme RF
# --------------------------------------------------------

# fit the RF model
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
extra = clf.fit(x_train, y_train)

# predicted class in training data only
extra_y_pred = extra.predict(x_test)

print 'Extreme RF Accuracy Score:', round(accuracy_score(y_test, extra_y_pred), 6)
cv_results_extra = cross_val_score(extra, x_train, y_train, cv=10)
print 'Extreme RF CV Results:',round(cv_results_extra.mean(), 6), '\n'

# Extreme RF Accuracy Score: 0.724138
# Extreme RF CV Results: 0.730366


# --------------------------------------------------------
# Ensemble (with weights)
# --------------------------------------------------------

# SVM Accuracy Score: 0.896552
# KNN Accuracy Score: 0.784483
# Naive Bayes Accuracy Score: 0.775862
# Gradient Boosting Accuracy Score: 0.767241
# Random Forest Accuracy Score: 0.758621
# Extreme RF Accuracy Score: 0.724138
# Logistic Regression Accuracy Score: 0.706897

# Ensembling didn't improve the model, therefore, we will use the best model, SVM

y_pred = svm_y_pred
print 'Ensemble Accuracy Score:', round(accuracy_score(y_test, y_pred), 6)


# --------------------------------------------------------
# Model Deployment
# --------------------------------------------------------

test = pd.read_csv('test.csv')

#trim at the 95% level
test['Months since Last Donation'] = np.where(test['Months since Last Donation']>23, 23, test['Months since Last Donation'])
test['Number of Donations'] = np.where(test['Number of Donations']>15, 15, test['Number of Donations'])
test['Total Volume Donated (c.c.)'] = np.where(test['Total Volume Donated (c.c.)']>3750, 3750, test['Total Volume Donated (c.c.)'])

#create a new variable using the log10 of Total Volume Donated (c.c.)
test['log_Total_Vol_Donated'] = np.log10(test['Total Volume Donated (c.c.)']+1)

test_X = np.array([np.array(test['Months since Last Donation']), np.array(test['Number of Donations']),
                    np.array(test['log_Total_Vol_Donated']), np.array(test['Months since First Donation'])]).T

predictions = svm_model_fit.predict_proba(test_X)
