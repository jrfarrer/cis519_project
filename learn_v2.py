from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from pprint import pprint
from scipy import stats

from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from time import time
import logging

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from itertools import cycle

filename = 'preprocessed.csv'
data = pd.read_csv(filename, delimiter=',')
X = np.asarray(data.ix[:, 0])
y = np.asarray(data.ix[:, 1])
y_int = pd.factorize(y)[0]
y_classes = pd.factorize(y)[1]

vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', sublinear_tf=True)
X_vectorized = vectorizer.fit_transform(X)

##########################################################
##  Parameter Tuning
##########################################################
##########################################################
##  Logistic Regression
##########################################################
lr_text_clf = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
                     ('tfidf', TfidfTransformer(sublinear_tf=True)),
                     ('clf', LogisticRegression()),
])


params = {'clf__C': [0.001, 0.01, 0.5, 1, 5, 10]}
gs_clf = GridSearchCV(lr_text_clf, params, n_jobs=-1, cv = 10)
gs_clf = gs_clf.fit(X, y_int)
#gs_clf.best_params_
# {'clf__C': 5}
lr_text_clf_best = gs_clf.best_estimator_

##########################################################
##  NB
##########################################################
nb_text_clf = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
                     ('tfidf', TfidfTransformer(sublinear_tf=True)),
                     ('clf', MultinomialNB()),
])


params = {'clf__alpha': [0.001, 0.01, 0.25, 0.5, 0.75, 1, 2]}
gs_clf = GridSearchCV(nb_text_clf, params, n_jobs=-1, cv = 10)
gs_clf = gs_clf.fit(X, y_int)
#gs_clf.best_params_
# {'clf__alpha': 0.25}
nb_text_clf_best = gs_clf.best_estimator_


##########################################################
##  Metrics Table using cross-validation
##########################################################
df = pd.DataFrame(columns = ['model','metric','value'])
##########################################################
## 10-fold cross-validation for NB and LogReg
#########################################################
rkf = RepeatedKFold(n_splits = 10, n_repeats = 1)
for train_index, test_index in rkf.split(X_vectorized):
    X_train, X_test = X_vectorized[train_index], X_vectorized[test_index]
    y_train, y_test = y[train_index], y[test_index]

    log_y_pred = LogisticRegression(C = 5).fit(X_train, y_train).predict(X_test)
    df = df.append({'model':'LogReg','metric':'accuracy','value':accuracy_score(y_test,log_y_pred)}, ignore_index=True)
    df = df.append({'model':'LogReg','metric':'precision','value':precision_score(y_test,log_y_pred, average='weighted')}, ignore_index=True)
    df = df.append({'model':'LogReg','metric':'recall','value':recall_score(y_test,log_y_pred, average='weighted')}, ignore_index=True)
    df = df.append({'model':'LogReg','metric':'f1','value':f1_score(y_test,log_y_pred, average='weighted')}, ignore_index=True)

    nb_y_pred = MultinomialNB(alpha = 0.25).fit(X_train, y_train).predict(X_test)
    df = df.append({'model':'MultinomialNB','metric':'accuracy','value':accuracy_score(y_test,nb_y_pred)}, ignore_index=True)
    df = df.append({'model':'MultinomialNB','metric':'precision','value':precision_score(y_test,nb_y_pred, average='weighted')}, ignore_index=True)
    df = df.append({'model':'MultinomialNB','metric':'recall','value':recall_score(y_test,nb_y_pred, average='weighted')}, ignore_index=True)
    df = df.append({'model':'MultinomialNB','metric':'f1','value':f1_score(y_test,nb_y_pred, average='weighted')}, ignore_index=True)

##########################################################
## 10-fold cross-validation for SVM with Cosine Similiary 
## 10ths of data because it would not run on full data set
#########################################################
rs = ShuffleSplit(n_splits = 10, train_size=len(y_int)/10, test_size=0)
for train_index, test_index in rs.split(X_vectorized):
	rkf = RepeatedKFold(n_splits = 10, n_repeats = 1)
	X_vectorized_tenth = X_vectorized[train_index]
	y_tenth = y[train_index]
	for train_index_rkf, test_index_rkf in rkf.split(X_vectorized_tenth):
	    X_train, X_test = X_vectorized_tenth[train_index_rkf], X_vectorized_tenth[test_index_rkf]
	    y_train, y_test = y_tenth[train_index_rkf], y_tenth[test_index_rkf]

	    svm_y_pred = SVC(kernel = cosine_similarity).fit(X_train, y_train).predict(X_test)
    	df = df.append({'model':'SVM','metric':'accuracy','value':accuracy_score(y_test,svm_y_pred)}, ignore_index=True)
    	df = df.append({'model':'SVM','metric':'precision','value':precision_score(y_test,svm_y_pred, average='weighted')}, ignore_index=True)
    	df = df.append({'model':'SVM','metric':'recall','value':recall_score(y_test,svm_y_pred, average='weighted')}, ignore_index=True)
    	df = df.append({'model':'SVM','metric':'f1','value':f1_score(y_test,svm_y_pred, average='weighted')}, ignore_index=True)
##########################################################
## Summary
#########################################################
df_stats = df.groupby(['model','metric'], as_index=False)['value'].agg({'mean' : np.mean, 'std_err' : lambda x: stats.sem(x, ddof = 0)})
df_stats
df_stats.to_csv('model_metrics.csv')
##########################################################
## Final Model: Logistic Regression
##########################################################

##########################################################
##  ROC - only LogReg and NB because of computation efficiency
##########################################################
X_train_roc, X_test_roc, y_train_roc, y_test_roc = train_test_split(X, y_int, test_size = 0.20, random_state=42)

lr_text_clf_best.fit(X_train_roc, y_train_roc)
nb_text_clf_best.fit(X_train_roc, y_train_roc)


idx_classes = [i for i, item in enumerate(y_classes) if item in y_classes]

lr_y_score = lr_text_clf_best.decision_function(X_test_roc)
nb_y_score = nb_text_clf_best.predict_proba(X_test_roc)
#svm_y_score = svm_text_clf_best.decision_function(X)

y_test = label_binarize(y_test_roc, classes=range(len(y_classes)))

lr_fpr = dict()
lr_tpr = dict()
lr_roc_auc = dict()
for i in idx_classes:
    lr_fpr[i], lr_tpr[i], _ = metrics.roc_curve(y_test[:, i], lr_y_score[:, i])
    lr_roc_auc[i] = metrics.auc(lr_fpr[i], lr_tpr[i])  

nb_fpr = dict()
nb_tpr = dict()
nb_roc_auc = dict()
for i in idx_classes:
    nb_fpr[i], nb_tpr[i], _ = metrics.roc_curve(y_test[:, i], nb_y_score[:, i])
    nb_roc_auc[i] = metrics.auc(nb_fpr[i], nb_tpr[i])

  
plt.figure(figsize=(8,10))
lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'black', 'green'])
plt.subplot(2, 1, 1)
for i, color, name in zip(idx_classes, colors, y_classes):
	plt.plot(lr_fpr[i], lr_tpr[i], color=color, lw=lw, label='{0} (area = {1:0.2f})'.format(name, lr_roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression (C = 5)')
plt.legend(loc="lower right")
plt.subplot(2, 1, 2)
for i, color, name in zip(idx_classes, colors, y_classes):
	plt.plot(nb_fpr[i], nb_tpr[i], color=color, lw=lw, label='{0} (area = {1:0.2f})'.format(name, nb_roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MultinomialNB (alpha = 0.25)')
plt.legend(loc="lower right")
plt.suptitle('ROC Curves of Final Models using 80/20 Train/Test Split')
plt.savefig('viz/ROC_curves.png')
##########################################################
##  END 
##########################################################