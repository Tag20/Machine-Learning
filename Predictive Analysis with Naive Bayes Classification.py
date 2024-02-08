# -*- coding: utf-8 -*-
"""ROC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/188tzcrJB7lAMvXbqCTO7uYp_vJQ3rHLh
"""


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn import metrics

from google.colab import files

uploaded = files.upload()

data = pd.read_csv('/content/Book1.csv',encoding='unicode_escape')

data.describe

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['age'] = label_encoder.fit_transform(data['age'])
data['income'] = label_encoder.fit_transform(data['income'])
data['student'] = label_encoder.fit_transform(data['student'])
data['credit_rating'] = label_encoder.fit_transform(data['credit_rating'])
data['buys_computer'] = label_encoder.fit_transform(data['buys_computer'])

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

data

from sklearn.naive_bayes import MultinomialNB, ComplementNB
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.26, random_state=1)
model = MultinomialNB()
model.fit(X_train, y_train)

preds = model.predict(X_test)

metrics.confusion_matrix(y_test, preds)

print(metrics.classification_report(y_test,preds))

preds

metrics.RocCurveDisplay.from_predictions(y_test, preds, color='orange')

twomodel = ComplementNB()
twomodel.fit(X_train, y_train)

preds = model.predict(X_test)

metrics.confusion_matrix(y_test, preds)

print(metrics.classification_report(y_test,preds))

preds

metrics.RocCurveDisplay.from_predictions(y_test, preds, color='orange')

