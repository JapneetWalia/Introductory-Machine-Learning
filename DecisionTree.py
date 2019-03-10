# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:19:08 2019

@author: Japneet Singh
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


from sklearn import datasets

df= datasets.load_iris()

print(df.data)
print(df.target)

x_train, x_test, y_train, y_test = train_test_split(df.data,df.target, test_size=0.3)

clf_gini = DecisionTreeClassifier(criterion = "gini")
clf_gini.fit(x_train, y_train)

y_pred = clf_gini.predict(x_test)
print(y_pred)
print("Accuracy is:", accuracy_score(y_test, y_pred)*100)

print("Confusion Matrix :\n ", confusion_matrix(y_test, y_pred))
