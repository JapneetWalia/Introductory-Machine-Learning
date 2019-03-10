# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:53:43 2019

@author: Japneet Singh
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


df=pd.read_csv("C:/Users/Japneet Singh/Desktop/7th March/diabetes.csv")

print(df.columns)

df_data = df.iloc[:,0:8]
df_label = df['Outcome']

df_data_train, df_data_test, df_label_train, df_label_test = train_test_split(df_data, df_label, test_size=0.4)

Model = KNeighborsClassifier(n_neighbors = 5)
Model.fit(df_data_train, df_label_train)
y_pred = Model.predict(df_data_test)

print("Accuracy is:", accuracy_score(df_label_test, y_pred)*100)
print("Confusion Matrix :\n ", confusion_matrix(df_label_test, y_pred))