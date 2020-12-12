# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:49:50 2020

@author: alons
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





yeastData = pd.read_csv(r"C:\Users\alons.DESKTOP-354NE3C\Downloads\Trabajo3\Yeast\yeast.data",sep='\s+', header=None );
yeastData.columns = ['Name','mcg','gvh','alm','mit','erl','pox','vac','nuc','Class' ]

yeastData = yeastData.drop('Name',axis=1)


X = yeastData.drop('Class', axis=1)
y = yeastData['Class']

print(X)
print(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()

X_train = st_x.fit_transform(X_train)
X_test = st_x.transform(X_test)

from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
cMatrix = confusion_matrix(y_test,y_pred)
cReport= classification_report(y_test,y_pred)

