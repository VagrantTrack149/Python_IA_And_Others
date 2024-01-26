#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 10:09:39 2022

@author: mac1
"""
# Artificial Neural Network

# Importacion de Librerias
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
tf.__version__

# Part 1 - Pre-porcesamiento de datos

# importar set de datos
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Codificando dato categorico
#  "Gender" 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# Codificacion One Hot  "Geography" 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Scalado de variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)

# Separacion de set entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Part 2 -Construccion  ANN

# Inicializacion ANN
ann = tf.keras.models.Sequential()

# primera capa oculra
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# segunda capa oculta
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))

# Capa de Salida
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Entrenamiento ANN

# Compilar ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Entrenamiento con set numero de muestras para actualizar pesos
ann.fit(X_train, y_train, batch_size =32 , epochs = 100)

# Part 4 - Clasificacion 

# Prediccion con set de prueba
y_pred = ann.predict(X_test).round()


# Contruyendo Mztriz de Confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False,True])
cm_display.plot()
plt.show()



from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label='AUR='+str(auc)) #% (auc))#label=r"AUC=")
plt.legend()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid()
plt.show()

 
