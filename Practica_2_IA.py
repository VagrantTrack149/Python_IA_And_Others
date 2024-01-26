# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:26:26 2023

@author: NeilO
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# Load the dataset
dataset = pd.read_csv('Bean_Dataset.csv')
X = dataset.iloc[:, 0:16].values
Y = dataset.iloc[:, -1].values

# Standardize the input features
sc = StandardScaler()S
X = sc.fit_transform(X)

# Encode the target variable
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
Y_onehot = to_categorical(Y_encoded, num_classes=7)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=0)

# Build the neural network model
bean = tf.keras.models.Sequential()
bean.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=16))
bean.add(tf.keras.layers.Dense(units=15, activation='relu'))
bean.add(tf.keras.layers.Dense(units=7, activation='softmax'))

# Compile the model
bean.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
bean.fit(X_train, Y_train, batch_size=128, epochs=64)

# Evaluate the model
y_pred = bean.predict(X_test)
score = bean.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Convert one-hot encoded predictions and true labels back to categorical labels for confusion matrix
y_pred_labels = np.argmax(y_pred, axis=1)
Y_test_labels = np.argmax(Y_test, axis=1)

# Display confusion matrix
print('Confusion matrix:\n', confusion_matrix(Y_test_labels, y_pred_labels))


