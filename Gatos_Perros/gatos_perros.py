# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:08:17 2023

@author: NeilO
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
#Part 1 creación
#Generación de set de entrenamiento
train_datagen=ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
#Genración de set de prueba
test_datagen=ImageDataGenerator(rescale=1./255)
#Generación de set de entrenamiento
training_set=train_datagen.flow_from_directory('training_set',target_size=(64,64),batch_size=32,class_mode='binary')
#Generación de set de prueba
test_set=test_datagen.flow_from_directory('test_set',target_size=(64,64),batch_size=32,class_mode='binary')

#part 2 Bulding CNN - Convolucional Neuronal network
#Inicializar
cnn=tf.keras.models.Sequential()
#Convolución 1-capa
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding="same",activation="relu",input_shape=[64,64,3]))
#pooling 2-capa
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='valid'))
#Convlución 3-capa
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding="same",activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2,padding='valid'))
#Aplanado
cnn.add(tf.keras.layers.Flatten())
#Conexión capa densa 4-capa 
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=50,activation='relu'))
#capa de salida 5-capa
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#Part 3- entranar CNN
#Compilar
cnn.compile(optimizer='adam',loss='binary_cossentropy',metrics=['accuracy'])
#Entrenar cnn con el set de entrenamiento y test_set se prueba
cnn.fit(training_set,
        steps_per_epoch=250,
        epoch=25,
        validation_data=test_set,
        validation_set=62)
#steps_per_epoch=len(Xtrain)//batch_size
#validation_steps=len(Xtest)//batch_size

import joblib
joblib.dump(cnn,'Modelo_CNN.pkl')
pred=cnn.predict(test_set,batch_size=(32))
pred2=(pred>0.5).astype(int)

from sklearn.metrics import confusion_matrix
import numpy as np
y=np.copy(test_set.labels)
cm=confusion_matrix(y,pred2)
print(cm)