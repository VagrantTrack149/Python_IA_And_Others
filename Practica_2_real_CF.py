import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Cargar el conjunto de datos MNIST
mnist = fetch_openml('mnist_784') #Cargar datos para numpy
X = mnist.data.astype('float32') #Datos brutos en float
y = mnist.target.astype('int64') #Etiquetas de los datos

# Dividir los datos en 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalizar los datos, 0 o 1 para facilitar el calculo
X_train /= 255.
X_test /= 255.

# Convertir las etiquetas a one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Crear modelo de red neuronal
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(784,)),
  tf.keras.layers.Dense(units=50, activation='sigmoid'), #oculta 1
  tf.keras.layers.Dense(units=20, activation='sigmoid'), #oculta 2
  tf.keras.layers.Dense(units=10, activation='sigmoid')  #salida
])

# Compilar modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#binary_crossentropy
#categorical_crossentropy
# Entrenar modelo
history = model.fit(X_train, y_train, epochs=20, batch_size=128)

# Evaluar modelo
y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Confusion matrix:\n', confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
