
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Flatten, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
import tensorflow as tf
from keras.datasets import cifar10
from matplotlib import pyplot
import sys

#from tensorflow.keras.applications import VGG16
#vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

#argando set de entrenamiento
from sklearn.model_selection import train_test_split
import glob, cv2, numpy as np
from tqdm import tqdm
categorias = ['P1','P2','P3','P4']
X = []
Y = []
for categoria in tqdm(categorias):  
  ruta_imgs = glob.glob('./Imagenes_7a/'+ categoria +'/*.png')
  for ruta_img in ruta_imgs:
    img = cv2.resize(cv2.cvtColor(cv2.imread(ruta_img), cv2.COLOR_RGB2BGR), (150, 150))
    X.append(img)
    if categoria == 'P1':
      Y.append(0)
    elif categoria == 'P2':
      Y.append(1)
    elif categoria == 'P3':
      Y.append(2)    
    else:
      Y.append(3)
X = np.asarray(X).astype('uint8')
Y = np.expand_dims(np.asarray(Y).astype('uint8'), axis = 1)
x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=42)

print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

for i in range(9):
  pyplot.subplot(330 + 1 + i)
  pyplot.imshow(x_train[i])
pyplot.show()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
print(y_train[0])

from tensorflow.keras.applications import DenseNet121

vgg_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
vgg_model.summary()

for layer in vgg_model.layers[:426]:  #MobilNet 85, MobileNet 86 DenseNet121 427
    layer.trainable = False
for i, layer in enumerate(vgg_model.layers):
    print(i, layer.name, layer.trainable)

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense
from keras.layers import Dropout, Lambda  #.core
from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D #.convolutional
from keras.layers import MaxPooling2D
#from keras.layers.merge import concatenate
from keras import optimizers
from keras.layers import BatchNormalization

x = vgg_model.output
x = Flatten()(x)
x= BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.6)(x)
x = Dense(4, activation='softmax')(x)
vgg_custom_model=Model(inputs=vgg_model.input, outputs=x)
learning_rate= 1e-4
vgg_custom_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=["accuracy"])
vgg_custom_model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test))

target_names = ['P1', 'P2', 'P3', 'P4']
y_pred = vgg_custom_model.predict(x_test, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
from sklearn.metrics import classification_report, confusion_matrix
reporte = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
print(reporte)

#from sklearn.metrics import cohen_kappa_score
#cohen_kappa_score(y_pred, y_test)

#target_names = ['gato', 'perro']
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd, numpy as np
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
cm = pd.DataFrame(cm,  range(4),range(4))
plt.figure(figsize = (10,10))
sns.heatmap(cm, annot=True, annot_kws={"size": 14},fmt="d",linewidths=.5,xticklabels=target_names, yticklabels=target_names,cmap="YlGnBu" ) # font size
plt.show()