import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Cargar el set de entrenamiento
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,-1].values

#Separacion de sets(entrenamiento y prueba)
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.25, random_state=30)

#Escalado para que todos tengan una media en 0 y la desviación estandar en 1
# Y todos tengan una proporcion similar
from sklearn.preprocessing import StandardScaler
trans=StandardScaler()
Xtrain=trans.fit_transform(Xtrain)
Xtest=trans.transform(Xtest)

#K_NN
from sklearn.neighbors import KNeighborsClassifier
classi=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classi.fit(Xtrain,Ytrain)


#Prueba
ypre=classi.predict(Xtest)
print(ypre)
print(Ytest)

from matplotlib.colors import ListedColormap
X_set,y_set=Xtrain,Ytrain
X1,X2= np.meshgrid(np.arange(start=X_set[:,0].min() -1, stop=X_set[:,0].max()+1,step=0.1),
                   np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1,step=0.1))
plt.contourf(X1,X2,classi.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha=0.5,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('K-NN(SET DE ENTRENAMIENTO)')
plt.xlabel('años')
plt.ylabel('salario estimado')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set,y_set=Xtest,Ytest
X1,X2= np.meshgrid(np.arange(start=X_set[:,0].min() -1, stop=X_set[:,0].max()+1,step=0.1),
                   np.arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1,step=0.1))
plt.contourf(X1,X2,classi.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
            alpha=0.5,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
plt.title('K-NN(SET DE ENTRENAMIENTO)')
plt.xlabel('años')
plt.ylabel('salario estimado')
plt.legend()
plt.show()