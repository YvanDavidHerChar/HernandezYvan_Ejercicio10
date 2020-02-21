import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Codigo Jaime
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
i=20 # este es uno de esos digitos
_ = plt.imshow(imagenes[i])
plt.title('{}'.format(target[i]))
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
# Vamos a hacer un split training test
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# Vamos a entrenar solamente con los digitos iguales a 1
numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]
#La idea es usar el componente principal del PCA realizado sobre los 1, al realizar el producto punto, si el elemento no tiene un 1, este calculo deberia dar 0.

#Codigo Yvan
vectorunos = vectores
#PCA sobre todo train
covTrain = np.cov(x_train.T)
valoresTrain, vectoresTrain = np.linalg.eig(covTrain)
valoresTrain = np.real(valoresTrain)
vectoresTrain = np.real(vectoresTrain)
ii = np.argsort(-valoresTrain)
valoresTrain = valoresTrain[ii]
vectoresTrain = vectoresTrain[:,ii]
#Ahora tengo los vectores principales de los 1 y los vectores principales de todos. El producto punto entre los tres principales de 1 y el punto, proyecto mis datos sobre los vectores principales del 1
Resultados = x_train@vectorunos
Resultados = Resultados**2
numeroTrain = len(y_train)
scoreEsUn1 = np.zeros(numeroTrain)
scoreEsUn1 = np.sqrt(np.sum(Resultados,axis=1))
def esUn1(score,yreal):
    tamanio = len(y_train)
    EsUn1 = np.zeros(tamanio)
    EstaBien = np.zeros(tamanio)
    lametrica = np.mean(score)
    for i in range(tamanio):
        if score[i]>lametrica:
            EsUn1[i] = 1
            if yreal[i] == 1:
                EstaBien[i] = 1
            else:
                EstaBien[i] = 0
        else:
            if yreal[i] == 1:
                EstaBien[i] = 0
            else:
                EstaBien[i] = 1
                
    return EsUn1, EstaBien
a,b =esUn1(scoreEsUn1,y_train)
yconfusio = np.zeros(numeroTrain)
dd = np.where(y_train==1)
yconfusio[dd] = 1
Confusio = confusion_matrix(yconfusio,a)
tn, fp, fn, tp = Confusio.ravel()
P = tp/(fp+tp)
R = tp/(tp+fn)
ParaGraficar = np.array([[tp,fn],[fp,tn]]
                       )



fig, (ax1, ax2)= plt.subplots(nrows=1,ncols=2)
im = ax1.imshow(ParaGraficar)
for i in range(2):
    for j in range(2):
        text = ax1.text(j, i, ParaGraficar[i, j],
                       ha="center", va="center", color="w")
ax1.set_xticks(np.arange(2))
ax1.set_yticks(np.arange(2))
ax1.set_xticklabels(['Positive', 'Negative'])
ax1.set_yticklabels(['True', 'False'])
#
plt.title('Matrix F1 train')
plt.savefig('matriz_de_confusion.png')