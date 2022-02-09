import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


# importar de sklearn el modulo para crear de manera artificial matrices para clasificación.
from sklearn.datasets import make_classification
# Generar la matriz con variables numericas y el vector a predecir
matriz, vector = make_classification(n_samples = 100,
n_features = 3,
n_informative = 3,
n_redundant = 0,
n_classes = 2,
weights = [.25, .75],
random_state = 1)

#Convertirlo en un objeto dataframe para realizar el proceso de limpiar, transfomar y reducir en caso de ser necesario.
#Agregar la matriz a und ataframe
df = pd.DataFrame( matriz , columns= ["x1","x2","x3"])
print("Matriz agrgada en dataframe \n", df.head())
#Agregar la variable a clasificar
df["target"] = vector
print("Vector agregado al dataframe \n", df.head())
print( "Estdisticas del dataframe \n", df.describe() )
print( "Mostrar el numero datos perdidos por variable\n", df.isnull().sum())

#Mostrar la forma que tienen las variables 
fig, axes = plt.subplots(1, df.shape[1], figsize=(15, 4))
for i in range( df.shape[1]):
    axes[i].hist( df.iloc[:,i] )    
plt.show()


#Construir nuestro modelo en este caso es "logistic regression"
class LogisticRegression:
    def __init__( self, x, y, lr, iter_):
        self.x = x 
        self.y = y 
        self.lr = lr 
        self.iter = iter_
        self.cost_list = []
        #Iniciar los  pesos y el parametro "bias" 
        self.m = self.x.shape[0]
        self.n = self.x.shape[1]
        self.weights = np.zeros( (self.n,1))
        self.bias = 0 
    
    #Aqui construimos la funcion de activacion en este caso es sigmoide por ser una clasificación
    def sigmoid(self, x ):
        return 1 / ( 1 + np.exp(-x) )
    
    def train( self ):
        #Vamos a enrenar el modelo con el numero de iteracione que nos pasen por parametro
        for i in range( self.iter ):
            #Esta es una multiplicacion de matrices ( w1x1 + w2x2 + ... + w3x3) + bias
            z = np.dot( self.x, self.weights ) + self.bias
            #El resultado de z son los x para la funcion sigmoide que dibujara una s y pondra los valores entre 1 y -1 
            a = self.sigmoid( z )
            
            #Creamos la función de costo
            cost = -( 1/ self.m ) * np.sum( self.y * np.log( a ) + (1 - self.y )*np.log (1 - a) )
            #Necestamos la derivada parcial del costo con respecto a los pesos
            dcost_dw = ( 1 / self.m ) * np.dot( self.x.T , a - self.y )
            #Necesitamos la derivada parcial del costo con respecto al sesgo.
            dcost_db = ( 1 / self.m ) * np.sum( a - self.y )
            
            #Aqui utilizamos el "gradient descent" para actualizar los pesos y el sesgo con cad iteración.
            
            self.weights = self.weights - lr * dcost_dw
            self.bias = self.bias - lr * dcost_db
                        
            #Vamos a almacener los costos durante la iteración. Lo que esperamos es que se vaya reduciendo con cada iteración.
            self.cost_list.append( cost )
            
            if (i%(self.iter/10) == 0):
                print( "Cost0 después de ", i, " ietración : ", cost)
                
        return self.weights, self.bias, self.cost_list
    
    def predict(self, x):
        yhat = np.dot( x , self.weights)
        for i in range(len(yhat)):
            if yhat[i] < 0.5:
                yhat[i] = 0
            else:
                yhat[i] = 1
        return yhat



#Entrenemos el modelo 
#Así que primero nos quedamos con los valores numericos del dataframe.
x = df.iloc[:,0:3].values
y = df.iloc[:,-1].values
#Dividimos las variables para tener una parte en trainig y otra en testing 
xtrain = x[:70]
xtest = x[70:]
ytrain = y[:70]
ytest = y[70:]
#Verificamso al dimensiones de los objetos
print( "Dimensiones de xtrain \n", xtrain.shape)
print( "Dimensiones de xtest \n", xtest.shape)
print( "Dimensiones de ytrain \n", ytrain.shape)
print( "Dimensiones de ytest \n", ytest.shape)

#para no tener problemas con la variable a predecir en sus dimensiones la voy a volver un array.
ytrain = np.array( ytrain).reshape(ytrain.shape[0],1)
ytest = np.array( ytest).reshape(ytest.shape[0],1)
print( "Dimensiones de ytrain \n", ytrain.shape)
print( "Dimensiones de ytest \n", ytest.shape)

#Pasemos los parametros 
lr = .01
iter_ = 1000
logreg = LogisticRegression(xtrain, ytrain, lr, iter_)
weights, bias, cost_list = logreg.train()
iter_range = np.arange( iter_ )
plt.plot( iter_range, cost_list )
plt.show()

#Utilizemos el test set para predecir 
yhat = logreg.predict( xtest )
print( "El modelo predijo de manera correcta : ", np.sum( yhat == ytest ) / ytest.shape[0], "%")


#Entrenemos el modelo 
#Así que primero nos quedamos con los valores numericos del dataframe.
x = df.iloc[:,0:3].values
y = df.iloc[:,-1].values
#Dividimos las variables para tener una parte en trainig y otra en testing 
xtrain = x[:70]
xtest = x[70:]
ytrain = y[:70]
ytest = y[70:]
#Verificamso al dimensiones de los objetos
print( "Dimensiones de xtrain \n", xtrain.shape)
print( "Dimensiones de xtest \n", xtest.shape)
print( "Dimensiones de ytrain \n", ytrain.shape)
print( "Dimensiones de ytest \n", ytest.shape)

#Utilizemos la libreria sklearn
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression().fit(xtrain, ytrain)
print("predicción en training: {:.3f}".format(logreg.score(xtrain, ytrain)))
print("predicción en testing: {:.3f}".format(logreg.score(xtest, ytest)))

logreg100 = LogisticRegression(C=100).fit(xtrain, ytrain)
print("predicción en training: {:.3f}".format(logreg100.score(xtrain, ytrain)))
print("predicción en testing: {:.3f}".format(logreg100.score(xtest, ytest)))

logreg001 = LogisticRegression(C=0.01).fit(xtrain, ytrain)
print("predicción en training: {:.3f}".format(logreg001.score(xtrain, ytrain)))
print("predicción en testing: {:.3f}".format(logreg001.score(xtest, ytest)))


