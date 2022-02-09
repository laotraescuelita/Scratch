import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

# importar de sklearn el modulo para crear de manera artificial matrices para clasificación.
from sklearn.datasets import make_regression
# Generar la matriz con variables numericas y el vector a predecir
matriz, vector, coefficients = make_regression(
n_samples = 100,
n_features = 3,
n_informative = 3,
n_targets = 1,
noise = 0.0,
coef = True,
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
print( "Tipo de dato del dataframe \n", df.info() )

#Mostrar la forma que tienen las variables 
fig, axes = plt.subplots(1, df.shape[1], figsize=(15, 4))
for i in range( df.shape[1]):
    axes[i].hist( df.iloc[:,i] )    
plt.show()

#Construir nuestro modelo en este caso es "logistic regression"
class LinearRegression:
    def __init__( self, x, y, lr, iter_):
        self.x = x
        self.y = y
        self.lr = lr
        self.iter = iter_
        #Inicializmos los pesos 
        self.weights = np.zeros( (self.x.shape[1], 1) )
        self.cost = 0
        self.cost_list = []

    def train( self ):
        m = self.x.shape[0]        

        for i in range( self.iter):
            #vamos a predicir el vector con los valores de la data de entrenamiento
            y_pred = np.dot( self.x, self.weights )
            #Creamos la función de costo
            self.cost = ( 1/(2*m))*np.sum( np.square( y_pred - self.y ))
            #Calculamos la derivad del costo con respecto a los pesos
            dcost_dweights = (1/m)*np.dot(self.x.T, y_pred-self.y)
            #Actualizamos los pesos de la data esperamos se reduzcan , este es el "gradient descent"
            self.weights = self.weights - self.lr * dcost_dweights
            #vamos a guaradar los costos en caditeracion
            self.cost_list.append( self.cost )
            
            if (i%(self.iter/10) == 0):
                print( "Costo después de ", i, " ietración : ", self.cost)
                

        return self.weights, self.cost_list

    def predict( self, x):
        y_predict =  np.dot( x , self.weights)
        return y_predict

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
lr = 0.001
iter_ = 1000
linreg = LinearRegression( xtrain, ytrain, lr, iter_)
weigths, costlist = linreg.train()

rng = np.arange( 0, iter_)
plt.plot( costlist, rng )
plt.show()

#Utilizemos el test set para predecir 
yhat = linreg.predict(xtest)

fig, axes = plt.subplots(1,2, figsize=(15,4) )
#Mostrar la forma que tienen las variables 
axes[0].scatter( xtest[:,1], ytest )
axes[1].scatter( xtest[:,1], yhat )
plt.show()

#Entrenemos el modelo 
#Así que primero nos quedamos con los valores numericos del dataframe.
x = df.iloc[:,0:3].values
y = df.iloc[:,-1].values
#Dividimos las variables para tener una parte en trainig y otra en testing 
X_train = x[:70]
X_test = x[70:]
y_train = y[:70]
y_test = y[70:]
#Verificamso al dimensiones de los objetos
print( "Dimensiones de xtrain \n", xtrain.shape)
print( "Dimensiones de xtest \n", xtest.shape)
print( "Dimensiones de ytrain \n", ytrain.shape)
print( "Dimensiones de ytest \n", ytest.shape)

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X_train, y_train)
#print("lr.coef_: {}".format(lr.coef_))
#print("lr.intercept_: {}".format(lr.intercept_))
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(ridge.coef_ != 0)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(ridge10.coef_ != 0)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(ridge01.coef_ != 0)))


from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

# we increase the default setting of "max_iter",
# otherwise the model would warn us that we should increase max_iter.
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

