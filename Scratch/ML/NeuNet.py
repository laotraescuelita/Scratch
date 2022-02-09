import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
fig, axes = plt.subplots(1, df.shape[1], figsize=(10, 4))
for i in range( df.shape[1]):
    axes[i].hist( df.iloc[:,i] )    
plt.show()

#Escribimos las funciones de activacion que podemos utilizar
def tanh( x ):
    return np.tanh( x )

def relu( x ):
    return np.maximum( x, 0 )

def sigmoid( x ):
    return 1/( 1 + np.exp(-x))

def softmax( x ):
    expx = np.exp( x )
    return expx/ np.sum( expx, axis = 0 )
    #return expx/ np.sum( expx)

#Derivadas de las funciones de activacion

def derivative_tanh( x ):
    return ( 1 - np.power( x, 2 ))

def derivative_relu( x ):
    return np.array( x > 0, dtype=np.float32)

def derivative_sigmoid( x ):
    return sigmoid( x ) * ( 1 - sigmoid( x ))


#Construir nuestro modelo en este caso es "NEural Network"
class NeuralNetwork:
    def __init__( self, x, y, lr, iter_, nx, nh, ny):
        self.x = x 
        self.y = y 
        self.lr = lr 
        self.iter = iter_
        self.cost_list = []
        self.nx = nx 
        self.nh = nh 
        self.ny = ny 
    
    #Vamos a iniciar los pesos para la red y el sesgo
    def init_weights_bias( self ):
        w1 = np.random.rand( self.nh, self.nx)
        b1 = np.random.rand( self.nh,1 )
        w2 = np.random.rand( self.ny, self.nh )
        b2 = np.random.rand( self.ny,1 )

        weights_bias = {
            "w1":w1,
            "b1":b1,
            "w2":w2,
            "b2":b2
        }

        return weights_bias

    def forward_propagation(self, weights_bias):
        #Multiplicamos los pesos por la entradas y sumamos el sesgo
        z1 = np.dot( weights_bias["w1"], self.x) + weights_bias["b1"]
        #Esos resultados son el eje x para la función de activación. En este caso utilizamos relu aunque podemos utilizar tanh.
        a1 = relu( z1 )
        #Multiplicamos el resultado de a1 por los pesos y sumamos el sesgo
        z2 = np.dot( weights_bias["w2"], a1) + weights_bias["b2"]
        #A ese resultado le aplicamos la función sigmoide por ser una red de clasificacion binaria.
        a2 = sigmoid( z2 )

        forward = {
        "z1":z1,
        "a1":a1,
        "z2":z2,
        "a2":a2
        }

        return forward

    
    #Una vez que hemos obteido las predicciones de forward propagatin hay que medirals para minmizar el costo
    def backward_propagation( self, weights_bias, forward):
        w1 = weights_bias["w1"]
        b1 = weights_bias["b1"]
        w2 = weights_bias["w2"]
        b2 = weights_bias["b2"]
        a1 = forward["a1"]
        a2 = forward["a2"]

        m = self.x.shape[1]

        #Necesitamos las derivadas parciales de la función costo para poder aplicar el "gradiant descent"
        dz2 = ( a2 - self.y )
        dw2 = (1/m)*np.dot(dz2,a1.T)
        db2 = (1/m)*np.sum(dz2, axis=1, keepdims=True)

        dz1 = (1/m)*np.dot(w2.T, dz2)*derivative_relu(a1)
        dw1 = (1/m)*np.dot(dz1,self.x.T)
        db1 = (1/m)*np.sum(dz1, axis=1, keepdims=True)

        backward = {
        "dw1":dw1,
        "db1":db1,
        "dw2":dw2,
        "db2":db2
        }

        return backward
    
    def update_weights_bias(self, weights_bias, backward):
        w1 = weights_bias["w1"]
        b1 = weights_bias["b1"]
        w2 = weights_bias["w2"]
        b2 = weights_bias["b2"]

        w1 = w1 - self.lr * backward["dw1"]
        b1 = b1 - self.lr * backward["db1"]
        w2 = w2 - self.lr * backward["dw2"]
        b2 = b2 - self.lr * backward["db2"]

        weights_bias = {
        "w1":w1,
        "b1":b1,
        "w2":w2,
        "b2":b2
        }

        return weights_bias
    
    def cost_function(self, a2 ):
        m = self.y.shape[0]
        #cost = -(1/m)*np.sum( y*np.log( a2 ))
        cost = -(1/m)*np.sum( self.y*np.log(a2) + (1-self.y)*np.log(1-a2))
        return cost


    def train( self ):
        
        weights_bias = self.init_weights_bias()

        for i in range( self.iter ):
            forward = self.forward_propagation( weights_bias)
            cost = self.cost_function( forward["a2"])
            backward = self.backward_propagation( weights_bias, forward)
            weights_bias = self.update_weights_bias( weights_bias, backward)

            self.cost_list.append( cost )

            if( i%(iter_/10) == 0):
                print( "Cost after: ", i, " iterations is: ", cost )

        return weights_bias, forward, self.cost_list

    def predict( self, a2 ):
        return np.argmax( a2 , 0) 

    def accuracy(self, y_hat, y):        
        return np.sum( y_hat == y) / y.size

    
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
lr = .0001
iter_ = 200
nx = xtrain.shape[1]
ny = ytrain.shape[1]
nh = 2


neunet = NeuralNetwork(xtrain.T, ytrain.T, lr, iter_, nx, nh, ny)
weights, forward, cost_list = neunet.train()
iter_range = np.arange( iter_ )
plt.plot( iter_range, cost_list )
plt.show()

y_hat = neunet.predict(forward["a2"])
accuracy = neunet.accuracy(y_hat, ytest)
print( y_hat )
print( accuracy )


#Entrenemos el modelo  con una libreria de sklearn 
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
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=0)
mlp.fit(xtrain, ytrain)
print("Accuracy on training set: {:.2f}".format(mlp.score(xtrain, ytrain)))
print("Accuracy on test set: {:.2f}".format(mlp.score(xtest, ytest)))

mlp = MLPClassifier(random_state=0, hidden_layer_sizes=[10])
mlp.fit(xtrain, ytrain)
print("Accuracy on training set: {:.2f}".format(mlp.score(xtrain, ytrain)))
print("Accuracy on test set: {:.2f}".format(mlp.score(xtest, ytest)))

