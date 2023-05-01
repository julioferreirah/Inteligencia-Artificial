import numpy as np # operações 

def step(x):
    if x > 0:
        return 1
    else:
        return 0
    
class Perceptron:
    def __init__(self, n_entrada):
        #self.pesos = [n_entrada + 1] # array para armazenar os pesos, incluindo o bias
        #for i in range(n_entrada):
        #    self.pesos[i] = np.random.random_sample() # inicializando pesos com numeros aleatorios de 0 a 1
        self.pesos = np.random.random_sample(n_entrada + 1)

    def aplicacao(self, x): # x e array de entrada
        Yin = np.dot(x,self.pesos[1:]) + self.pesos[0] # soma ponderada dos pesos + bias
        Y = step(Yin) # mapeamento com a funcao de ativacao
        return Y
       
    def treinamento(self, X, y, n_epocas, taxa_aprendizado): # X e matriz de entrada e y array de resultados esperados
        for epoca in range(n_epocas):
            for i in range(X.shape[0]): #para cada dados de entrada passado
                y_est = self.aplicacao(X[i]) #feedfoard
                erro = y[i] - y_est
                #ajustes de pesos:
                self.pesos[0] += erro*taxa_aprendizado  #bias
                self.pesos[1:] += erro*taxa_aprendizado*X[i]
###################
# PROBLEMA DO AND #
###################

# Criacao do modelo
modelo = Perceptron(n_entrada = 2)

# Dados de entrada para treinamento
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Chamando algoritmo de treinamento
modelo.treinamento(X, y, n_epocas = 15, taxa_aprendizado = 0.1)

# Chamada da execucao do algoritmo treinado
print(modelo.aplicacao([0,0]))
print(modelo.aplicacao([0,1]))
print(modelo.aplicacao([1,0]))
print(modelo.aplicacao([1,1]))

# Teste em escala
for i in range(30):
    if(modelo.aplicacao([0,0]) != 0): print("!")
    if(modelo.aplicacao([0,1]) != 0): print("!")
    if(modelo.aplicacao([1,0]) != 0): print("!")
    if(modelo.aplicacao([1,1]) != 1): print("!")

###################
# PROBLEMA DO OR #
###################

# Criacao do modelo
modelo = Perceptron(n_entrada = 2)

# Dados de entrada para treinamento
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

# Chamando algoritmo de treinamento
modelo.treinamento(X, y, n_epocas = 15, taxa_aprendizado = 0.1)

# Chamada da execucao do algoritmo treinado
print(modelo.aplicacao([0,0]))
print(modelo.aplicacao([0,1]))
print(modelo.aplicacao([1,0]))
print(modelo.aplicacao([1,1]))

# Teste em escala
for i in range(30):
    if(modelo.aplicacao([0,0]) != 0): print("!")
    if(modelo.aplicacao([0,1]) != 1): print("!")
    if(modelo.aplicacao([1,0]) != 1): print("!")
    if(modelo.aplicacao([1,1]) != 1): print("!")