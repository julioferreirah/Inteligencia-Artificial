import numpy as np # operações 

def step(x):
    if x > 0:
        return 1
    else:
        return 0
    
class Perceptron:
    def __init__(self, n_entrada):
        self.pesos = [n_entrada + 1] # array para armazenar os pesos, incluindo o bias
        for i in range(n_entrada + 1):
            self.pesos[i] = np.random.random_sample() # inicializando pesos com numeros aleatorios de 0 a 1

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
