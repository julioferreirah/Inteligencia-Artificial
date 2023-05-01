import Perceptron
import numpy as np

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