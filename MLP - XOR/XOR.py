import numpy as np
import MLP

# Criacao do modelo
modelo = MLP.MultilayerPerceptron(nn_entrada = 2, nn_saida = 1,nn_escondida = 2)

# Dados de entrada para treinamento
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Chamando algoritmo de treinamento
modelo.treinamento(X, y, n_epocas = 100, taxa_aprendizado = 0.1)

# Chamada da execucao do algoritmo treinado
print(modelo.rodar([0,0]))
print(modelo.rodar([0,1]))
print(modelo.rodar([1,0]))
print(modelo.rodar([1,1]))