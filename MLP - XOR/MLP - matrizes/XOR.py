import numpy as np
import MultilayerPerceptron

# Problema do XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Treinamento do modelo
model = MultilayerPerceptron.MLP(tamanho_entrada = 2, tamanho_escondida = 3, tamanho_saida = 1)
print(model.train(X= X,y= y, epochs= 10000, taxa_aprendizado= 0.1))

# Chamada da execucao do algoritmo treinado
a = model.feedforward([0,0])
b = int(np.round(a[0][0]))
print("Saída modelo: %.3f / Saída interpretada: %i" %(a,b))
a = model.feedforward([0,1])
b = int(np.round(a[0][0]))
print("Saída modelo: %.3f / Saída interpretada: %i" %(a,b))
a = model.feedforward([1,0])
b = int(np.round(a[0][0]))
print("Saída modelo: %.3f / Saída interpretada: %i" %(a,b))
a = model.feedforward([1,1])
b = int(np.round(a[0][0]))
print("Saída modelo: %.3f / Saída interpretada: %i" %(a,b))