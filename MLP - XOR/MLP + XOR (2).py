import numpy as np
import time as t
# Função de ativação sigmóide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Função de perda - Erro Quadrático Médio
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialização dos pesos
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def feedforward(self, X):
        # Camada oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)

        # Camada de saída
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backpropagation(self, X, y, y_pred, learning_rate):
        # Gradiente da camada de saída
        delta2 = (y_pred - y) * sigmoid(self.z2) * (1 - sigmoid(self.z2))
        dW2 = np.dot(self.a1.T, delta2)
        db2 = delta2.sum(axis=0, keepdims=True)

        # Gradiente da camada oculta
        delta1 = np.dot(delta2, self.W2.T) * sigmoid(self.z1) * (1 - sigmoid(self.z1))
        dW1 = np.dot(X.T, delta1)
        db1 = delta1.sum(axis=0)

        # Atualização dos pesos
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        t0 = t.time()
        for i in range(epochs):
            # Feedforward
            y_pred = self.feedforward(X)

            # Backpropagation
            self.backpropagation(X, y, y_pred, learning_rate)

            # Cálculo do erro
            loss = mse_loss(y, y_pred)

            # Impressão do erro a cada 1000 épocas
            if i % 1000 == 0:
                print(f"Epoch {i}: loss = {loss:.4f}")
        tf = t.time()
        return tf-t0

# Problema do XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Treinamento do modelo
model = NeuralNetwork(input_size = 2, hidden_size = 3, output_size = 1)
print(model.train(X= X,y= y, epochs= 10000, learning_rate= 0.1))

# Chamada da execucao do algoritmo treinado
print(model.feedforward([0,0]))
print(model.feedforward([0,1]))
print(model.feedforward([1,0]))
print(model.feedforward([1,1]))
