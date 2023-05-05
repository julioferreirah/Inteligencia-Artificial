import numpy as np
import time as t
# Função de ativação sigmóide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Função de perda - Erro Quadrático Médio
def erro_quadratico(y_dado, y_predito):
    return ((y_dado- y_predito) ** 2).mean() # média do quadrado de cada erro de uma entrada

class MLP:
    def __init__(self, tamanho_entrada, tamanho_escondida, tamanho_saida):
        # Inicialização dos pesos
        self.W1 = np.random.randn(tamanho_entrada, tamanho_escondida) # pesos primeira camada
        self.b1 = np.zeros((1, tamanho_escondida)) # bias primeira camada
        self.W2 = np.random.randn(tamanho_escondida, tamanho_saida) # pesos segunda camada
        self.b2 = np.zeros((1, tamanho_saida)) # bias segunda camada

    def feedforward(self, X):
        # Camada oculta
        self.z1 = np.dot(X, self.W1) + self.b1 # multiplicação de matriz equivale a soma ponderada
        self.a1 = sigmoide(self.z1) # funcao de ativacao

        # Camada de saída
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoide(self.z2)

        return self.a2

    def backpropagation(self, X, y, y_predito, taxa_aprendizado):
        # Gradiente da camada de saída
        delta2 = (y_predito - y) * sigmoide(self.z2) * (1 - sigmoide(self.z2)) #erro + direção inversa (equivalente a derivada)
        dW2 = np.dot(self.a1.T, delta2) # 'peso' do erro (+direcao) na matriz de peso
        db2 = delta2.sum(axis=0, keepdims=True) # 'peso' do erro no bias

        # Gradiente da camada oculta
        delta1 = np.dot(delta2, self.W2.T) * sigmoide(self.z1) * (1 - sigmoide(self.z1))
        dW1 = np.dot(X.T, delta1)
        db1 = delta1.sum(axis=0)

        # Atualização dos pesos
        self.W2 -= taxa_aprendizado * dW2 #aplicacao da taxa de aprendizado ao erro direcionado ao inverso 
        self.b2 -= taxa_aprendizado * db2
        self.W1 -= taxa_aprendizado * dW1
        self.b1 -= taxa_aprendizado * db1

    def train(self, X, y, epochs, taxa_aprendizado):
        t0 = t.time()
        
        for i in range(epochs):
            # Feedforward
            y_predito = self.feedforward(X)

            # Backpropagation
            self.backpropagation(X, y, y_predito, taxa_aprendizado)

            # Cálculo do erro
            loss = erro_quadratico(y, y_predito)

            # Impressão do erro a cada 1000 épocas
            if i % 1000 == 0:
                print(f"Epoch {i}: loss = {loss:.4f}")

        tf = t.time()
        return tf-t0

