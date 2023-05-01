import numpy as np

class MLP:
    def __init__(self, nn_entrada, nn_saida,nn_escondida):
        self.pesos1 = np.random.randn(nn_entrada, nn_escondida) # matris de pesos - cada neuronio de entrada vai para cada um na camada escondida
        self.bias1 = np.zeros(nn_escondida) #vetor de pesos bias pra cada neuronio da camada escondida
        self.pesos2 = np.random.randn(nn_escondida, nn_saida) # matris de pesos - cada neuronio da camada escondida vai para cada um na camada de saida
        self.bias2 = np.zeros(nn_saida)
    
    def sigmoide(self, x): # funcao de ativacao - retorna valor de (0,1) e tem derivada em todos os pontos
        return 1/(1+np.exp(-x))
    
    def derivada_sigmoide(self, x): # precisa da derivada para saber a direcao do erro e fazer a correcao dos pesos no treinamento
        return x*(1-x)
    
    def feedfoard(self, x): #recebe um vetor de tamanho = nn_entrada 
        YinE = np.dot(x, self.pesos1) + self.bias1 # soma ponderada primeira camanda
        YE = self.sigmoide(YinE) # saida camada escondida
        YinS = np.dot(YE, self.pesos2) + self.bias2 # soma ponderada primeira camanda
        YS = self.sigmoide(YinS) #saida camada de saida (tamanho passado na inicializacao)
        return YS
    
    def aplicacao(self, x): #arredondadmento para resultado
        round(self.feedfoard(x))

    def treinamento(self, X, y, n_epocas, taxa_aprendizado):
        for epoca in range(n_epocas): #passar n_epocas vezes pelo conjunto de dados de teste
            for i in range(X.shape[0]): #para cada vetor de nn_entrada passado
                saida = self.feedfoard(X[i])
                # Backpropagation
                for j in range(self.nn_saida):
                    #para cada neuronio da camada de saida
                    erro_saida = np.abs(saida[j] - y[j])
                    delta_saida = erro_saida + self.derivada_sigmoide(saida)

                    #ajuste de pesos
                    self.pesos2[][] += taxa_aprendizado * 

