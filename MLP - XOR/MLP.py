import numpy as np

class MultilayerPerceptron:
    def __init__(self, nn_entrada, nn_saida,nn_escondida):
        self.pesos1 = np.random.randn(nn_entrada, nn_escondida) # matris de pesos - cada neuronio de entrada vai para cada um na camada escondida
        self.bias1 = np.zeros(nn_escondida) #vetor de pesos bias pra cada neuronio da camada escondida
        self.pesos2 = np.random.randn(nn_escondida, nn_saida) # matris de pesos - cada neuronio da camada escondida vai para cada um na camada de saida
        self.bias2 = np.zeros(nn_saida)
        self.nc_escondidas = nn_escondida
    
    def sigmoide(self, x): # funcao de ativacao - retorna valor de (0,1) e tem derivada em todos os pontos
        return 1/(1+np.exp(-x))
    
    def derivada_sigmoide(self, x): # precisa da derivada para saber a direcao do erro e fazer a correcao dos pesos no treinamento
        return x*(1-x)
    
    def rodar(self, x): #recebe um vetor de tamanho = nn_entrada 
        YinE = np.dot(x, self.pesos1) + self.bias1 # soma ponderada primeira camanda
        saida_intermediaria = self.sigmoide(YinE) # saida camada escondida
        YinS = np.dot(saida_intermediaria, self.pesos2) + self.bias2 # soma ponderada primeira camanda
        saida = self.sigmoide(YinS) #saida camada de saida (tamanho passado na inicializacao)
        return saida

    def treinamento(self, X, y, n_epocas, taxa_aprendizado):
        for epoca in range(n_epocas): #passar n_epocas vezes pelo conjunto de dados de teste
            for i in range(X.shape[0]): #para cada vetor de nn_entrada passado
                # Feedfoward
                saida_intermediaria = self.sigmoide(np.dot(X[i], self.pesos1) + self.bias1)
                saida = self.sigmoide(np.dot(saida_intermediaria, self.pesos2) + self.bias2)

                # Backpropagation
                erro_saida = y - saida
                erro_saida = erro_saida[:2]
                delta_saida = erro_saida*self.derivada_sigmoide(saida)

                erro_intermediaria = np.zeros(self.nc_escondidas)
                for i in range(len(saida)):
                    for j in range(self.nc_escondidas):
                        erro_intermediaria[j] += erro_saida[i]*self.pesos2.T[i][j]
                delta_intermediaria = erro_intermediaria*self.derivada_sigmoide(saida_intermediaria)


                #ajuste de pesos
                self.pesos2 += taxa_aprendizado*saida_intermediaria.T.dot(delta_saida)
                self.bias2 += taxa_aprendizado*np.sum(delta_saida, axis=0)
                self.pesos1 += taxa_aprendizado*X[i].T.dot(delta_intermediaria)
                self.bias1 += taxa_aprendizado*np.sum(delta_intermediaria, axis=0)
                        