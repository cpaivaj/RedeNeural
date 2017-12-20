# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:33:34 2017

@author: Carlos
"""

# Utiliza o metodo feed forward, que parte da camada principal para a camada oculta

import numpy as np

# Para descobrir o valor de ativacao, com base na soma feita apos todos os calculos de vertices e aarestas
def sigmoid(soma):
    # formula sigmoid -> y = 1 / 1 + e^-x
    return 1 / (1 + np.exp(-soma)) # nao retorna valores negativos

# Calculo de derivada com base no valor de ativacao (obtido pela sigmoid)
# para saber, no calculodo gradiente, qual direcao a escolha sera feita, para o minimo global (ajuste dos pesos)
def sigmoidDerivada(sig):
    # formula derivada -> d = y * (1 - y)
    return sig * (1 - sig)

# Calculo do delta de SAIDA com base no resultado da derivada
def deltaSaidaCalculo(erro, derivada):
    return erro * derivada


# resultCerto = 1
# soma = -0.381
# ativacao = sigmoid(soma)
# erro = resultCerto - ativacao
# derivada = sigmoidDerivada(ativacao)
# deltaSaida = deltaSaidaCalculo(erro, derivada)
# peso = -0.893
# derivada = 0.25
# deltaSaida = -0.098

#a = sigmoid(50)

# Ajuste de pesos usando multicamada (problemas nao linear)
# Usando multicamadas, sao definidos varios valores diferentes para a ativacao da camada oculta, usando a funcao sigmoid
# Diferente de uma camada so, que verifica a ativacao com 1 ou 0
entradas = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])
    
# respostas que ja conheco
saidas = np.array([[0], [1], [1], [0]])

# sinapse entre a camada de entrada e a camada oculta
# usando pesos conhecidos (EXECUCAO DE SIMULAÇÃO)
#pesosEntradaOculta = np.array([[-0.424, -0.740, -0.961],
#                   [0.358, -0.577, -0.469]])
    
# sinapse entre a camada oculta e a camada de saida
# usando pesos conhecidos
#pesosOcultaSaida = np.array([[-0.017], [-0.893], [0.148]])

# Usando pesos aleatórios (PROBLEMAS REAIS USAM ALEATORIEDADE)
# 2 neuronios de entrada, e 3 ocultos
pesosEntradaOculta = 2*np.random.random((2, 3)) - 1 # Multiplica por 2 e subtrai 1, gera alguns numeros positivos e outros negativos
# 3 neuronios na camada oculta e um de saida
pesosOcultaSaida   = 2*np.random.random((3, 1)) - 1

# quantidade de vezes que vai atualizar os pesos, controlador de execucoes desejadas (controla o for)
#epocas = 1000000 # tempo de treinamento
contEpocas = 0 # incrementado a cada giro do while
taxaAprendizagem = 0.3
limiteErro = 5.0
momento = 1
erroInicial = 0
erro = limiteErro + 1


#for j in range(epocas):
while(erro > limiteErro):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesosEntradaOculta) # calculo de entradas e sinapses da camada entrada ate a oculta
    camadaOculta = sigmoid(somaSinapse0) # valores finais para cada parte da camada oculta
    
    # apos obter os valores da camada oculta, o calculo é o mesmo que o usado no algoritmo de uma camada
    somaSinapse1 = np.dot(camadaOculta, pesosOcultaSaida) # sinapse da oculta para a saida
    
    # resultado para comparacao com o  saida
    camadaSaida = sigmoid(somaSinapse1) # valor de ativacao de cada registro
    
    # Calcuar erro
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida)) # media absoluta, usa todos os valores sem considerar o sinal
    erro = mediaAbsoluta*100
    print("Erro: " + str(erro) + "%")
    
    if(contEpocas == 0):
        erroInicial = mediaAbsoluta*100
    
    # Para melhorar melhorar essa rede, precisa DIMINUIR essa media
    # Para saber a porcentagem de acerto, divide 1 pela media
    
    # O calculo do gradiente serve para saber quanto ajustar nos pesos, é o exemplo da bola solta em uma tigela,
    # ela vai movendo ate conseguir parar no fundo da melhor forma possivel
    # Existe o minimo local, e global, sempre ensinar o algoritmo a seguir o global
    # Para ir na direcao certa (global) usa a derivada parcial, usando o resultado do sigmoid e aplica na formula da derivada (sigmoid * (1 - sigmoid))
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = deltaSaidaCalculo(erroCamadaSaida, derivadaSaida)
    
    # Calcula o delta da camada OCULTA com base nos calculos anteriores
    # peso que esta indo para a proxima camada
    # aqui esta a logica de BackPropagation, 1º calcula a derivada de saida, depois volta para a oculta pora calcular o delta da oculta com base no calculo feito com a saida (atualizacao dos pesos)
    # com base nos erros, o backPropagation faz a atualizacao dos pesos anteriores
    # Esse calculo é  feito para cada UM DOS NÓS da camada oculta
    
    # Matriz transposta para multiplicar duas matrizes com tamanhos diferentes
    pesosOcultaSaidaTransposta = pesosOcultaSaida.T
    deltaSaida_x_Peso = deltaSaida.dot(pesosOcultaSaidaTransposta)
    deltaCamadaOculta = deltaSaida_x_Peso * sigmoidDerivada(camadaOculta)
    # as linhas da matriz sao cada um dos registros
    # as colunas da matriz sao cada um dos vertices da camada oculta
    
    # Atualizacao dos pesos
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesosOcultaSaida = (pesosOcultaSaida * momento) + (pesosNovo1 * taxaAprendizagem)
    
    # Atualizacao dos pesos da camada oculta
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesosEntradaOculta = (pesosEntradaOculta * momento) + (pesosNovo0 * taxaAprendizagem)
    
    contEpocas = contEpocas + 1

print()
print()
print("Quantidade de epocas: " + str(contEpocas))
print("Porcentagem de acerto inicial: " + str(100 - erroInicial) + "%")
print("Porcentagem de acerto final: " + str(100 - erro) + "%")
    
    
    
    
    