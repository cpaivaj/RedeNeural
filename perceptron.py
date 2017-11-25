# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:26:59 2017

@author: pc1
"""
#
#
#       O OBJETICO DESSE ALGORITMO É ENCONTRAR OS PESOS
#
#


import numpy as np

# Clasificaçao do operador AND
#entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # vertices
#saidas = np.array([0, 0, 0, 1]) # saidas, quando entrada for 00, saida eh 0 e por ai vai

# Classificação do operador OR
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # vertices
saidas = np.array([0, 1, 1, 1]) # saidas, quando entrada for 00, saida eh 0 e por ai vai

# Classificação do operador XOR (so eh 1 se os elementos forem diferentes)
# So funciona se usar multicamadas, uma camada nao consegue resolver
#entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # vertices
#saidas = np.array([0, 1, 1, 0]) # saidas, quando entrada for 00, saida eh 0 e por ai vai


pesos = np.array([0.0, 0.0]) # sao apenas 2 registros, por isso 2 pesos
taxaAprendizagem = 0.1

# verifica se eh valido ou nao
def stepFunction(soma):
    if soma >= 1:
        return 1
    return 0

# calculos necessarios
def calculaSaida(registro):
    s = registro.dot(pesos) # somatorio e multipicacao do registro com os pesos
    return stepFunction(s)

# treinamento para ajuste de peso
def treinar():
    erroTotal = 1
    while (erroTotal != 0): # na maioria dos casos nao vai funcionar, pq nem sempre tem 100% de acerto
        erroTotal = 0
        for i in range(len(saidas)): # len eh o tamanho de saidas
            saidaCalculada = calculaSaida(np.asarray(entradas[i])) # entradas eh uma matriz, converte pra vetor
            erro = abs(saidas[i] - saidaCalculada) # abs para nao ter valor negativo
            erroTotal += erro
            for j in range(len(pesos)):
                # formula para calculo dos pesos
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print('Peso atualizado: ' + str(pesos[j]))
        print('Total de erros: ' + str(erroTotal))
            
treinar()
print('Rede Neural treinada')
print(calculaSaida(entradas[0]))
print(calculaSaida(entradas[1]))
print(calculaSaida(entradas[2]))
print(calculaSaida(entradas[3]))







