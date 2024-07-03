import numpy as np

entradas = np.array([[0,0] , [0,1] , [1,0] , [1,1]])
saidas = np.array([0 , 0 , 0 , 1])
pesos = np.array([0.0 , 0.0])
taxaAprendizagem = 0.1

# funcao de ativacao
def stepFunction(soma): 
    if (soma >= 1):
        return 1
    return 0

# multiplica entradas pelos pesos
def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepFunction(s)

def treinar():
    erroTotal = 1
    while (erroTotal != 0):
        # comment: enquanto tivermos erros
        erroTotal = 0
        # para cada registro
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada) #abs remove o sinal
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print('peso atualizado: ' + str(pesos[j]))
        print('Total de erros ' + str(erroTotal))
    # end while

treinar()
