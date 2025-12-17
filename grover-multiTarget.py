# Algoritmo de Grove para n-qubit para múltiplos alvos

# Importando as bibliotecas necessárias
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import time

# Definição de funções
def grover_oracle(circuit: QuantumCircuit, marked_states_list: list[str]):
    """
    Implementa o oráculo de Grover para marcar múltiplos estados específicos.
    O oráculo inverte a fase de cada estado marcado.

    Args:
        circuit (QuantumCircuit): O circuito quântico onde o oráculo será aplicado.
        marked_states_list (list[str]): Uma lista de estados binários a serem marcados (ex: ['110', '001'] para 3 qubits).
        Cada string é lida como Big Endian (q_n-1 ... q_0).
    """
    n = circuit.num_qubits

    for marked_state_str in marked_states_list:
        # Validação do tamanho da string do estado marcado
        if len(marked_state_str) != n:
            raise ValueError(f"O estado marcado '{marked_state_str}' tem {len(marked_state_str)} qubits, mas o circuito tem {n} qubits. Todos os estados marcados devem ter o mesmo número de qubits que o circuito.")

        # 1. Aplicar portas X para converter '0's em '1's para o estado marcado
        # Isso transforma o estado marcado em |11...1⟩ para o MCZ gate.
        # A indexação da string é MSB-first (q_n-1 ... q_0), enquanto o Qiskit é LSB-first (q0 ... q_n-1). 
        #    marked_state_str[n - 1 - i] acessa o bit de q_i.
        for i in range(n):
            if marked_state_str[n - 1 - i] == '0':
                circuit.x(i)

        # 2. Aplicar uma porta MCZ (Multi-Controlled Z).
        #    Para n qubits, é um (n-1)-CCZ implementada como H + MCX + H.
        #    q(0) a q(n-2) atuam sobre q(n-1)
        circuit.h(n - 1) # H no qubit mais significativo (target da MCX)
        circuit.mcx(list(range(n - 1)), n - 1) # MCX com q0, ..., q_n-2 como controles e q_n-1 como target
        circuit.h(n - 1) # H no qubit mais significativo

        # 3. Reverter as portas X aplicadas no passo 1
        for i in range(n):
            if marked_state_str[n - 1 - i] == '0':
                circuit.x(i)
    circuit.barrier() # Barreira para visualização

def grover_diffuser(circuit: QuantumCircuit):
    """
    Implementa o operador difusor (inversão em torno da média).
    Amplifica a amplitude dos estados marcados e diminui as amplitudes dos outros estados.

    Args:
        circuit (QuantumCircuit): O circuito quântico onde o difusor será aplicado.
    """
    n = circuit.num_qubits

    # 1. Aplicar Hadamard a todos os qubits
    circuit.h(range(n))
    # 2. Aplicar X a todos os qubits
    circuit.x(range(n))

    # 3. Aplicar um MCZ (Multi-Controlled Z) gate que inverte a fase do estado |11...1⟩
    # É o mesmo padrão H + MCX + H do oráculo.
    circuit.h(n - 1) # H no qubit mais significativo
    circuit.mcx(list(range(n - 1)), n - 1) # MCX com q_0, ..., q_n-2 como controles e q_n-1 como target
    circuit.h(n - 1) # H no qubit mais significativo

    # 4. Reverter as X
    circuit.x(range(n))
    # 5. Reverter as H
    circuit.h(range(n))
    circuit.barrier()

# -----------------
# PROGRAMA PRINCIPAL
# -----------------
# Definição dos Parâmetros do Algoritmo
# Define múltiplos estados na forma de uma lista
# n_qubits = 3  # Número de qubits
# marked_states_list = ['110', '001']
n_qubits = 5
marked_states_list = ['10100', '00101', '11111']
# n_qubits = 16
# marked_states_list = ['1100110011001100', '0011001100110011', '1010101010101010']


# Numero de estados na base de busca
N = 2**n_qubits

# Número de estados marcados (agora é o tamanho da lista)
M = len(marked_states_list)

# Calculando o número ótimo de iterações
theta = np.arcsin(np.sqrt(M/N))
num_iterations = int(np.round((np.pi/2 - theta) / (2 * theta)))
# Garante que sempre haja pelo menos 1 iteração se o estado inicial não for o marcado
if num_iterations == 0 and theta != 0:
    num_iterations = 1
elif num_iterations == 0 and theta == 0 and M > 0: # Caso M > 0 mas theta é 0
    num_iterations = 1


# Exibindo os Parâmetros do Algoritmo
print(f"Número de qubits: {n_qubits}")
print(f"Número total de estados (N): {N}")
print(f"Estados marcados: {marked_states_list}")
print(f"Número de estados marcados (M): {M}")
print(f"Número ótimo de iterações do Grover: {num_iterations}\n")

# Criação do Circuito Quântico
qc = QuantumCircuit(n_qubits, n_qubits)


# Inicialização para Superposição Uniforme
qc.h(range(n_qubits))
qc.barrier() # Barreira para visualização clara no circuito


# Loop das Iterações do Grover
for i in range(num_iterations): 
    print(f"Aplicando iteração Grover...{i+1}/{num_iterations}", end='\r')
    grover_oracle(qc, marked_states_list) # Agora passa a lista de estados
    grover_diffuser(qc)
print("\n")

# Medição
qc.measure(range(n_qubits), range(n_qubits))


# # Visualização do Circuito Final (descomente para salvar a imagem do circuito)
# print("Salvando circuito final do Algoritmo de Grover para múltiplos alvos...")
# qc.draw('mpl', fold=100, scale=0.5, plot_barriers=False, filename='./figures/grover_multi_target_circuit_'+str(n_qubits)+'-bit.png')
# print("Circuito salvo em './figures/grover_multi_target_circuit_"+str(n_qubits)+"-bit.png'\n")

# ----------
# Simulação e Resultados
# ----------
print("Iniciando simulação do circuito...")
start_time = time.perf_counter() # inicio da simulação

simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)

# Executa o circuito no simulador
shots = 1024
job = simulator.run(compiled_circuit, shots=shots)
result = job.result()
counts = result.get_counts(qc)

end_time = time.perf_counter() # fim da simulação
print("Simulação concluída.")

elapsed_time = end_time - start_time # Tempo decorrido
print(f"Tempo de simulação: {elapsed_time:.4f} segundos")


# Calcula o total de contagens
total_counts = sum(counts.values())

# Dicionário para armazenar as probabilidades percentuais
probabilities = {state: count / total_counts for state, count in counts.items()}

print("\nResultados das Medições (Percentuais):")
# Ordena para uma exibição mais limpa e formata como percentual
for state in sorted(probabilities, key=lambda s: probabilities[s], reverse=True):
    print(f"  Estado |{state}⟩: {probabilities[state]:.2f}%")


print("\nSalvando histograma dos resultados...")
# plot_histogram() com um dicionário de probabilidades (valores de 0 a 1)
plot_histogram(probabilities, title=f"Probabilidades após {num_iterations} iterações do Grover: {M} alvos e {shots} shots").savefig('./figures/grover_multi_target_histogram_'+str(n_qubits)+'-bit.png')
print("Histograma salvo em './figures/grover_multi_target_histogram_"+str(n_qubits)+"-bit.png'\n")