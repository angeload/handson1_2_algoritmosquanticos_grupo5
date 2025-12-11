# Algoritmo de Grove para 2-bit

# Importando as bibliotecas necessárias
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import time

# Definição de funções
def grover_oracle(circuit: QuantumCircuit, marked_state_str: str):
    """
    Implementa o oráculo de Grover para marcar um estado específico.
    O oráculo inverte a fase do estado marcado |marked_state_str⟩.

    Args:
        circuit (QuantumCircuit): O circuito quântico onde o oráculo será aplicado.
        marked_state_str (str): O estado binário a ser marcado (ex: '110' para 3 qubits). A string é lida como Big Endian (q_n-1 ... q_0).
    """
    n = circuit.num_qubits

    # 1. Aplicar portas X para converter '0's em '1's para o estado marcado
    #    Isso transforma o estado marcado em |11...1⟩ para o MCZ gate.
    #    A indexação da string é MSB-first (q_n-1 ... q_0), enquanto o Qiskit é LSB-first (q0 ... q_n-1). 
    # marked_state_str[n - 1 - i] acessa o bit de q_i.
    for i in range(n):
        if marked_state_str[n - 1 - i] == '0':
            circuit.x(i)

    # 2. Aplicar uma porta MCZ (Multi-Controlled Z).
    #    Para 3 qubits, é uma CCZ implementada como H + MCX + H.
    #    A MCX atua como uma Toffoli de 2 controles.
    #    q(0) e q(1) atuam sobre q(2)
    circuit.h(n - 1) # H no qubit mais significativo (target da MCX)
    circuit.mcx(list(range(n - 1)), n - 1) # MCX com q0, q1 como controles e q2 como target
    circuit.h(n - 1) # H no qubit mais significativo

    # 3. Reverter as portas X aplicadas no passo 1
    for i in range(n):
        if marked_state_str[n - 1 - i] == '0':
            circuit.x(i)
    circuit.barrier() # Barreira para visualização

def grover_diffuser(circuit: QuantumCircuit):
    """
    Implementa o operador difusor (inversão em torno da média).
    Amplifica a amplitude do estado marcado e diminui as amplitudes dos outros estados.

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
    circuit.mcx(list(range(n - 1)), n - 1) # MCX com q_0, q_n-2 como controles e q_n-1 como target
    circuit.h(n - 1) # H no qubit mais significativo

    # 4. Reverter as X
    circuit.x(range(n))
    # 5. Reverter as H
    circuit.h(range(n))
    circuit.barrier()

# -----------------
# PROGRAMA PRINICIPAL
# -----------------
# Definição dos Parâmetros do Algoritmo
# n_qubits = 2  # Número de qubits
# marked_state = '11' # O estado que queremos encontrar
# n_qubits = 3  # Número de qubits
# marked_state = '110' # O estado que queremos encontrar
# n_qubits = 16  # Número de qubits
# marked_state = '1100110011001100' # O estado que queremos encontrar
# n_qubits = 8  # Número de qubits
# marked_state = '11001100' # O estado que queremos encontrar
n_qubits = 10  # Número de qubits
marked_state = '1100110011' # O estado que queremos encontrar

# Numero de estados na base de busca
N = 2**n_qubits

# Número de estados marcados
M = 1

# Calculando o número ótimo de iterações (arredondado para o inteiro mais próximo)
# aproximadamente (pi/4) * sqrt(N/M)
num_iterations = int(np.round(np.pi / 4 * np.sqrt(N / M)))
print(f"Número de qubits: {n_qubits}")
print(f"Número total de estados (N): {N}")
print(f"Estado marcado: |{marked_state}⟩")
print(f"Número ótimo de iterações do Grover: {num_iterations}\n")

# Criação do Circuito Quântico
# O circuito terá n_qubits qubits quânticos e n_qubits bits clássicos para medição
qc = QuantumCircuit(n_qubits, n_qubits)


# Inicialização para Superposição Uniforme
# Aplica uma porta Hadamard (H) a cada qubit.
# Isso coloca todos os qubits em uma superposição uniforme de todos os 2^n estados possíveis.
# Cada estado tem uma amplitude de 1/sqrt(N).
qc.h(range(n_qubits))
qc.barrier() # Barreira para visualização clara no circuito


# Loop das Iterações do Grover
for i in range(num_iterations): 
    print(f"Aplicando iteração Grover...{i+1}/{num_iterations}", end='\r')
    grover_oracle(qc, marked_state)
    grover_diffuser(qc)
print("\n")

# Medição
# Mede todos os qubits quânticos e armazena os resultados nos bits clássicos.
qc.measure(range(n_qubits), range(n_qubits))


# Visualização do Circuito Final
print("Salvando circuito final do Algoritmo de Grover...")
# Desenha o circuito. 'mpl' para visualização em matplotlib.
# Se usar 'fold=-1' evita que o circuito seja dobrado em múltiplas linhas
qc.draw('mpl', fold=100, scale=0.5, plot_barriers=False, filename='./figures/grover_circuit_'+str(n_qubits)+'-bit.png')
# plt.show()
print("Circuito salvo em './figures/grover_circuit_"+str(n_qubits)+"-bit.png'\n")

# ----------
# Simulação e Resultados
# ----------
print("Iniciando simulação do circuito...")
start_time = time.perf_counter() # inicio da simulação

simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)

# Executa o circuito no simulador
shots = 10
job = simulator.run(compiled_circuit, shots=shots)
result = job.result()
counts = result.get_counts(qc)

end_time = time.perf_counter() # fim da simulação
print("Simulação concluída.")

elapsed_time = end_time - start_time # Tempo decorrido
print(f"Tempo de simulação: {elapsed_time:.4f} segundos")

print("\nResultados das Medições:")
print(counts)

# Salvando o histograma em arquivo
print("Salvando histograma dos resultados...")
plot_histogram(counts, title=f"Probabilidades após {num_iterations} iterações do Grover").savefig('./figures/grover_histogram_'+str(n_qubits)+'-bit.png')
print("Histograma salvo em './figures/grover_histogram_"+str(n_qubits)+"-bit.png'\n"
      )
