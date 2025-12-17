# Algoritmo de Grove para n-qubit para múltiplos alvos com ruído

# Importando as bibliotecas necessárias
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
# Importações para ruído
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import time

# Definição de funções 
def grover_oracle(circuit: QuantumCircuit, marked_states_list: list[str]):
    n = circuit.num_qubits
    for marked_state_str in marked_states_list:
        if len(marked_state_str) != n:
            raise ValueError(f"O estado marcado '{marked_state_str}' tem {len(marked_state_str)} qubits, mas o circuito tem {n} qubits. Todos os estados marcados devem ter o mesmo número de qubits que o circuito.")
        for i in range(n):
            if marked_state_str[n - 1 - i] == '0':
                circuit.x(i)
        circuit.h(n - 1)
        circuit.mcx(list(range(n - 1)), n - 1)
        circuit.h(n - 1)
        for i in range(n):
            if marked_state_str[n - 1 - i] == '0':
                circuit.x(i)
    circuit.barrier()

def grover_diffuser(circuit: QuantumCircuit):
    n = circuit.num_qubits
    circuit.h(range(n))
    circuit.x(range(n))
    circuit.h(n - 1)
    circuit.mcx(list(range(n - 1)), n - 1)
    circuit.h(n - 1)
    circuit.x(range(n))
    circuit.h(range(n))
    circuit.barrier()

# -----------------
# PROGRAMA PRINCIPAL
# -----------------
# Definição dos Parâmetros do Algoritmo
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

# Número de estados marcados
M = len(marked_states_list)

# Calculando o número ótimo de iterações
theta = np.arcsin(np.sqrt(M/N))
num_iterations = int(np.round((np.pi/2 - theta) / (2 * theta)))
if num_iterations == 0 and theta != 0:
    num_iterations = 1
elif num_iterations == 0 and theta == 0 and M > 0:
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
qc.barrier()

# Loop das Iterações do Grover
for i in range(num_iterations): 
    print(f"Aplicando iteração Grover...{i+1}/{num_iterations}", end='\r')
    grover_oracle(qc, marked_states_list)
    grover_diffuser(qc)
print("\n")

# Medição
qc.measure(range(n_qubits), range(n_qubits))


# ----------
# Configuração do Modelo de Ruído
# ----------
print("Configurando o modelo de ruído...")

# 1. Cria um modelo de ruído vazio
noise_model = NoiseModel()

# 2. Define as probabilidades de erro
# Para portas de 1 qubit (H, X, etc.)
p1q = 0.001  # Probabilidade de erro de depolarização em portas de 1 qubit (0.1%)
# Para portas de 2 qubits (CX, etc.)
p2q = 0.01   # Probabilidade de erro de depolarização em portas de 2 qubits (1%)
# Para erros de leitura (probabilidade de ler 0 como 1, ou 1 como 0)
p_readout = 0.01 # 1% de erro de leitura

# 3. Cria os objetos de erro
# Erro depolarizador para portas de 1 qubit
error_1q = depolarizing_error(p1q, 1)
# Erro depolarizador para portas de 2 qubits
error_2q = depolarizing_error(p2q, 2)
# Erro de leitura (matriz de probabilidades de [p(0|0) p(1|0)], [p(0|1) p(1|1)])
readout_matrix = [[1 - p_readout, p_readout], [p_readout, 1 - p_readout]]
# Crie um objeto ReadoutError a partir da matriz
readout_error_object = ReadoutError(readout_matrix)


# 4. Adiciona os erros ao modelo de ruído
# Adiciona erro depolarizador para todas as portas de 1 qubit padrão do Qiskit (incluindo H, X, etc.)
noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'sx', 'rz', 'id']) # 'id' é a porta de identidade (qubit ocioso)
# Adicione erro depolarizador para todas as portas de 2 qubits padrão (principalmente 'cx')
noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'swap'])
# Adiciona erro de leitura para todas as medições
noise_model.add_all_qubit_readout_error(readout_error_object)

print(f"Modelo de ruído configurado com p1q={p1q}, p2q={p2q}, p_readout={p_readout}.\n")


# ----------
# Simulação e Resultados com Ruído
# ----------
print("Iniciando simulação do circuito com ruído...")
start_time = time.perf_counter() # inicio da simulação

simulator = AerSimulator()
# O transpiler tenta otimizar o circuito para o backend (simulador),
# o que inclui decompor portas complexas como MCX em portas nativas 
# do backend onde aplicamos o ruído.
compiled_circuit = transpile(qc, simulator)

# Executa o circuito no simulador, passando o noise_model
shots = 1024
job = simulator.run(compiled_circuit, shots=shots, noise_model=noise_model) 
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
    print(f"  Estado |{state}⟩: {probabilities[state]*100:.2f}%")


print("\nSalvando histograma dos resultados com ruído...")
plot_histogram(probabilities, title=f"Probabilidades após {num_iterations} iterações do Grover (com ruído) e {shots} shots").savefig('./figures/grover_noisy_multi_target_histogram_'+str(n_qubits)+'-bit.png')
print("Histograma salvo em './figures/grover_noisy_multi_target_histogram_"+str(n_qubits)+"-bit.png'\n")
