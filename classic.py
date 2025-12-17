import random
import time
import math

def classical_search_simulated(n_qubits, marked_state_binary):
    """
    Simula uma busca clássica linear em um espaço de 2^n_qubits estados.
    A função retorna o número de "consultas" (tentativas) para encontrar o estado marcado.
    Simula a ordem de busca aleatória para representar um caso médio.

    Args:
        n_qubits (int): O número de qubits, que define o tamanho do espaço de busca (2^n_qubits).
        marked_state_binary (str): O estado binário a ser encontrado (e.g., '101' para 3 qubits).

    Returns:
        int: O número de consultas/tentativas até encontrar o estado marcado.
    """
    N = 2**n_qubits
    
    # Converter o estado marcado para um inteiro para facilitar a comparação
    marked_state_int = int(marked_state_binary, 2)

    num_queries = 0
    # Criar uma lista de possíveis posições que a busca poderia seguir.
    # Embaralhar esta lista simula uma ordem aleatória de busca,
    # e, em média, representa a complexidade O(N/2).
    search_order = list(range(N))
    random.shuffle(search_order) 

    for queried_index in search_order:
        num_queries += 1
        if queried_index == marked_state_int:
            break
            
    return num_queries

def calculate_grover_optimal_iterations(N, M=1):
    """
    Calcula o número ótimo de iterações para o algoritmo de Grover.
    Cada iteração envolve uma "consulta" ao oráculo.

    Args:
        N (int): Tamanho do espaço de busca (2^n_qubits).
        M (int): Número de estados marcados. Assume M=1 para a maioria dos cenários.

    Returns:
        int: Número ótimo de iterações (consultas ao oráculo).
    """
    if M == 0 or N == 0:
        return 0 # Sem estados marcados ou espaço de busca vazio, 0 consultas.
    
    if M == N: # Todos os estados são marcados, 0 consultas (já sabemos a resposta).
        return 0
        
    theta = math.asin(math.sqrt(M / N))
    iterations = int(round((math.pi/2 - theta) / (2 * theta)))
    
    # Caso de borda: se o cálculo arredonda para 0, mas ainda há algo a ser buscado (M < N),
    # então pelo menos 1 iteração (consulta) é necessária.
    if iterations == 0 and M < N: 
        iterations = 1
        
    return iterations


if __name__ == "__main__":
    print("--- Simulação de Busca Clássica vs. Grover ---")

    # Lista de números de qubits para testar
    n_qubits_list = [2, 3, 5, 10, 15, 20, 25] 

    print("\nComparando o número de 'consultas' (operações essenciais):")
    print(f"{'N de Qubits':<15} {'Espaço de Busca (N)':<20} {'Clássico (Médio)':<20} {'Grover (Ótimo)':<20} {'Speedup (Clássico/Grover)':<28}")
    print("-" * 105)

    for n_qubits in n_qubits_list:
        N = 2**n_qubits
        
        # Gerar um estado marcado binário aleatório para cada N
        marked_state_binary = format(random.randint(0, N-1), '0' + str(n_qubits) + 'b')

        # --- Busca Clássica (Simulada para o caso médio) ---
        start_time_classical = time.perf_counter()
        num_queries_classical = classical_search_simulated(n_qubits, marked_state_binary)
        end_time_classical = time.perf_counter()
        
        # --- Grover (Número de iterações ótimas) ---
        num_queries_grover = calculate_grover_optimal_iterations(N)
        
        # Cálculo do "Speedup" (ganho de velocidade em termos de consultas)
        # É a razão entre as consultas clássicas e as consultas quânticas.
        actual_speedup_queries = num_queries_classical / num_queries_grover if num_queries_grover > 0 else float('inf')

        print(f"{n_qubits:<15} {N:<20} {num_queries_classical:<20} {num_queries_grover:<20} {actual_speedup_queries:<28.2f}")
        
        # Imprime o tempo real de simulação para o clássico
        print(f"  - Tempo de simulação Clássica para {N} itens: {end_time_classical - start_time_classical:.6f} segundos")

