import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Globais e Constantes ---
h = 0.005
tc = 0.5
t_inicial = 0.0
t_final = 2.0
w = 0.025
sigma1 = w
sigma2 = sigma1

N = int((t_final - t_inicial) / h)

# Valores REAIS (Verdadeiros) para gerar os Pseudo-Dados
P1_REAL = 1.417
P2_REAL = 0.25
P3_REAL = 15.0
P4_REAL = 0.3

# --- Funções Auxiliares (Parametrizadas) ---

def a_of_T2(T2, P2, P3, P4):
    return P2 + np.exp(-(P3*(T2-P4))**2)

def u_of_t(t):
    # P1 é mantido fixo/conhecido neste problema
    u = -(P1_REAL)/tc
    return u if t < tc else 0.0 

def f1(T1, T2, u_val):
    return u_val - (T1 - T2)

def f2(T1, T2, P2, P3, P4):
    a = a_of_T2(T2, P2, P3, P4)
    return (T1 - T2) / a

def rk4_step(T1n, T2n, u_val, P2, P3, P4):
    # k1
    k11 = f1(T1n, T2n, u_val)
    k12 = f2(T1n, T2n, P2, P3, P4)

    # k2
    T1_k2 = T1n + 0.5*h*k11
    T2_k2 = T2n + 0.5*h*k12
    k21 = f1(T1_k2, T2_k2, u_val)
    k22 = f2(T1_k2, T2_k2, P2, P3, P4)

    # k3
    T1_k3 = T1n + 0.5*h*k21
    T2_k3 = T2n + 0.5*h*k22
    k31 = f1(T1_k3, T2_k3, u_val)
    k32 = f2(T1_k3, T2_k3, P2, P3, P4)

    # k4
    T1_k4 = T1n + h*k31
    T2_k4 = T2n + h*k32
    k41 = f1(T1_k4, T2_k4, u_val)
    k42 = f2(T1_k4, T2_k4, P2, P3, P4)

    T1_np1 = T1n + (h/6.0)*(k11 + 2*k21 + 2*k31 + k41)
    T2_np1 = T2n + (h/6.0)*(k12 + 2*k22 + 2*k32 + k42)
    return T1_np1, T2_np1

# --- Solvers ---

def problema_direto_solve(u, P2, P3, P4, T1_init=1.0, T2_init=1.0):
    """
    Resolve o problema direto para um dado vetor u e parâmetros P2, P3, P4.
    """
    t = t_inicial
    times = [t]
    T1hist = [T1_init]
    T2hist = [T2_init]
    
    curr_T1 = T1_init
    curr_T2 = T2_init
    
    # O vetor u tem tamanho N. O loop deve rodar N vezes ou N-1 dependendo da malha.
    # Ajustando para iterar sobre o tamanho de u disponível.
    limit = min(len(u), N)
    
    for i in range(limit):
        curr_T1, curr_T2 = rk4_step(curr_T1, curr_T2, u[i], P2, P3, P4)
        t += h
        times.append(t)
        T1hist.append(curr_T1)
        T2hist.append(curr_T2)
        
    return np.array(T1hist), np.array(T2hist)

def pseudo_dados():
    """
    Gera dados sintéticos usando os parâmetros REAIS globais.
    """
    rng = np.random.default_rng(123)
    times = np.linspace(t_inicial, t_final, N+1) # +1 para incluir ponto final na malha visual
    
    # Gera u_real analítico
    u_vec = np.array([u_of_t(t) for t in times[:-1]]) # u é entrada nos intervalos
    
    # Resolve usando os parâmetros REAIS
    T1, T2 = problema_direto_solve(u_vec, P2_REAL, P3_REAL, P4_REAL, T1_init=1.0, T2_init=1.0)
    
    # Adiciona Ruído
    T1_noisy = T1 + rng.normal(0.0, w, size=len(T1))
    T2_noisy = T2 + rng.normal(0.0, w, size=len(T2))
    
    # Retorna times ajustado para o tamanho dos vetores T
    return times, T1_noisy, T2_noisy