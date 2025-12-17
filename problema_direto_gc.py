import numpy as np
import matplotlib.pyplot as plt

# Globais
h = 0.005
tc = 0.5

t_inicial = 0.0
t_final = 2.0
w = 0.025
sigma1 = w
sigma2 = sigma1

N = int(t_final / h)

P1 = 1.417
P2 = 0.25
P3 = 15
P4 = 0.3

def problema_direto(u, T1=1.0, T2=1.0):
    t = t_inicial
    times = [t]
    T1hist = [T1]
    T2hist = [T2]
    for i in range(N-1):
        T1, T2 = rk4_step(T1, T2, u[i])
        t += h
        times.append(t)
        T1hist.append(T1)
        T2hist.append(T2)
    return np.array(T1hist), np.array(T2hist)

def problema_adjunto(L1, L2, T1, T2, Y1, Y2):
    t = t_inicial
    times = [t]
    L1hist = [L1]
    L2hist = [L2]
    T1 = np.flip(T1)
    T2 = np.flip(T2)
    Y1 = np.flip(Y1)
    Y2 = np.flip(Y2)
    for i in range(N-1):
        L1, L2 = rk4_step_adjunto(L1, L2, T1[i], T2[i], Y1[i], Y2[i])
        t += h
        times.append(t)
        L1hist.append(L1)
        L2hist.append(L2)
    return np.flip(np.array(L1hist)), np.flip(np.array(L2hist))

def pseudo_dados():
    rng = np.random.default_rng(123)
    times = np.linspace(t_inicial, t_final, N, endpoint=False)
    u = np.array([u_of_t(t) for t in times])
    T1, T2 = problema_direto(u, T1=1.0, T2=1.0)
    T1_real = T1 + rng.normal(0.0, w, size=len(T1))
    T2_real = T2 + rng.normal(0.0, w, size=len(T2))
    return times, T1_real, T2_real

# -- funções helper problema direto -- #

def a_of_T2(T2):
    return P2 + np.exp(-(P3*(T2-P4))**2)

def u_of_t(t):
    u = -(P1)/tc
    return u if t < tc else 0.0 

def f1(T1, T2, u):
    return u - (T1 - T2)

def f2(T1, T2):
    a = a_of_T2(T2)
    return (T1 - T2) / a

def rk4_step(T1n, T2n, u):
    # k1
    k11 = f1(T1n, T2n, u)
    k12 = f2(T1n, T2n)

    # k2 (t + h/2)
    T1_k2 = T1n + 0.5*h*k11
    T2_k2 = T2n + 0.5*h*k12
    k21 = f1(T1_k2, T2_k2, u)
    k22 = f2(T1_k2, T2_k2)

    # k3 (t + h/2)
    T1_k3 = T1n + 0.5*h*k21
    T2_k3 = T2n + 0.5*h*k22
    k31 = f1(T1_k3, T2_k3, u)
    k32 = f2(T1_k3, T2_k3)

    # k4 (t + h)
    T1_k4 = T1n + h*k31
    T2_k4 = T2n + h*k32
    k41 = f1(T1_k4, T2_k4, u)
    k42 = f2(T1_k4, T2_k4)

    T1_np1 = T1n + (h/6.0)*(k11 + 2*k21 + 2*k31 + k41)
    T2_np1 = T2n + (h/6.0)*(k12 + 2*k22 + 2*k32 + k42)
    return T1_np1, T2_np1

# -- funções helper problema adjunto -- #

def C_de_T2(T2):
    return P2 + np.exp(-(P3*(T2-P4))**2)

def dL1(L1, L2, Y1, T1):
    L = L1 - L2 - 2 * ((Y1 -T1)/sigma1**2)
    return L

def dL2(L1, L2, Y2, T2):
    L = L2 - L1 - 2 * ((Y2 -T2)/sigma2**2)
    L = L/C_de_T2(T2)
    return L

def rk4_step_adjunto(L1, L2, T1, T2, Y1, Y2):
    # k1
    k11 = dL1(L1, L2, Y1, T1)
    k12 = dL2(L1, L2, Y2, T2)
    
    # k2 (t + h/2)
    L1_k2 = L1 + 0.5*h*k11
    L2_k2 = L2 + 0.5*h*k12
    k21 = dL1(L1_k2, L2_k2, Y1, T1)
    k22 = dL2(L1_k2, L2_k2, Y2, T2)

    # k3 (t + h/2)
    L1_k3 = L1 + 0.5*h*k21
    L2_k3 = L2 + 0.5*h*k22
    k31 = dL1(L1_k3, L2_k3, Y1, T1)
    k32 = dL2(L1_k3, L2_k3, Y2, T2)

    # k4 (t + h)
    L1_k4 = L1 + h*k31
    L2_k4 = L2 + h*k32
    k41 = dL1(L1_k4, L2_k4, Y1, T1)
    k42 = dL2(L1_k4, L2_k4, Y2, T2)

    L1_np1 = L1 - (h/6.0)*(k11 + 2*k21 + 2*k31 + k41)
    L2_np1 = L2 - (h/6.0)*(k12 + 2*k22 + 2*k32 + k42)

    return L1_np1, L2_np1

# ----------
def problema_sensibilidade(Du, T1, T2, DT1=0.0, DT2=0.0):
    t = t_inicial
    times = [t]
    DT1hist = [DT1]
    DT2hist = [DT2]
    for i in range(N-1):
        DT1, DT2 = rk4_step_sensibilidade(DT1, DT2, Du[i], T1[i], T2[i])
        t += h
        times.append(t)
        DT1hist.append(DT1)
        DT2hist.append(DT2)
    return np.array(DT1hist), np.array(DT2hist)

# -- funções helper problema de sensibilidade -- #

def rk4_step_sensibilidade(DT1, DT2, Du, T1, T2):
    # k1
    k11 = dDT1(DT1, DT2, Du)
    k12 = dDT2(DT1, DT2, T1, T2)
    
    # k2 (t + h/2)
    DT1_k2 = DT1 + 0.5*h*k11
    DT2_k2 = DT2 + 0.5*h*k12
    k21 = dDT1(DT1_k2, DT2_k2, Du)
    k22 = dDT2(DT1_k2, DT2_k2, T1, T2)

    # k3 (t + h/2)
    DT1_k3 = DT1 + 0.5*h*k21
    DT2_k3 = DT2 + 0.5*h*k22
    k31 = dDT1(DT1_k3, DT2_k3, Du)
    k32 = dDT2(DT1_k3, DT2_k3, T1, T2)

    # k4 (t + h)
    DT1_k4 = DT1 + h*k31
    DT2_k4 = DT2 + h*k32
    k41 = dDT1(DT1_k4, DT2_k4, Du)
    k42 = dDT2(DT1_k4, DT2_k4, T1, T2)

    DT1_np1 = DT1 + (h/6.0)*(k11 + 2*k21 + 2*k31 + k41)
    DT2_np1 = DT2 + (h/6.0)*(k12 + 2*k22 + 2*k32 + k42)

    return DT1_np1, DT2_np1

def dDT1(DT1, DT2, Du):
    return DT2 - DT1 + Du

def dDT2(DT1, DT2, T1, T2):
    return (DT1 - (1 + d_C_de_T2(T2) * dT2_dt(T1, T2)) * DT2)/C_de_T2(T2)

def dT2_dt(T1, T2):
    return (T1 - T2)/C_de_T2(T2)

# --- Função Auxiliar: Derivada de a(T2) em relação a T2 ---
def d_C_de_T2(T2):
    term = - (P3 * (T2 - P4))**2
    result = -2 * (((P3**2) * T2) - (P4 * (P3**2)) ) * np.exp(term)
    return result 

def beta_k(T, Y, DT, sigma):
    numerador = np.sum(((Y - T)/sigma**2)*DT)
    denominador = np.sum((DT**2)/(sigma**2))
    return float(numerador/denominador)

def passo(T1, T2, Y1, Y2, DT1, DT2):

    # Numerador Global
    res1 = (Y1 - T1) / (sigma1**2)
    res2 = (Y2 - T2) / (sigma2**2)
    numerador = np.sum(res1 * DT1) + np.sum(res2 * DT2)
    
    # Denominador Global
    sens1 = (DT1**2) / (sigma1**2)
    sens2 = (DT2**2) / (sigma2**2)
    denominador = np.sum(sens1) + np.sum(sens2)
    
    if denominador == 0: return 0.0
    return float(numerador / denominador)
    