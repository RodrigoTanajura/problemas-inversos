import numpy as np
import matplotlib.pyplot as plt

# Globais
h = 0.01
tc = 0.5
resolution = 100

t = 0.0
t_final = 2.0
w = 0.005
sigma1 = w
sigma2 = sigma1

N = int(t_final / h)
T1 = 1.0
T2 = 1.0
W = np.eye(2*(N+1))*w

P1 = 1.417
P2 = 0.25
P3 = 15
P4 = 0.3

def problema_direto(T1=T1, T2=T2, u=0.0):
    times = [0.0]
    T1hist = [0]
    T2hist = [0]
    for i in range(N):
        T1, T2 = rk4_step(t, T1, T2, u)
        t += h
        times.append(t)
        T1hist.append(T1)
        T2hist.append(T2)
    return times, T1hist, T2hist

def problema_adjunto(T, T_pseudo, u):
    times = [0.0]
    L1hist = [0]
    L2hist = [0]
    T = np.flip(T)
    T_pseudo = np.flip(T_pseudo)
    for i in range(N):
        L1, L2 = rk4_step_adjunto(t, T,  T_pseudo, u)
        t += h
        times.append(t)
        L1hist.append(L1)
        L2hist.append(L2)
    return np.array(times), np.array(L1hist), np.array(L2hist)

# -- funções helper -- #

def a_of_T2(T2, P2, P3, P4):
    return P2 + np.exp(-(P3*(T2-P4))**2)

def u_of_t(t, P1):
    u = -(P1)/tc
    return u if t < tc else 0.0 

def f1(t, T1, T2, P1):
    return u_of_t(t, P1) - (T1 - T2)

def f2(t, T1, T2, P2, P3, P4):
    a = a_of_T2(T2, P2, P3, P4)
    return (T1 - T2) / a

def rk4_step(tn, T1n, T2n, P1, P2, P3, P4):
    # k1
    k11 = f1(tn, T1n, T2n, P1)
    k12 = f2(tn, T1n, T2n, P2, P3, P4)

    # k2 (t + h/2)
    T1_k2 = T1n + 0.5*h*k11
    T2_k2 = T2n + 0.5*h*k12
    k21 = f1(tn + 0.5*h, T1_k2, T2_k2, P1)
    k22 = f2(tn + 0.5*h, T1_k2, T2_k2, P2, P3, P4)

    # k3 (t + h/2)
    T1_k3 = T1n + 0.5*h*k21
    T2_k3 = T2n + 0.5*h*k22
    k31 = f1(tn + 0.5*h, T1_k3, T2_k3, P1)
    k32 = f2(tn + 0.5*h, T1_k3, T2_k3, P2, P3, P4)

    # k4 (t + h)
    T1_k4 = T1n + h*k31
    T2_k4 = T2n + h*k32
    k41 = f1(tn + h, T1_k4, T2_k4, P1)
    k42 = f2(tn + h, T1_k4, T2_k4, P2, P3, P4)

    T1_np1 = T1n + (h/6.0)*(k11 + 2*k21 + 2*k31 + k41)
    T2_np1 = T2n + (h/6.0)*(k12 + 2*k22 + 2*k32 + k42)
    return T1_np1, T2_np1


def pseudo_dados():
    rng = np.random.default_rng(123)
    T1, T2 = problema_direto()
    T1_real = (np.array(T1) + rng.normal(0.0, w, size=len(T1))).tolist()
    T2_real = (np.array(T2) + rng.normal(0.0, w, size=len(T2))).tolist()
    return T1_real, T2_real

# -- funções helper -- #

def C_de_T2(T2):
    return P2 + np.exp(-(P3*(T2-P4))**2)

def dL1(L1, L2, Y1, T1):
    L = L1 - L2 - 2 * ((Y1 -T1)/sigma1)
    return L

def dL2(L1, L2, Y2, T2):
    L = L2 - L1 - 2 * ((Y2 -T2)/sigma2)
    L = L/C_de_T2(T2)
    return L

def rk4_step_adjunto(tn, L1, L2, Y1, T1, Y2, T2):
    # k1
    k11 = dL1(L1, L2, Y1, T1)
    k12 = dL2(L1, L2, Y2, T2)
    
    # k2 (t + h/2)
    L1_k2 = L1 + 0.5*h*k11
    L2_k2 = L2 + 0.5*h*k12
    k21 = dL1(tn + 0.5*h, L1_k2, L2_k2)
    k22 = dL2(tn + 0.5*h, L1_k2, L2_k2)

    # k3 (t + h/2)
    L1_k3 = L1 + 0.5*h*k21
    L2_k3 = L2 + 0.5*h*k22
    k31 = dL1(tn + 0.5*h, L1_k3, L2_k3)
    k32 = dL2(tn + 0.5*h, L1_k3, L2_k3)

    # k4 (t + h)
    L1_k4 = L1 + h*k31
    L2_k4 = L2 + h*k32
    k41 = dL1(tn + h, L1_k4, L2_k4)
    k42 = dL2(tn + h, L1_k4, L2_k4)

    L1_np1 = L1 + (h/6.0)*(k11 + 2*k21 + 2*k31 + k41)
    L2_np1 = L2 + (h/6.0)*(k12 + 2*k22 + 2*k32 + k42)

    return L1_np1, L2_np1
