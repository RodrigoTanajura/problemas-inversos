import numpy as np
import matplotlib.pyplot as plt

# Globais
h = 0.005
tc = 0.5

t_inicial = 0.0
t_final = 2.5
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

