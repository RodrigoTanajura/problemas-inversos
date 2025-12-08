import numpy as np
from time import time
from matplotlib import pyplot as plt
from problema_direto import problema_direto, pseudo_dados, problema_adjunto, problema_sensibilidade, passo, u_of_t
from problema_direto import N

L1_0 = 0
L2_0 = 0

def objective_functional(T1, T2, Y1, Y2):
    return np.sum((T1 - Y1)**2) + np.sum((T2 - Y2)**2)

def direction_of_descent(i, grad, grad_old, dk_old, version='fs'):
    if version == 'fs':
        y_k  = (np.sum((grad)**2))/ np.sum(((grad_old)**2)) if i !=0 else 0
        psi_k = 0
        dk_q = 0

    elif version == 'pr':
        psi_k = 0
        dk_q = 0
        
    elif version == 'hs':
        psi_k = 0
        dk_q = 0

    elif version == 'pb':
        pass
    
    d_k = -grad + (y_k * dk_old) + psi_k * dk_q
    return d_k

def gradiente_conjugado(eps, u_0, version='fs'):
    i = 0
    u = u_0
    i_max = 1000
    grad_old = 1
    dk_old = 0
    t, Y1, Y2 = pseudo_dados()
    a = time()
    record = ["i,S(u)"]
    while (i < i_max):
        print(f"Passo: {i+1}")
        # Step 1
        T1, T2 = problema_direto(u, T1=1.0, T2=1.0)

        # Step 2
        S_g = objective_functional(T1, T2, Y1, Y2)
        record.append(f"{i+1}, {np.around(S_g, 3)}")
        
        if S_g < eps:
            return t, u
        # Step 3: solve adjoint problem. T e Y serão invertidos dentro da função.
        L1, L2 = problema_adjunto(L1_0, L2_0, T1, T2, Y1, Y2)
        # Lambdas retornam já invertidos

        # Step 4: compute the gradient of S
        grad = -L1

        # Step 5: compute yk, psik and dk. Use all four formulations
        d_k = direction_of_descent(i, grad, grad_old, dk_old, version='fs')
        grad_old = grad
        
        # Step 6: set Deltau equal to dk and solve the sensitivity problem
        delta_u = d_k
        delta_T1, delta_T2 = problema_sensibilidade(delta_u, T1, T2, DT1=0.0, DT2=0.0)
        
        # Step 7: calculate beta
        beta_k = passo(T1, T2, Y1, Y2, delta_T1, delta_T2)
        
        # Step 8: calculate u_new
        u = u + beta_k * d_k
        dk_old = d_k
        i += 1
    c = time()
    print(f"S_g final: {S_g}")
    print(f"Nº de iterações: {i+1}")
    print(f"Tempo total: {c - a} seg")
    return t, u

if __name__ == "__main__":
    print("Iniciando Otimização...")
    
    # 1. Roda o Gradiente Conjugado
    # Retorna o eixo do tempo e o vetor u otimizado
    u_0 = np.zeros(N)
    t_axis, u_estimado = gradiente_conjugado(0.005, u_0=u_0)
    
    print("\nOtimização Concluída. Gerando gráficos...")

    # 2. Gera os dados finais para plotagem
    # Calcula as temperaturas finais com o 'u' otimizado
    T1_final, T2_final = problema_direto(u_estimado, T1=1.0, T2=1.0)
    
    # Recupera os dados "medidos" (ruidosos) usados na otimização
    # Nota: pseudo_dados retorna (times, Y1, Y2)
    _, Y1_medido, Y2_medido = pseudo_dados()
    
    # Recupera o chute inicial de u para comparação
    u_inicial = u_0

    # --- NOVO: Gera o u(t) Real (Analítico) para comparação ---
    # Usa a função u_of_t definida no código para criar o vetor exato
    u_real = np.array([u_of_t(t) for t in t_axis])

    # 3. Configuração dos Plots (3 linhas, 1 coluna)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- Plot 1: Sensor 1 (T1 vs Y1) ---
    axs[0].plot(t_axis, Y1_medido, 'ro', markersize=3, alpha=0.4, label='Y1 (Medido)')
    axs[0].plot(t_axis, T1_final, 'b-', linewidth=1.5, label='T1 (Estimado)')
    axs[0].set_ylabel('T1 (°C)')
    axs[0].set_title('Sensor 1: Ajuste aos Dados')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Sensor 2 (T2 vs Y2) ---
    axs[1].plot(t_axis, Y2_medido, 'ro', markersize=3, alpha=0.4, label='Y2 (Medido)')
    axs[1].plot(t_axis, T2_final, 'g-', linewidth=1.5, label='T2 (Estimado)')
    axs[1].set_ylabel('T2 (°C)')
    axs[1].set_title('Sensor 2: Ajuste aos Dados')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 3: Comparação da Fonte u(t) ---
    axs[2].plot(t_axis, u_inicial, 'k--', alpha=0.5, label='Chute Inicial')
    
    # Plota o u(t) Real (Alvo)
    axs[2].plot(t_axis, u_real, 'c:', linewidth=2.5, label='u(t) Real (Analítico)')
    
    # Plota o u(t) Estimado pelo algoritmo
    axs[2].plot(t_axis, u_estimado, 'm-', linewidth=2, label='u(t) Recuperado')
    
    axs[2].set_xlabel('Tempo (s)')
    axs[2].set_ylabel('Fonte de Calor u(t)')
    axs[2].set_title('Resultado da Otimização: Fonte Recuperada vs Real')
    axs[2].legend(loc='lower right')
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()