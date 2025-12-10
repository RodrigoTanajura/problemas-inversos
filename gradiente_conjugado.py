import numpy as np
from time import time
from matplotlib import pyplot as plt
from problema_direto_gc import problema_direto, pseudo_dados, problema_adjunto, problema_sensibilidade, passo, u_of_t
from problema_direto_gc import N, w

L1_0 = 0
L2_0 = 0

def objective_functional(T1, T2, Y1, Y2):
    return ((np.sum((T1 - Y1)**2 + (T2 - Y2)**2))/(2*N))**(1/2)

def direction_of_descent(i, grad, grad_old, dk_old, dk_restart, version='fr'):
    # Inicialização padrão
    y_k = 0
    psi_k = 0
    dk_q = dk_restart # Recupera o valor antigo por padrão

    # Diff de gradientes (usado em PR, HS e PB)
    dg = grad - grad_old
    
    if i == 0: # Primeira iteração: Steepest Descent
        d_k = -grad
        dk_q = d_k # O primeiro passo é um restart implícito
        return d_k, dk_q

    if version == 'fr': # Fletcher-Reeves
        num = np.sum(grad**2)
        den = np.sum(grad_old**2)
        y_k = num / den if den != 0 else 0

    elif version == 'pr': # Polak-Ribiere
        num = np.sum(grad * dg)
        den = np.sum(grad_old**2)
        y_k = num / den if den != 0 else 0
        y_k = max(0, y_k) # Reset automático do PR se y_k < 0

    elif version == 'hs': # Hestenes-Stiefel
        num = np.sum(grad * dg)
        den = np.sum(dk_old * dg)
        y_k = num / den if den != 0 else 0

    elif version == 'pb': # Powell-Beale Restart
        # 1. Critério de Restart (Ortogonalidade perdida ou pouco progresso)
        # Condição: |g_k . g_old| >= 0.2 ||g_k||^2
        dot_gg_old = np.abs(np.sum(grad * grad_old))
        norm_g2 = np.sum(grad**2)
        
        restart_condition = dot_gg_old >= 0.2 * norm_g2
        
        # Denominador comum para gamma e psi (igual ao HS)
        den = np.sum(dk_old * dg)

        if restart_condition or den == 0:
            # --- RESTART ---
            # Reseta para a direção do gradiente negativo (ou steepest descent)
            # E definimos este momento como o novo "q" (tempo de restart)
            y_k = 0
            psi_k = 0
            d_k = -grad 
            dk_q = d_k # Atualizamos a memória do restart para o passo atual
            return d_k, dk_q # Retorna imediatamente
        
        else:
            # --- CONTINUAÇÃO (Conjugação Padrão PB) ---
            # Gamma (y_k) é igual ao Hestenes-Stiefel
            num_gamma = np.sum(grad * dg)
            y_k = num_gamma / den
            
            # Psi (psi_k) conecta com o vetor do último restart (dk_restart)
            # Denominador do psi usa o dk_restart
            den_psi = np.sum(dk_restart * dg)
            
            if den_psi != 0:
                num_psi = np.sum(grad * dg)
                psi_k = num_psi / den_psi
            else:
                psi_k = 0
                
            # Na fórmula do PB, dk_q é usado multiplicando psi_k
            dk_q = dk_restart # Mantém o antigo, não atualiza
            
    else:
        print("Método não selecionado! Usando Steepest Descent.")
        return -grad, dk_restart

    # Fórmula Geral de Atualização: d_k = -g + y_k*d_old + psi_k*d_restart
    d_k = -grad + (y_k * dk_old) + (psi_k * dk_q)
    
    return d_k, dk_q

def gradiente_conjugado(eps, u_0, version='fr'):
    i = 1
    u = u_0
    i_max = 100
    grad_old = 1
    dk_old = np.zeros_like(u_0)
    dk_restart = np.zeros_like(u_0)
    t, Y1, Y2 = pseudo_dados()
    a = time()
    record = ["i,S(u)"]
    while (i < i_max):
        # Step 1
        T1, T2 = problema_direto(u, T1=1.0, T2=1.0)

        # Step 2
        S_g = objective_functional(T1, T2, Y1, Y2)
        record.append(f"{i}, {np.around(S_g, 3)}")
        print(S_g)
        if S_g < eps:
            return t, u
        # Step 3: solve adjoint problem. T e Y serão invertidos dentro da função.
        L1, L2 = problema_adjunto(L1_0, L2_0, T1, T2, Y1, Y2)
        # Lambdas retornam já invertidos

        # Step 4: compute the gradient of S
        grad = (-1) * L1

        # Step 5: compute yk, psik and dk. Use all four formulations
        # --- Chamada atualizada ---
        d_k, dk_restart = direction_of_descent(i, grad, grad_old, dk_old, dk_restart, version=version)
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

    T1, T2 = problema_direto(u_0, T1=1.0, T2=1.0)
    S_g_0 = objective_functional(T1, T2, Y1, Y2)
    print(f"S_g inicial: {np.around(S_g_0, 4)}")
    print(f"S_g final: {np.around(S_g, 4)}")
    print(f"Sigma: {w}")
    print(f"Nº de iterações: {i}")
    print(f"Tempo total: {c - a} seg")
    return t, u

if __name__ == "__main__":
    print("Iniciando Otimização...")
    
    # 1. Roda o Gradiente Conjugado
    u_0 = np.zeros(N)
    
    times, Y1_medido, Y2_medido = pseudo_dados()
    t, u_estimado = gradiente_conjugado(w, u_0, 'fr')
    t_axis = times

    print("\nOtimização Concluída. Gerando gráficos...")

    # 2. Gera os dados para plotagem
    
    # A) Temperaturas Finais (com u Otimizado)
    T1_final, T2_final = problema_direto(u_estimado, T1=1.0, T2=1.0)
    
    # B) Temperaturas Iniciais (com u Chute Inicial = 0)
    T1_inicial, T2_inicial = problema_direto(u_0, T1=1.0, T2=1.0)
    
    # C) Dados Reais (Analíticos) e Medidos
    if 'Y1_medido' not in locals():
        _, Y1_medido, Y2_medido = pseudo_dados()

    # D) u(t) Real para comparação
    u_real = np.array([u_of_t(t) for t in t_axis])

    # 3. Configuração dos Plots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # --- Plot 1: Sensor 1 (T1) ---
    axs[0].plot(t_axis, Y1_medido, 'ro', markersize=3, alpha=0.3, label='Y1 (Medido)')
    axs[0].plot(t_axis, T1_inicial, 'k--', linewidth=1.5, alpha=0.6, label='T1 (Inicial/Chute)') # <--- Adicionado
    axs[0].plot(t_axis, T1_final, 'b-', linewidth=2.0, label='T1 (Estimado)')
    axs[0].set_ylabel('T1 (°C)')
    axs[0].set_title('Sensor 1: Ajuste aos Dados')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Sensor 2 (T2) ---
    axs[1].plot(t_axis, Y2_medido, 'ro', markersize=3, alpha=0.3, label='Y2 (Medido)')
    axs[1].plot(t_axis, T2_inicial, 'k--', linewidth=1.5, alpha=0.6, label='T2 (Inicial/Chute)') # <--- Adicionado
    axs[1].plot(t_axis, T2_final, 'g-', linewidth=2.0, label='T2 (Estimado)')
    axs[1].set_ylabel('T2 (°C)')
    axs[1].set_title('Sensor 2: Ajuste aos Dados')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 3: Fonte u(t) ---
    axs[2].plot(t_axis, u_0, 'k--', alpha=0.5, label='Chute Inicial (u=0)')
    axs[2].plot(t_axis, u_real, 'c:', linewidth=2.5, label='u(t) Real (Analítico)')
    axs[2].plot(t_axis, u_estimado, 'm-', linewidth=2, label='u(t) Estimado')
    
    axs[2].set_xlabel('Tempo (s)')
    axs[2].set_ylabel('Fonte de Calor u(t)')
    axs[2].set_title('Resultado: Fonte Estimada vs Real')
    axs[2].legend(loc='lower right')
    axs[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()