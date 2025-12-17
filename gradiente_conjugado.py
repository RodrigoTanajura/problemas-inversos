import numpy as np
import os
from time import time
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from problema_direto_gc import (problema_direto, pseudo_dados, problema_adjunto, 
                                problema_sensibilidade, passo, u_of_t, N, w)

# --- Configurações Globais ---
L1_0 = 0
L2_0 = 0

def objective_functional(T1, T2, Y1, Y2):
    return ((np.sum((T1 - Y1)**2 + (T2 - Y2)**2))/(2*N))**(1/2)

def direction_of_descent(i, grad, grad_old, dk_old, dk_restart, version='fr'):
    # Inicialização padrão
    y_k = 0
    psi_k = 0
    dk_q = dk_restart 

    dg = grad - grad_old
    
    if i == 0: # Primeira iteração: Steepest Descent
        d_k = -grad
        dk_q = d_k 
        return d_k, dk_q

    if version == 'fr': # Fletcher-Reeves
        num = np.sum(grad**2)
        den = np.sum(grad_old**2)
        y_k = num / den if den != 0 else 0

    elif version == 'pr': # Polak-Ribiere
        num = np.sum(grad * dg)
        den = np.sum(grad_old**2)
        y_k = num / den if den != 0 else 0
        y_k = max(0, y_k) 

    elif version == 'hs': # Hestenes-Stiefel
        num = np.sum(grad * dg)
        den = np.sum(dk_old * dg)
        y_k = num / den if den != 0 else 0

    elif version == 'pb': # Powell-Beale Restart
        dot_gg_old = np.abs(np.sum(grad * grad_old))
        norm_g2 = np.sum(grad**2)
        restart_condition = dot_gg_old >= 0.2 * norm_g2
        den = np.sum(dk_old * dg)

        if restart_condition or den == 0:
            y_k = 0
            psi_k = 0
            d_k = -grad 
            dk_q = d_k 
            return d_k, dk_q 
        else:
            num_gamma = np.sum(grad * dg)
            y_k = num_gamma / den
            
            den_psi = np.sum(dk_restart * dg)
            if den_psi != 0:
                num_psi = np.sum(grad * dg)
                psi_k = num_psi / den_psi
            else:
                psi_k = 0
            dk_q = dk_restart 
            
    else:
        print("Método não selecionado! Usando Steepest Descent.")
        return -grad, dk_restart

    d_k = -grad + (y_k * dk_old) + (psi_k * dk_q)
    return d_k, dk_q

def gradiente_conjugado(eps, u_0, version='fr'):
    i = 0
    u = u_0.copy()
    i_max = 100
    grad_old = 1
    dk_old = np.zeros_like(u_0)
    dk_restart = np.zeros_like(u_0)
    
    t, Y1, Y2 = pseudo_dados()
    
    start_time = time()
    S_history = []
    
    # Loop principal
    while (i < i_max):
        # Step 1: Direct Problem
        T1, T2 = problema_direto(u, T1=1.0, T2=1.0)

        # Step 2: Objective Function
        S_g = objective_functional(T1, T2, Y1, Y2)
        S_history.append(S_g)
        
        # Verbose simples para acompanhar progresso
        if i % 10 == 0:
            print(f"Iter {i}: S(u) = {S_g:.5f}")

        if S_g < eps:
            break
            
        # Step 3: Adjoint Problem
        L1, L2 = problema_adjunto(L1_0, L2_0, T1, T2, Y1, Y2)

        # Step 4: Gradient
        grad = (-1) * L1

        # Step 5: Descent Direction
        d_k, dk_restart = direction_of_descent(i, grad, grad_old, dk_old, dk_restart, version=version)
        grad_old = grad
        
        # Step 6: Sensitivity Problem
        delta_u = d_k
        delta_T1, delta_T2 = problema_sensibilidade(delta_u, T1, T2, DT1=0.0, DT2=0.0)
        
        # Step 7: Step size (beta)
        beta_k = passo(T1, T2, Y1, Y2, delta_T1, delta_T2)
        
        # Step 8: Update
        u = u + beta_k * d_k
        dk_old = d_k
        i += 1
        
    end_time = time()
    elapsed = end_time - start_time

    # Cálculo final para garantir consistência dos retornos
    T1, T2 = problema_direto(u, T1=1.0, T2=1.0)
    S_final = objective_functional(T1, T2, Y1, Y2)
    
    print(f"\n--- Fim do GC ({version.upper()}) ---")
    print(f"S_g inicial: {S_history[0]:.4f}")
    print(f"S_g final:   {S_final:.4f}")
    print(f"Iterações:   {i}")
    print(f"Tempo total: {elapsed:.3f} seg")
    
    return t, u, S_history, elapsed, i

# --- Função de Plotagem Refatorada ---
def plot_gc_results(times, T1_m, T2_m, u_real, T1_true, T2_true, # <--- Added T_true args
                   T1_est, T2_est, u_est, 
                   S_history, elapsed_time, iterations, version_name, sigma_noise,
                   filename_suffix=""):
    
    # Métricas de Erro
    rmse_T1 = np.sqrt(np.mean((T1_est - T1_m)**2))
    rmse_T2 = np.sqrt(np.mean((T2_est - T2_m)**2))
    final_obj = S_history[-1]

    # Cores e Estilos
    COLOR_T1, COLOR_T2 = '#083464', '#880424'
    
    fig = plt.figure(figsize=(14, 15))
    
    # Layout igual ao EKF: 4 linhas, última dividida
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.15, width_ratios=[1, 1]) 
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    ax4 = fig.add_subplot(gs[3, 0])
    ax_text = fig.add_subplot(gs[3, 1])
    ax_text.axis('off')

    lbl_y_pos = -0.25 

    # --- Plot T1 (a) ---
    ax1.plot(times, T1_m, 'x', color=COLOR_T1, markersize=5, alpha=0.5, label='Medidas T1')
    # Adicionado: Curva exata em cinza
    ax1.plot(times, T1_true, '-', color='gray', linewidth=1.5, alpha=0.7, label='Exato T1')
    # Alterado: Estilo para ponto (.)
    ax1.plot(times, T1_est, '.', color=COLOR_T1, markersize=3, label='GC Est. T1')
    
    ax1.set_ylabel('T1 (°C)')
    ax1.set_xlabel('Tempo (s)')
    ax1.text(0.5, lbl_y_pos, '(a)', transform=ax1.transAxes, ha='center', va='top', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # --- Plot T2 (b) ---
    ax2.plot(times, T2_m, 'x', color=COLOR_T2, markersize=5, alpha=0.5, label='Medidas T2')
    # Adicionado: Curva exata em cinza
    ax2.plot(times, T2_true, '-', color='gray', linewidth=1.5, alpha=0.7, label='Exato T2')
    # Alterado: Estilo para ponto (.)
    ax2.plot(times, T2_est, '.', color=COLOR_T2, markersize=3, label='GC Est. T2')

    ax2.set_ylabel('T2 (°C)')
    ax2.set_xlabel('Tempo (s)')
    ax2.text(0.5, lbl_y_pos, '(b)', transform=ax2.transAxes, ha='center', va='top', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # --- Plot u(t) (c) ---
    ax3.plot(times, u_real, '-', color='dimgray', linewidth=3.0, alpha=0.6, label='Exato u(t)')
    ax3.plot(times, u_est, 'k-', linewidth=1.5, label='GC Est. u(t)')
    
    ax3.set_ylabel('Fonte u(t)')
    ax3.set_xlabel('Tempo (s)')
    ax3.text(0.5, lbl_y_pos, '(c)', transform=ax3.transAxes, ha='center', va='top', fontweight='bold', fontsize=12)
    ax3.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    y_min, y_max = min(u_real), max(u_real)
    margin = (y_max - y_min) * 0.2 if y_max != y_min else 1.0
    ax3.set_ylim(y_min - margin, y_max + margin)

    # --- Plot Resíduos Finais (d) ---
    res_t1 = T1_est - T1_m
    res_t2 = T2_est - T2_m
    
    ax4.set_title("Resíduos Finais (Estimado - Medido)")
    # Alterado labels de Erro -> Resíduo
    ax4.plot(times, res_t1, color=COLOR_T1, alpha=0.6, linewidth=1, label='Resíduo T1')
    ax4.plot(times, res_t2, color=COLOR_T2, alpha=0.6, linewidth=1, label='Resíduo T2')
    ax4.axhline(0, color='k', linestyle='--', linewidth=1, alpha=1)
    
    ax4.axhline(2*sigma_noise, color='k', linestyle=':', linewidth=1, label=r'$\pm 2\sigma$ Ruído')
    ax4.axhline(-2*sigma_noise, color='k', linestyle=':', linewidth=1)
    
    ax4.set_ylabel(r'Resíduo (°C)') 
    ax4.set_xlabel('Tempo (s)')
    ax4.text(0.5, lbl_y_pos, '(d)', transform=ax4.transAxes, ha='center', va='top', fontweight='bold', fontsize=12)
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3)

    # --- TABELA DE DADOS ---
    s_final_str = f"{final_obj:.4f}"
    rmse1_str = f"{rmse_T1:.4f}"
    rmse2_str = f"{rmse_T2:.4f}"
    
    # Alterado: Organização dos dados e remoção de "(Dados)" e movimentação do Tempo
    dados_input = [
        ('Método', version_name.upper()), 
        ('Sigma', f"{sigma_noise}"), # Removeu (Dados)
        ('Iterações', f"{iterations}")
    ]
    dados_output = [
        ('S[u(t)] Final', s_final_str), 
        ('RMSE T1', rmse1_str), 
        ('RMSE T2', rmse2_str),
        ('Tempo (s)', f"{elapsed_time:.2f}") # Movido para Outputs
    ]
    
    W1, W2 = 14, 14
    SEP_V, SEP_MID = "│", "║"
    LINE_L, LINE_R = "─" * (W1 + 3 + W2), "─" * (W1 + 3 + W2)
    
    def fmt_cell(label, val): return f"{label:>{W1}} {SEP_V} {val:<{W2}}"

    # Alterado headers para INPUTS e OUTPUTS
    header = f"{'INPUTS':^{len(LINE_L)}}{SEP_MID}{'OUTPUTS':^{len(LINE_R)}}"
    subhead = f"{fmt_cell('Parâmetro', 'Valor')}{SEP_MID}{fmt_cell('Métrica', 'Valor')}"
    separator = f"{LINE_L}{SEP_MID}{LINE_R}"
    
    body_lines = []
    max_rows = max(len(dados_input), len(dados_output))
    for i in range(max_rows):
        left_txt = fmt_cell(*dados_input[i]) if i < len(dados_input) else " " * (W1 + 3 + W2)
        right_txt = fmt_cell(*dados_output[i]) if i < len(dados_output) else " " * (W1 + 3 + W2)
        body_lines.append(f"{left_txt}{SEP_MID}{right_txt}")
        
    texto_final = f"{header}\n{subhead}\n{separator}\n" + "\n".join(body_lines)
    
    ax_text.text(0.5, 0.5, texto_final, fontsize=12, fontname='DejaVu Sans Mono', 
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
    
    plt.subplots_adjust(bottom=0.06, top=0.97, left=0.08, right=0.97)
    
    output_folder = "gc_results"
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"GC_{filename_suffix}.png")
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Gráfico salvo: {filename}")
    plt.close(fig)

if __name__ == "__main__":
    print("Iniciando Otimização...")
    
    # 1. Configurações
    u_0 = np.zeros(N)
    version = 'pr' #hs pb pr fs
    
    # 2. Dados Medidos
    times, Y1_medido, Y2_medido = pseudo_dados()
    
    # 3. Rodar GC
    # w é o sigma importado de problema_direto_gc
    t, u_estimado, S_hist, elapsed, iters = gradiente_conjugado(w, u_0, version=version)
    
    # 4. Calcular respostas FINAIS Estimadas
    T1_final_est, T2_final_est = problema_direto(u_estimado, T1=1.0, T2=1.0)
    
    # 5. Obter DADOS EXATOS para plotagem
    u_real = np.array([u_of_t(ti) for ti in times])
    # Rodamos o problema direto com o u_real (sem ruído) para ter a referência exata
    T1_true, T2_true = problema_direto(u_real, T1=1.0, T2=1.0)
    
    print("\nGerando gráficos no estilo EKF...")
    
    filename_suffix = f"{version}_Iter={iters}"
    
    plot_gc_results(times, Y1_medido, Y2_medido, u_real, 
                    T1_true, T2_true, # Passando os exatos
                    T1_final_est, T2_final_est, u_estimado, 
                    S_hist, elapsed, iters, version, w,
                    filename_suffix=filename_suffix)