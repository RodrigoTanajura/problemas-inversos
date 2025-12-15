import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# --- Globals and Constants ---
h = 0.005
tc = 0.5
t_inicial = 0.0
t_final = 2.0
w_real_noise = 0.025
N = int((t_final - t_inicial) / h)

# True Parameters
P1_REAL, P2_REAL, P3_REAL, P4_REAL = 1.417, 0.25, 15.0, 0.3

# --- Helper Functions ---
def a_of_T2(T2, P2, P3, P4):
    return P2 + np.exp(-(P3 * (T2 - P4))**2)

def u_of_t_real(t):
    u = -(P1_REAL) / tc
    return u if t < tc else 0.0

def f1(T1, T2, u_val): return u_val - (T1 - T2)
def f2(T1, T2, P2, P3, P4): return (T1 - T2) / a_of_T2(T2, P2, P3, P4)

def rk4_step_internal(T1n, T2n, u_val, P2, P3, P4, dt):
    k11 = f1(T1n, T2n, u_val)
    k12 = f2(T1n, T2n, P2, P3, P4)
    T1_k2, T2_k2 = T1n + 0.5 * dt * k11, T2n + 0.5 * dt * k12
    k21, k22 = f1(T1_k2, T2_k2, u_val), f2(T1_k2, T2_k2, P2, P3, P4)
    T1_k3, T2_k3 = T1n + 0.5 * dt * k21, T2n + 0.5 * dt * k22
    k31, k32 = f1(T1_k3, T2_k3, u_val), f2(T1_k3, T2_k3, P2, P3, P4)
    T1_k4, T2_k4 = T1n + dt * k31, T2n + dt * k32
    k41, k42 = f1(T1_k4, T2_k4, u_val), f2(T1_k4, T2_k4, P2, P3, P4)
    T1_new = T1n + (dt / 6.0) * (k11 + 2 * k21 + 2 * k31 + k41)
    T2_new = T2n + (dt / 6.0) * (k12 + 2 * k22 + 2 * k32 + k42)
    return T1_new, T2_new

def pseudo_dados():
    rng = np.random.default_rng(123)
    times = np.arange(t_inicial, t_final + h, h)
    T1_true, T2_true, u_true = [], [], []
    curr_T1, curr_T2 = 1.0, 1.0
    for t in times:
        u_val = u_of_t_real(t)
        u_true.append(u_val)
        T1_true.append(curr_T1)
        T2_true.append(curr_T2)
        curr_T1, curr_T2 = rk4_step_internal(curr_T1, curr_T2, u_val, P2_REAL, P3_REAL, P4_REAL, h)
    
    T1_true = np.array(T1_true)
    T2_true = np.array(T2_true)
    
    T1_meas = T1_true + rng.normal(0, w_real_noise, len(T1_true))
    T2_meas = T2_true + rng.normal(0, w_real_noise, len(T2_true))
    
    return times, T1_meas, T2_meas, np.array(u_true), T1_true, T2_true

# --- EKF Implementation ---
def get_jacobian_F(x_est, dt, P2, P3, P4):
    T1, T2 = x_est[0, 0], x_est[1, 0]
    C_val = a_of_T2(T2, P2, P3, P4)
    term_exp = np.exp(-(P3 * (T2 - P4))**2)
    dC_dT2 = -2 * (P3**2) * (T2 - P4) * term_exp
    
    F = np.eye(3)
    F[0, 0], F[0, 1], F[0, 2] = 1 - dt, dt, dt
    F[1, 0] = dt / C_val
    deriv_inner = (-1.0 / C_val) - ((T1 - T2) * dC_dT2) / (C_val**2)
    F[1, 1] = 1.0 + dt * deriv_inner
    return F

def run_ekf(times, T1_meas, T2_meas, Q_diag, R_sigma):
    n_steps = len(times)
    x = np.array([[1.0], [1.0], [0.0]]) 
    P = np.diag([0.01, 0.01, 1.0])      
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) 
    
    R = np.array([[R_sigma**2, 0.0], [0.0, R_sigma**2]])
    Q = np.diag(Q_diag)

    x_history = []
    P_history = []
    y_history = [] 

    for i in range(n_steps):
        u_curr = x[2, 0]
        T1_pred, T2_pred = rk4_step_internal(x[0, 0], x[1, 0], u_curr, P2_REAL, P3_REAL, P4_REAL, h)
        x_pred = np.array([[T1_pred], [T2_pred], [u_curr]])
        F = get_jacobian_F(x, h, P2_REAL, P3_REAL, P4_REAL)
        P_pred = F @ P @ F.T + Q
        
        z = np.array([[T1_meas[i]], [T2_meas[i]]])
        y = z - (H @ x_pred) 
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + (K @ y)
        P = (np.eye(3) - (K @ H)) @ P_pred
        
        x_history.append(x.flatten())
        P_history.append(P)
        y_history.append(y.flatten())

    return np.array(x_history), np.array(P_history), np.array(y_history)

# --- Main Plotting Function (Updated Labels) ---
def plot_ekf_results(times, T1_m, T2_m, u_real, T1_true, T2_true, 
                     x_hist, P_hist, y_hist, Q_diag, R_sigma, filename_suffix=""):
    
    T1_est = x_hist[:, 0]
    T2_est = x_hist[:, 1]
    u_est = x_hist[:, 2]
    
    u_std = np.sqrt(P_hist[:, 2, 2])
    u_lower = u_est - 1.96 * u_std
    u_upper = u_est + 1.96 * u_std

    rmse_T1 = np.sqrt(np.mean((T1_est - T1_m)**2))
    rmse_T2 = np.sqrt(np.mean((T2_est - T2_m)**2))

    COLOR_T1, COLOR_T2 = '#083464', '#880424'
    
    fig = plt.figure(figsize=(14, 11))
    # Increased hspace to 0.5 to allow room for the (a), (b)... labels below the X label
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1], hspace=0.35) 
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[0:2, 1])
    ax_text = fig.add_subplot(gs[2, 1])
    ax_text.axis('off')

    # Shared parameters for text positioning
    # y = -0.35 in transAxes puts it below the x-label (which is usually at ~ -0.15)
    lbl_y_pos = -0.20 

    # --- Plot T1 (a) ---
    ax1.plot(times, T1_m, 'x', color=COLOR_T1, markersize=5, alpha=0.3, label='Medidas T1')
    ax1.plot(times, T1_true, '-', color='black', linewidth=1.5, alpha=0.7, label='Exato T1')
    ax1.plot(times, T1_est, '.', color=COLOR_T1, markersize=3, label='EKF Est. T1')
    
    ax1.set_ylabel('T1 (°C)')
    ax1.set_xlabel('Tempo (s)')
    # Label (a) centered below
    ax1.text(0.5, lbl_y_pos, '(a)', transform=ax1.transAxes, 
             ha='center', va='top', fontweight='bold', fontsize=12)
    
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # --- Plot T2 (b) ---
    ax2.plot(times, T2_m, 'x', color=COLOR_T2, markersize=5, alpha=0.3, label='Medidas T2')
    ax2.plot(times, T2_true, '-', color='black', linewidth=1.5, alpha=0.7, label='Exato T2')
    ax2.plot(times, T2_est, '.', color=COLOR_T2, markersize=3, label='EKF Est. T2')
    
    ax2.set_ylabel('T2 (°C)')
    ax2.set_xlabel('Tempo (s)')
    # Label (b) centered below
    ax2.text(0.5, lbl_y_pos, '(b)', transform=ax2.transAxes, 
             ha='center', va='top', fontweight='bold', fontsize=12)
    
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # --- Plot u(t) (c) ---
    ax3.plot(times, u_real, '-', color='dimgray', linewidth=3.0, label='Real u(t)')
    ax3.plot(times, u_est, 'k-', linewidth=1.5, label='EKF Est. u(t)')
    ax3.fill_between(times, u_lower, u_upper, color='gray', alpha=0.4, label='IC 95%')
    
    ax3.set_ylabel('Fonte u(t)')
    ax3.set_xlabel('Tempo (s)')
    # Label (c) centered below
    ax3.text(0.5, lbl_y_pos, '(c)', transform=ax3.transAxes, 
             ha='center', va='top', fontweight='bold', fontsize=12)
    
    ax3.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(min(u_real)*1.5, max(u_real)+1.0)

    # --- Plot Innovations (d) ---
    y_scaled = y_hist * 1000
    R_scaled = R_sigma * 1000
    
    ax4.set_title("Histórico de Inovações (Medida - Predição)")
    ax4.plot(times, y_scaled[:, 0], color=COLOR_T1, alpha=0.6, linewidth=1, label='Inovação T1')
    ax4.plot(times, y_scaled[:, 1], color=COLOR_T2, alpha=0.6, linewidth=1, label='Inovação T2')
    
    ax4.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axhline(2*R_scaled, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='±2σ Ruído Esp.')
    ax4.axhline(-2*R_scaled, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax4.set_ylabel(r'Residual ($\times 10^{-3}$ °C)') 
    ax4.set_xlabel('Tempo (s)')
    # Label (d) centered below. Slightly adjusted y if needed due to layout, but -0.15 is relative to ax4
    # Since ax4 is taller (spans 2 rows), -0.15 might look different. 
    # Let's align it visually with the others (approx). -0.15 is safer here as the plot is taller.
    # Actually, to be consistent with "below the legend", let's use a specific offset.
    ax4.text(0.5, -0.10, '(d)', transform=ax4.transAxes, 
             ha='center', va='top', fontweight='bold', fontsize=12)
    
    ax4.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax4.grid(True, alpha=0.3)

    # --- TABELA ---
    q_t_str = f"{Q_diag[0]:.1e}"
    q_u_str = f"{Q_diag[2]:.1e}"
    r_str = f"{R_sigma:.3f}"

    dados_input = [
        ('Q(T1,T2) Diag', q_t_str),
        ('Q(u) Diag', q_u_str),
        ('R Sigma (W)', r_str),
        ('Passo', f"{h} s"),
    ]
    
    dados_output = [
        ('RMSE T1', f"{rmse_T1:.4f} °C"),
        ('RMSE T2', f"{rmse_T2:.4f} °C"),
        ('Final u_std', f"{u_std[-1]:.4f}"),
        ('Est. u(t<tc)', f"{np.mean(u_est[times<tc]):.3f}"),
    ]
    
    W1, W2 = 13, 22
    SEP_V, SEP_MID = "│", " ║ "
    LINE_L, LINE_R = "─" * (W1 + 3 + W2), "─" * (W1 + 3 + W2)
    
    def fmt_cell(label, val): return f"{label:>{W1}} {SEP_V} {val:<{W2}}"

    header = f"{'EKF INPUTS':^{len(LINE_L)}}{SEP_MID}{'EKF OUTPUTS':^{len(LINE_R)}}"
    subhead = f"{fmt_cell('Parâmetro', 'Valor')}{SEP_MID}{fmt_cell('Métrica', 'Valor')}"
    separator = f"{LINE_L}{SEP_MID}{LINE_R}"
    
    body_lines = []
    max_rows = max(len(dados_input), len(dados_output))
    
    for i in range(max_rows):
        left_txt = fmt_cell(*dados_input[i]) if i < len(dados_input) else " " * (W1 + 3 + W2)
        right_txt = fmt_cell(*dados_output[i]) if i < len(dados_output) else " " * (W1 + 3 + W2)
        body_lines.append(f"{left_txt}{SEP_MID}{right_txt}")
        
    texto_final = f"{header}\n{subhead}\n{separator}\n" + "\n".join(body_lines)
    
    ax_text.text(0.5, 0.5, texto_final, fontsize=9, fontname='DejaVu Sans Mono', 
                 verticalalignment='center', horizontalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor="gray", alpha=0.9))
    
    plt.subplots_adjust(bottom=0.08, hspace=0.55, wspace=0.15)
    output_folder = "kalman"
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"EKF_Styled_{filename_suffix}.png")
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"Gráfico salvo: {filename}")
    plt.close(fig)

if __name__ == "__main__":
    print("Gerando dados Exatos + Ruidosos...")
    times, T1_m, T2_m, u_real, T1_true, T2_true = pseudo_dados()
    
    q_u_tune = 2.5e-2
    Q_diag_current = [1e-4, 1e-4, q_u_tune]
    
    filename_suffix = f"_Qu={q_u_tune:.1e}_R={w_real_noise:.3f}"
    
    print(f"Rodando EKF...")
    x_hist, P_hist, y_hist = run_ekf(times, T1_m, T2_m, Q_diag_current, w_real_noise)
    
    plot_ekf_results(times, T1_m, T2_m, u_real, T1_true, T2_true, 
                     x_hist, P_hist, y_hist, Q_diag_current, w_real_noise, 
                     filename_suffix=filename_suffix)