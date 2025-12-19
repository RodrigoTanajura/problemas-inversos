import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import os

# --- CONFIGURAÇÃO GLOBAL ---
plt.rcParams.update({
    "text.usetex": False, 
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
    "font.size": 12
})

# --- Globals and Constants ---
h = 0.005
tc = 0.5
t_inicial = 0.0
t_final = 2.0
w_real_noise_data = 0.025 

# True Parameters (used for generating data and calculating error)
P1_REAL = 1.417
# P2, P3, P4 are now ESTIMATED, but we keep real values for ground truth comparison
P_REAL_VALS = [0.25, 15.0, 0.3] 

# --- Helper Functions (Physics) ---
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
    
    # Use REAL parameters to generate synthetic data
    p2, p3, p4 = P_REAL_VALS 
    
    for t in times:
        u_val = u_of_t_real(t)
        u_true.append(u_val)
        T1_true.append(curr_T1)
        T2_true.append(curr_T2)
        curr_T1, curr_T2 = rk4_step_internal(curr_T1, curr_T2, u_val, p2, p3, p4, h)
    
    T1_true = np.array(T1_true)
    T2_true = np.array(T2_true)
    
    T1_meas = T1_true + rng.normal(0, w_real_noise_data, len(T1_true))
    T2_meas = T2_true + rng.normal(0, w_real_noise_data, len(T2_true))
    
    return times, T1_meas, T2_meas, np.array(u_true), T1_true, T2_true

# --- SIR Algorithm ---
def SIR(n_particulas, sigma_rw_u, R_sigma, prior_means, sigma_prior_mult, rw_mult_params):
    # State Vector: [T1, T2, u, P2, P3, P4] (Size 6)
    
    # 1. Initialize Particles
    particles = np.zeros((n_particulas, 6))
    particles[:, 0] = 1.0  # T1
    particles[:, 1] = 1.0  # T2
    particles[:, 2] = 0.0  # u
    
    # Initialize Parameters based on Prior Gaussian N(mean, mean*sigma_mult)
    for i, (mu, sigma_mult) in enumerate(zip(prior_means, sigma_prior_mult)):
        sigma = mu * sigma_mult
        particles[:, 3 + i] = np.random.normal(mu, sigma, n_particulas)

    H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    R = np.array([[R_sigma**2, 0.0], [0.0, R_sigma**2]])
    R_inv = np.linalg.inv(R)
    
    times, Y1, Y2, _, _, _ = pseudo_dados()
    
    results_mean = []
    results_std  = [] 
    neff_history = [] 
    
    for k, t in enumerate(times):
        particles_pred = np.zeros_like(particles)
        log_weights = np.zeros(n_particulas) 
        current_meas = np.array([Y1[k], Y2[k]])
        
        # Determine parameter RW standard deviation (dynamic based on mean or current value)
        # Using fixed step based on initial mean for stability
        rw_steps = [m * rw_mult_params for m in prior_means]

        for i in range(n_particulas):
            # A. Evolution (Prior)
            # 1. Evolve 'u'
            u_prev = particles[i, 2]
            u_new = u_prev + np.random.normal(0, sigma_rw_u)
            
            # 2. Evolve Parameters (Random Walk)
            p2_prev, p3_prev, p4_prev = particles[i, 3], particles[i, 4], particles[i, 5]
            p2_new = p2_prev + np.random.normal(0, rw_steps[0])
            p3_new = p3_prev + np.random.normal(0, rw_steps[1])
            p4_new = p4_prev + np.random.normal(0, rw_steps[2])
            
            # 3. Evolve Physics (T1, T2) using NEW parameters and NEW u
            t1_prev, t2_prev = particles[i, 0], particles[i, 1]
            t1_new, t2_new = rk4_step_internal(t1_prev, t2_prev, u_new, p2_new, p3_new, p4_new, h)
            
            particles_pred[i, :] = [t1_new, t2_new, u_new, p2_new, p3_new, p4_new]
            
            # B. Likelihood
            y_pred = particles_pred[i, :2] 
            error = current_meas - y_pred
            log_weights[i] = -0.5 * (error.T @ R_inv @ error)
            
        # Normalize
        max_log_w = np.max(log_weights)
        weights = np.exp(log_weights - max_log_w)
        sum_weights = np.sum(weights)
        weights_norm = weights / sum_weights
        
        # Stats
        x_mean = np.average(particles_pred, weights=weights_norm, axis=0)
        variance = np.average((particles_pred - x_mean)**2, weights=weights_norm, axis=0)
        x_std = np.sqrt(variance)
        n_eff = 1.0 / np.sum(weights_norm**2)
        
        results_mean.append(x_mean)
        results_std.append(x_std)
        neff_history.append(n_eff)
        
        # Resampling
        particles_resampled = np.zeros_like(particles_pred)
        c = np.cumsum(weights_norm)
        u1 = np.random.uniform(0, 1/n_particulas)
        i_idx = 0
        for j in range(n_particulas):
            u_j = u1 + (j / n_particulas)
            while u_j > c[i_idx] and i_idx < n_particulas - 1:
                i_idx += 1
            particles_resampled[j, :] = particles_pred[i_idx, :]
            
        particles = particles_resampled 

    return times, np.array(results_mean), np.array(results_std), np.array(neff_history)

# --- Main Execution & Plotting ---
def main(N_PARTICLES, SIGMA_RW_U, R_SIGMA, PRIOR_MEANS, SIGMA_PRIOR_MULT, RW_MULT_PARAMS):
    
    start_time = time.time()
    times, res_mean, res_std, neff_hist = SIR(N_PARTICLES, SIGMA_RW_U, R_SIGMA, PRIOR_MEANS, SIGMA_PRIOR_MULT, RW_MULT_PARAMS)
    end_time = time.time()
    exec_time_seconds = end_time - start_time
    
    # Extract Data
    _, T1_m, T2_m, u_real, T1_true, T2_true = pseudo_dados()
    
    T1_est = res_mean[:, 0]
    T2_est = res_mean[:, 1]
    u_est  = res_mean[:, 2]
    
    # Extract Estimates for Parameters (Take mean of last 10% of simulation for stability)
    idx_stable = int(len(times) * 0.9)
    p2_est_final = np.mean(res_mean[idx_stable:, 3])
    p3_est_final = np.mean(res_mean[idx_stable:, 4])
    p4_est_final = np.mean(res_mean[idx_stable:, 5])
    
    est_params = [p2_est_final, p3_est_final, p4_est_final]
    
    # CI Bounds
    T1_upper, T1_lower = T1_est + 1.96*res_std[:,0], T1_est - 1.96*res_std[:,0]
    T2_upper, T2_lower = T2_est + 1.96*res_std[:,1], T2_est - 1.96*res_std[:,1]
    u_upper,  u_lower  = u_est  + 1.96*res_std[:,2], u_est  - 1.96*res_std[:,2]
    
    rmse_u = np.sqrt(np.mean((u_est - u_real)**2))
    
    # --- PLOTTING ---
    COLOR_T1, COLOR_T2 = '#083464', '#880424'
    COLOR_U = 'green'
    
    fig = plt.figure(figsize=(14, 15)) 
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.15, width_ratios=[0.9, 1]) 
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    ax4 = fig.add_subplot(gs[3, 0])      
    ax_text = fig.add_subplot(gs[3, 1])  
    ax_text.axis('off')
    
    lbl_y_pos = -0.25 

    # Plot T1 (a)
    ax1.plot(times, T1_m, 'x', color=COLOR_T1, markersize=5, alpha=0.7, label='Medidas T1')
    ax1.plot(times, T1_true, '-', color='black', linewidth=1.5, alpha=0.7, label='Exato T1')
    ax1.plot(times, T1_est, '.', color=COLOR_T1, markersize=3, label='SIR Est. T1')
    # ax1.fill_between(times, T1_lower, T1_upper, color=COLOR_T1, alpha=0.1)
    ax1.set_ylabel('T1 (°C)')
    ax1.text(0.5, lbl_y_pos, '(a)', transform=ax1.transAxes, ha='center', va='top', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot T2 (b)
    ax2.plot(times, T2_m, 'x', color=COLOR_T2, markersize=5, alpha=0.7, label='Medidas T2')
    ax2.plot(times, T2_true, '-', color='black', linewidth=1.5, alpha=0.7, label='Exato T2')
    ax2.plot(times, T2_est, '.', color=COLOR_T2, markersize=3, label='SIR Est. T2')
    # ax2.fill_between(times, T2_lower, T2_upper, color=COLOR_T2, alpha=0.1)
    ax2.set_ylabel('T2 (°C)')
    ax2.text(0.5, lbl_y_pos, '(b)', transform=ax2.transAxes, ha='center', va='top', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # Plot u(t) (c)
    ax3.plot(times, u_real, '-', color='dimgray', linewidth=3.0, label='Exato u(t)')
    ax3.plot(times, u_est, 'k-', linewidth=1.5, label='SIR Est. u(t)')
    ax3.fill_between(times, u_lower, u_upper, color='gray', alpha=0.4, label='IC 95%')
    ax3.set_ylabel('Fonte u(t)')
    ax3.set_xlabel('Tempo (s)')
    ax3.text(0.5, lbl_y_pos, '(c)', transform=ax3.transAxes, ha='center', va='top', fontweight='bold', fontsize=12)
    ax3.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    
    # Plot Diagnostic (d)
    ax4.plot(times, neff_hist, '-', color='black', linewidth=1.5, label='N_eff')
    ax4.axhline(y=N_PARTICLES, color='gray', linestyle='--', alpha=0.5, label='Max N')
    ax4.set_ylabel('Tamanho da Amostra Efetiva')
    ax4.set_xlabel('Tempo (s)')
    ax4.set_ylim(0, N_PARTICLES * 1.1)
    ax4.legend(loc='lower right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.text(0.5, lbl_y_pos, '(d)', transform=ax4.transAxes, ha='center', va='top', fontweight='bold', fontsize=12)

    # --- TABLE GENERATION ---
    m, s_time = divmod(exec_time_seconds, 60)
    time_str = f"{int(m)}m {int(s_time)}s"
    
    # Input Data Formatting
    dados_input = [
        ('Partículas', str(N_PARTICLES)),
        ('Sigma RW u', str(SIGMA_RW_U)),
        ('Sigma RW Pk', f"{RW_MULT_PARAMS} x Média(k)"),
        ('Sigma R', f"{R_SIGMA:.3f}"),
        ('P2 Prior', f"N({PRIOR_MEANS[0]:.3f}, {SIGMA_PRIOR_MULT[0]*PRIOR_MEANS[0]:.3f})"),
        ('P3 Prior', f"N({PRIOR_MEANS[1]:.2f}, {SIGMA_PRIOR_MULT[1]*PRIOR_MEANS[1]:.3f})"),
        ('P4 Prior', f"N({PRIOR_MEANS[2]:.3f}, {SIGMA_PRIOR_MULT[2]*PRIOR_MEANS[2]:.3f})"),
    ]
    
    # Calculate % error for params
    err_p2 = abs(est_params[0] - P_REAL_VALS[0])/P_REAL_VALS[0] * 100
    err_p3 = abs(est_params[1] - P_REAL_VALS[1])/P_REAL_VALS[1] * 100
    err_p4 = abs(est_params[2] - P_REAL_VALS[2])/P_REAL_VALS[2] * 100

    dados_output = [
        ('RMSE u(t)', f"{rmse_u:.4f}"),
        ('Tempo', time_str),
        ('N_eff Mean', f"{np.mean(neff_hist):.0f}"),
        ('P2 Est.', f"{est_params[0]:.3f} ({err_p2:.1f}%)"),
        ('P3 Est.', f"{est_params[1]:.2f} ({err_p3:.1f}%)"),
        ('P4 Est.', f"{est_params[2]:.3f} ({err_p4:.1f}%)"),
    ]
    
    W1, W2 = 12, 18
    SEP_V = "│"
    SEP_MID = " ║ "
    LINE_L = "─" * (W1 + 3 + W2)
    LINE_R = "─" * (W1 + 3 + W2)
    
    def fmt_cell(label, val):
        return f"{label:>{W1}} {SEP_V} {val:<{W2}}"

    header = f"{'INPUTS':^{len(LINE_L)}}{SEP_MID}{'OUTPUTS':^{len(LINE_R)}}"
    subhead = f"{fmt_cell('Param.', 'Valor')}{SEP_MID}{fmt_cell('Métrica', 'Valor')}"
    separator = f"{LINE_L}{SEP_MID}{LINE_R}"
    
    body_lines = []
    max_rows = max(len(dados_input), len(dados_output))
    
    for i in range(max_rows):
        left_txt  = fmt_cell(*dados_input[i]) if i < len(dados_input) else " " * (W1 + 3 + W2)
        right_txt = fmt_cell(*dados_output[i]) if i < len(dados_output) else " " * (W1 + 3 + W2)
        body_lines.append(f"{left_txt}{SEP_MID}{right_txt}")
        
    texto_final = f"{header}\n{subhead}\n{separator}\n" + "\n".join(body_lines)
    
    ax_text.text(0.5, 0.5, texto_final, 
                 fontsize=10, 
                 fontname='DejaVu Sans Mono', 
                 verticalalignment='center',
                 horizontalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9))
    
    plt.subplots_adjust(bottom=0.06, top=0.97, left=0.08, right=0.97)
    
    output_folder = "sir_results_params"
    os.makedirs(output_folder, exist_ok=True)
    filename = f"SIR_Params_N={N_PARTICLES}_RWu={SIGMA_RW_U}_R={R_SIGMA:.3f}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    print(f"Simulation Complete. Saved: {filepath}")
    plt.close(fig)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Prior Means (Guesses close to truth)
    # Real values are [0.25, 15.0, 0.3]
    w = w_real_noise_data
    prior_means = [0.25*1.01, 15.0*0.99, 0.3*1.01]
    
    # Standard deviation of priors as a MULTIPLIER of the mean
    # e.g., 0.1 means sigma = 10% of mean
    sigma_prior_mult = [0.025, 0.025, 0.025] 
    
    # Random Walk Step size as a MULTIPLIER of the mean
    # e.g., 0.001 means step_sigma = 0.1% of mean
    rw_mult_params = 0.001

    # Batch List: (N_PARTICLES, SIGMA_RW_U, R_SIGMA, PRIOR_MEANS, SIGMA_PRIOR_MULT, RW_MULT_PARAMS)
    # simulations = [
    #     (100, w*1, w*1, prior_means, sigma_prior_mult, rw_mult_params),
    #     (100, w*10, w*6, prior_means, sigma_prior_mult, rw_mult_params),
    #     # (100, w*10, w*6, prior_means, sigma_prior_mult, rw_mult_params),
    # ]
    
    # print(f"Starting Batch of {len(simulations)} simulations...")
    # for i, params in enumerate(simulations):
    #     print(f"\n--- Simulation {i+1}/{len(simulations)} ---")
    #     main(*params)
    
    prior_means = [0.25*1.05, 15.0*0.95, 0.3*1.05]
    sigma_prior_mult = [0.05, 0.05, 0.05] 
    simulations = [
    # (1000, w*10, w*6, prior_means, sigma_prior_mult, rw_mult_params),
    (5000, w*10, w*5, prior_means, sigma_prior_mult, rw_mult_params),
    ]
    
    print(f"Starting Batch of {len(simulations)} simulations...")
    for i, params in enumerate(simulations):
        print(f"\n--- Simulation {i+1}/{len(simulations)} ---")
        main(*params)