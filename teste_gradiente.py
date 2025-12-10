import numpy as np
from problema_direto_gc import problema_direto, problema_adjunto, problema_sensibilidade, u_of_t, N, sigma1, sigma2

def verificar_consistencia():
    print("--- INICIANDO TESTE DE CONSISTÊNCIA ---")
    
    # 1. Configurar um estado inicial arbitrário
    u = np.zeros(N) # u base
    # Uma perturbação aleatória delta_u
    delta_u = np.random.normal(0, 1.0, N) 
    
    # 2. Rodar o problema direto
    T1, T2 = problema_direto(u, T1=1.0, T2=1.0)
    
    # 3. Gerar um resíduo artificial (fingindo que Y são os dados)
    # Vamos criar um Y falso tal que o resíduo seja não-nulo
    Y1_falso = T1 + 0.1 
    Y2_falso = T2 + 0.1
    
    # 4. Calcular o Gradiente via ADJUNTO
    # Importante: O gradiente J'(u) é igual a -Lambda (ou +Lambda dependendo da definição)
    # Aqui calculamos o termo de fonte do adjunto baseado no Y_falso
    L1, L2 = problema_adjunto(0, 0, T1, T2, Y1_falso, Y2_falso)
    grad_adjunto = -L1 # Sua definição de gradiente
    
    # 5. Calcular a variação de Temperatura via SENSIBILIDADE
    DT1, DT2 = problema_sensibilidade(delta_u, T1, T2)
    
    # --- A HORA DA VERDADE (Identidade de Lagrange) ---
    # O produto escalar <Gradiente, Delta_u> TEM QUE SER IGUAL A <Resíduo, Delta_T>
    
    # Lado Esquerdo: O que o gradiente diz que vai mudar no custo
    lhs = np.sum(grad_adjunto * delta_u)
    
    # Lado Direito: O que a sensibilidade diz que mudou na temperatura ponderada pelo resíduo
    # Resíduo ponderado = (T - Y) / sigma^2
    res1 = (T1 - Y1_falso) / (sigma1**2)
    res2 = (T2 - Y2_falso) / (sigma2**2)
    
    # Integral no tempo (soma)
    rhs = np.sum(res1 * DT1) + np.sum(res2 * DT2)
    
    print(f"Produto Interno (Via Gradiente): {lhs:.8f}")
    print(f"Produto Interno (Via Sensib.):  {rhs:.8f}")
    print(f"Diferença Relativa: {abs((lhs-rhs)/lhs)*100:.4f}%")
    
    if abs((lhs-rhs)/lhs) < 1e-2: # Erro menor que 1%
        print(">> SUCESSO: O Gradiente (Adjunto) é consistente com a Sensibilidade!")
    else:
        print(">> FALHA CRÍTICA: O Gradiente e a Sensibilidade não batem.")
        print("   Isso significa que dL2 ou dDT2 tem erro de sinal ou termo faltando.")

if __name__ == "__main__":
    verificar_consistencia()