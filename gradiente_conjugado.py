import numpy as np
from problema_direto import problema_direto, pseudo_dados, problema_adjunto


def objective_functional(T1, T2, Y1, Y2):
    result = np.sum((T1 - Y1)**2) + np.sum((T2 - Y2)**2)

    return result

def gradiente_conjudado(eps, P0=[1.0, 0.2, 10.0, 0.2], version='fs'):
    i = 0
    P0 = np.array([P0])
    i_max = 100
    grad_old = 1
    dk_old = 0
    while (i < i_max):
        # Step 1
        t, T1, T2 = problema_direto(T1=T1, T2=T2, *P0)
        t, Y1, Y2 = pseudo_dados()

        # Step 2
        S_g = objective_functional(T1, T2, Y1, Y2)
        
        if S_g < eps:
            break

        # Step 3: solve adjoint problem
        t, L1, L2 = problema_adjunto(T1, T2, Y1, Y2)

        # Step 4: compute the gradient of S
        grad = -L1

        # Step 5: compute yk, psik and dk. Use all four formulations
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
        grad_old = grad
        # Step 6: set Deltag equal to dk and solve the sensitivity problem
        delta_g = d_k

    return