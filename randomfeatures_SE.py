import numpy as np
from numba import njit
import lossdependent_SE
from numpy.polynomial.legendre import leggauss


# ------------------------
# One-time Legendre setup
# ------------------------
legn_points = 200
legnodes, legweights = leggauss(legn_points)


# --- SE solution ---

@njit
def state_evolution_inplace(compute_conjugates, mu_1, mu_star, alpha, kappa, epsilon, lam, old_state, new_state, nodes, weights):
    # Unpack old state
    ms, qs, Vs, qw, Vw = old_state[0], old_state[1], old_state[2], old_state[3], old_state[4]
    
    # Update new state
    m = mu_1 * ms
    q = mu_1**2 * qs + mu_star**2 * qw
    V = mu_1**2 * Vs + mu_star**2 * Vw
    alpha_prime = alpha / kappa
    m_hat, q_hat, V_hat = compute_conjugates(alpha_prime, epsilon, m, q, V, nodes, weights)
    new_state[5] = mu_1 * kappa * m_hat # ms_hat_new
    new_state[6] = mu_1**2 * kappa * q_hat # qs_hat_new
    new_state[7] = mu_1**2 * kappa * V_hat # Vs_hat_new
    new_state[8] = mu_star**2 * q_hat # qw_hat_new
    new_state[9] = mu_star**2 * V_hat # Vw_hat_new
    z = (lam + new_state[9]) / new_state[7]
    gamma = 1 / kappa
    Delta = np.sqrt((1 + gamma + z)**2 - 4 * gamma)
    new_state[0] = new_state[5] * (z + 1 + gamma - Delta) / (2 * gamma * new_state[7]) # ms_new
    new_state[1] = (new_state[5]**2 + new_state[6]) * ((2 * z + gamma + 1) * Delta - 2 * z**2 - 3 * (gamma + 1) * z - (gamma - 1)**2) / (2 * gamma * new_state[7]**2 * Delta) - new_state[8] * (z * Delta - z**2 - (gamma + 1) * z) / (2 * gamma * new_state[7] * (lam + new_state[9]) * Delta) # qs_new
    new_state[2] = (z + 1 + gamma - Delta) / (2 * gamma * new_state[7]) # Vs_new
    new_state[3] = new_state[8] * ((1 - gamma) * Delta + (gamma + 1) * z + (gamma - 1)**2) / (2 * (lam + new_state[9])**2 * Delta) - (new_state[5]**2 + new_state[6]) * (z * Delta - z**2 - (gamma + 1) * z) / (2 * new_state[7] * (lam + new_state[9]) * Delta) # qw_new
    new_state[4] = (1 - gamma - z + Delta) / (2 * (lam + new_state[9])) # Vw_new


def randomfeatures_SE_solution(loss, mu_1, mu_star, alpha, kappa, epsilon, lam, ms0 = 0.1, qs0 = 0.8, Vs0 = 3, qw0 = 0.8, Vw0 = 3, tol = 1e-5, damping = False, delta = 0.1, max_iter = 5000, verbose = False):

    # Choose loss once
    if loss == "square":
        compute_conjugates = lossdependent_SE.compute_square_conjugates
    elif loss == "hinge":
        compute_conjugates = lossdependent_SE.compute_hinge_conjugates
    else:
        raise ValueError("Unknown loss")

    # Initialize
    m0 = mu_1 * ms0
    q0 = mu_1**2 * qs0 + mu_star**2 * qw0
    V0 = mu_1**2 * Vs0 + mu_star**2 * Vw0
    alpha_prime = alpha / kappa
    m_hat0, q_hat0, V_hat0 = compute_conjugates(alpha_prime, epsilon, m0, q0, V0, legnodes, legweights)
    ms_hat0 = mu_1 * kappa * m_hat0
    qs_hat0 = mu_1**2 * kappa * q_hat0
    Vs_hat0 = mu_1**2 * kappa * V_hat0
    qw_hat0 = mu_star**2 * q_hat0
    Vw_hat0 = mu_star**2 * V_hat0
    overlaps = np.array([ms0, qs0, Vs0, qw0, Vw0, ms_hat0, qs_hat0, Vs_hat0, qw_hat0, Vw_hat0])
    new_overlaps = np.empty(10)

    # Fixed point iteration scheme
    for idx_iter in range(max_iter):
        state_evolution_inplace(compute_conjugates, mu_1, mu_star, alpha, kappa, epsilon, lam, overlaps, new_overlaps, legnodes, legweights)
        err = 0.0
        for i in range(10):
            d = new_overlaps[i] - overlaps[i]
            err += d * d
        err = np.sqrt(err)
        if err < tol:
            if verbose:
                print(f"Fixed point iteration converged (alpha={alpha}, kappa={kappa}, epsilon={epsilon}, lambda={lam}) in {idx_iter+1} iters; err={err:.2e}")
            return overlaps, True
        if damping:
            new_overlaps[:] = (1 - delta) * overlaps + delta * new_overlaps
            overlaps, new_overlaps = new_overlaps, overlaps
        else:
            overlaps, new_overlaps = new_overlaps, overlaps
    
    # If convergence not achieved within max_iter
    if verbose:
        print(f"Fixed point iteration did not converge (alpha={alpha}, kappa={kappa}, epsilon={epsilon}, lambda={lam}) in {max_iter} iters; err={err:.2e}")
    return overlaps, False