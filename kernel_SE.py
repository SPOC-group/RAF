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
def state_evolution_inplace(compute_conjugates, mu_1, mu_star, alpha, epsilon, lam, old_state, new_state, nodes, weights):
    # Unpack old state
    ms, qs, Vs = old_state[0], old_state[1], old_state[2]
    
    # Update new state
    m = mu_1 * ms
    q = mu_1**2 * qs
    if lam == 0:
        V = mu_1**2 * Vs + mu_star**2
        m_hat, q_hat, V_hat = compute_conjugates(alpha, epsilon, m, q, V, nodes, weights)
        new_state[3] = mu_1 * m_hat # ms_hat_new
        new_state[4] = mu_1**2 * q_hat # qs_hat_new
        new_state[5] = mu_1**2 * V_hat # Vs_hat_new
        den = 1 + new_state[5]
        new_state[0] = new_state[3] / den # ms_new
        new_state[1] = (new_state[3]**2 + new_state[4]) / den**2 # qs_new
        new_state[2] = 1.0 / den # Vs_new
    else:
        V = mu_1**2 * Vs + mu_star**2 / lam
        m_hat, q_hat, V_hat = compute_conjugates(alpha, epsilon, m, q, V, nodes, weights)
        new_state[3] = mu_1 * m_hat # ms_hat_new
        new_state[4] = mu_1**2 * q_hat # qs_hat_new
        new_state[5] = mu_1**2 * V_hat # Vs_hat_new
        den = lam + new_state[5]
        new_state[0] = new_state[3] / den # ms_new
        new_state[1] = (new_state[3]**2 + new_state[4]) / den**2 # qs_new
        new_state[2] = 1.0 / den # Vs_new


def kernel_SE_solution(loss, mu_1, mu_star, alpha, epsilon, lam, ms0 = 0.1, qs0 = 0.8, Vs0 = 3, tol = 1e-5, damping = False, delta = 0.1, max_iter = 5000, verbose = False):

    # Choose loss once
    if loss == "square":
        if lam == 0:
            compute_conjugates = lossdependent_SE.compute_square_conjugates_kernel_zerolambda
        else:
            compute_conjugates = lossdependent_SE.compute_square_conjugates
    elif loss == "hinge":
        if lam == 0:
            compute_conjugates = lossdependent_SE.compute_hinge_conjugates_kernel_zerolambda
        else:
            compute_conjugates = lossdependent_SE.compute_hinge_conjugates
    else:
        raise ValueError("Unknown loss")

    # Initialize
    m0 = mu_1 * ms0
    q0 = mu_1**2 * qs0
    if lam == 0:
        V0 = mu_1**2 * Vs0 + mu_star**2
        m_hat0, q_hat0, V_hat0 = compute_conjugates(alpha, epsilon, m0, q0, V0, legnodes, legweights)
    else:
        V0 = mu_1**2 * Vs0 + mu_star**2 / lam
        m_hat0, q_hat0, V_hat0 = compute_conjugates(alpha, epsilon, m0, q0, V0, legnodes, legweights)
    ms_hat0 = mu_1 * m_hat0
    qs_hat0 = mu_1**2 * q_hat0
    Vs_hat0 = mu_1**2 * V_hat0
    overlaps = np.array([ms0, qs0, Vs0, ms_hat0, qs_hat0, Vs_hat0])
    new_overlaps = np.empty(6)

    # Fixed point iteration scheme
    for idx_iter in range(max_iter):
        state_evolution_inplace(compute_conjugates, mu_1, mu_star, alpha, epsilon, lam, overlaps, new_overlaps, legnodes, legweights)
        err = 0.0
        for i in range(6):
            d = new_overlaps[i] - overlaps[i]
            err += d * d
        err = np.sqrt(err)
        if err < tol:
            if verbose:
                print(f"Fixed point iteration converged (mu_1={mu_1}, mu_star={mu_star}, alpha={alpha}, epsilon={epsilon}, lambda={lam}) in {idx_iter+1} iters; err={err:.2e}")
            return overlaps, True
        if damping:
            new_overlaps[:] = (1 - delta) * overlaps + delta * new_overlaps
            overlaps, new_overlaps = new_overlaps, overlaps
        else:
            overlaps, new_overlaps = new_overlaps, overlaps
    
    # If convergence not achieved within max_iter
    if verbose:
        print(f"Fixed point iteration did not converge (mu_1={mu_1}, mu_star={mu_star}, alpha={alpha}, epsilon={epsilon}, lambda={lam}) in {max_iter} iters; err={err:.2e}")
    return overlaps, False