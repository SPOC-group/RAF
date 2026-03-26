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
def state_evolution_inplace(compute_conjugates, alpha, epsilon, lam, below_alpha_c, old_state, new_state, nodes, weights):
    # Unpack old state
    m, q, V = old_state[0], old_state[1], old_state[2]

    # Update new state
    new_state[3], new_state[4], new_state[5] = compute_conjugates(alpha, epsilon, m, q, V, nodes, weights)
    if lam == 0 and below_alpha_c:
        den = 1 + new_state[5]
    else:
        den = lam + new_state[5]
    new_state[0] = new_state[3] / den # m_new
    new_state[1] = (new_state[3]**2 + new_state[4]) / den**2 # q_new
    new_state[2] = 1.0 / den # V_new


def perceptron_SE_solution(loss, alpha, epsilon, lam, below_alpha_c = False, m0 = 0.1, q0 = 0.8, V0 = 3, tol = 1e-5, damping = False, delta = 0.1, max_iter = 10000, verbose = False):

    # Choose loss once
    if loss == "square":
        if lam == 0 and alpha <= 1:
            below_alpha_c = True
            compute_conjugates = lossdependent_SE.compute_square_conjugates_kernel_zerolambda
        else:
            compute_conjugates = lossdependent_SE.compute_square_conjugates
    elif loss == "hinge":
        if lam == 0 and below_alpha_c:
            compute_conjugates = lossdependent_SE.compute_hinge_conjugates_kernel_zerolambda
        else:
            compute_conjugates = lossdependent_SE.compute_hinge_conjugates
    else:
        raise ValueError("Unknown loss")

    # Initialize
    m_hat0, q_hat0, V_hat0 = compute_conjugates(alpha, epsilon, m0, q0, V0, legnodes, legweights)
    overlaps = np.array([m0, q0, V0, m_hat0, q_hat0, V_hat0])
    new_overlaps = np.empty(6)

    # Fixed point iteration scheme
    for idx_iter in range(max_iter):
        state_evolution_inplace(compute_conjugates, alpha, epsilon, lam, below_alpha_c, overlaps, new_overlaps, legnodes, legweights)
        err = 0.0
        for i in range(6):
            d = new_overlaps[i] - overlaps[i]
            err += d * d
        err = np.sqrt(err)
        if err < tol:
            if verbose:
                print(f"Fixed point iteration converged (alpha={alpha}, epsilon={epsilon}, lambda={lam}) in {idx_iter+1} iters; err={err:.2e}")
            return overlaps, True
        if damping:
            new_overlaps[:] = (1 - delta) * overlaps + delta * new_overlaps
            overlaps, new_overlaps = new_overlaps, overlaps
        else:
            overlaps, new_overlaps = new_overlaps, overlaps
    
    # If convergence not achieved within max_iter
    if verbose:
        print(f"Fixed point iteration did not converge (alpha={alpha}, epsilon={epsilon}, lambda={lam}) in {max_iter} iters; err={err:.2e}")
    return overlaps, False