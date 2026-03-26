import numpy as np
from scipy.integrate import quad
from scipy.special import erf


inf_integral = -5
sup_integral = 5


# State equation for the overlap

def F_qb(qb_hat):
    return qb_hat / (1 + qb_hat)


# State equation for the conjugate variable

def F_qb_hat(alpha, epsilon, qb):
    return 4 * (1 - epsilon)**2 * alpha / (2 * np.pi) / np.sqrt(2 * np.pi) / (1 - qb) * quad(lambda t: np.exp(- (1 + qb) * t**2 / (2 * (1 - qb))) / (1 + (1 - epsilon) * erf(np.sqrt(qb) * t / np.sqrt(2 * (1 - qb)))), inf_integral, sup_integral)[0]


# Solving the SP equations

def state_evolution(overlaps, alpha, epsilon):
    # Unpack current overlap and conjugate variable
    qb, _ = overlaps

    # Update conjugate variable
    qb_hat_new = F_qb_hat(alpha, epsilon, qb)

    # Update overlap
    qb_new = F_qb(qb_hat_new)

    # Collect updated state
    new_overlaps = np.array([qb_new, qb_hat_new])

    return new_overlaps


def bo_SP_solution(alpha, epsilon, qb0 = 0.8, tol = 1e-5, damping = False, delta = 0.1, max_iter = 5000, verbose = False):
    # Initialize conjugate variable (qb_hat) from initial overlap
    qb_hat0 = F_qb_hat(alpha, epsilon, qb0)

    # Collect all state variables into a single vector
    overlaps = np.array([qb0, qb_hat0])

    for i in range(max_iter):
        # Update overlaps using the state evolution equations
        new_overlaps = state_evolution(overlaps, alpha, epsilon)

        # Compute error (distance between successive states)
        err = np.linalg.norm(new_overlaps - overlaps)

        # Check convergence
        if err < tol:
            if verbose == True:
                print(f"[SE] converged (alpha={alpha}, epsilon={epsilon}) in {i+1} iters; err={err:.2e}")
            return overlaps, True
        
        # Apply damping if requested
        if damping == True:
            overlaps = delta * overlaps + (1 - delta) * new_overlaps
        else:
            overlaps = new_overlaps
    
    # If convergence not achieved within max_iter
    if verbose == True:
        print(f"[SE] did not converge (alpha={alpha}, epsilon={epsilon}) in {max_iter} iters; err={err:.2e}")
    return overlaps, False


# Bayes-optimal generalization error

def bo_generalization_error(qb):
    return np.where(qb == 0, 1/2, np.arccos(np.sqrt(qb)) / np.pi)