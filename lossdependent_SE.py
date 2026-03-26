from numba import njit
from math import sqrt, erf, exp, atan, pi
from numpy.polynomial.hermite import hermgauss
import numpy as np


@njit
def safe_erf(x):
    if x > 6.0:
        return 1.0
    elif x < -6.0:
        return -1.0
    else:
        return erf(x)

@njit
def safe_exp(x):
    if x < -700.0:  # underflow threshold
        return 0.0
    else:
        return exp(x)


# --- Square loss ---

@njit
def compute_square_conjugates(alpha, epsilon, m, q, V, nodes, weights):
    den = 1 + V
    m_hat = alpha * (1 - epsilon) * sqrt(2 / pi) / den
    q_hat = alpha * (1 + q - 2 * sqrt(2 / pi) * (1 - epsilon) * m) / den**2
    V_hat = alpha / den
    
    return m_hat, q_hat, V_hat

@njit
def compute_square_conjugates_kernel_zerolambda(alpha, epsilon, m, q, V, nodes, weights):
    m_hat = alpha * (1 - epsilon) * sqrt(2 / pi) / V
    q_hat = alpha * (1 + q - 2 * sqrt(2 / pi) * (1 - epsilon) * m) / V**2
    V_hat = alpha / V
    
    return m_hat, q_hat, V_hat


# --- Hinge loss ---

@njit
def compute_hinge_conjugates(alpha, epsilon, m, q, V, nodes, weights):
    # Utils
    eta = m**2 / q
    one_minus_eta = 1.0 - eta
    sqrt2q = sqrt(2 * q)
    sqrt_2pi = sqrt(2.0 * pi)
    sqrt_2q_one_minus_eta = sqrt(2.0 * q * one_minus_eta)
    
    # m_hat
    m_hat_term1 = 1.0 + (safe_erf(1.0 / sqrt_2q_one_minus_eta) + safe_erf((1.0 - V) / sqrt_2q_one_minus_eta) * (V - 1.0)) / V
    m_hat_term2 = sqrt_2q_one_minus_eta / (V * sqrt(pi)) * (safe_exp(-1.0 / (2.0 * q * one_minus_eta)) - safe_exp(-(1.0 - V)**2 / (2.0 * q * one_minus_eta)))
    m_hat = alpha * (1.0 - epsilon) / sqrt_2pi * (m_hat_term1 + m_hat_term2)
    
    # q_hat
    q_hat_term1 = alpha * (0.5 + (0.5 * safe_erf(1 / sqrt2q) * (1 + q) + 0.5 * safe_erf((1 - V) / sqrt2q) * (V**2 - 1 - q) + sqrt(q / (2 * pi)) * (safe_exp(-1 / (2 * q)) - safe_exp(-(1 - V)**2 / (2 * q)) * (1 + V))) / V**2)
    a1 = -(1 - V) / sqrt(q)
    b1 = 0.0
    integral1 = 0.0
    for i in range(len(nodes)):
        t = 0.5 * (b1 - a1) * nodes[i] + 0.5 * (b1 + a1)
        f = safe_exp(-t**2 / 2) / sqrt(2 * pi) * safe_erf(sqrt(eta) * t / sqrt(2 * one_minus_eta))
        integral1 += weights[i] * f
    integral1 *= 0.5 * (b1 - a1)
    a2 = (1 - V) / sqrt(q)
    b2 = 1 / sqrt(q)
    integral2 = 0.0
    for i in range(len(nodes)):
        t = 0.5 * (b2 - a2) * nodes[i] + 0.5 * (b2 + a2)
        f = safe_exp(-t**2 / 2) / sqrt(2 * pi) * safe_erf(sqrt(eta) * t / sqrt(2 * one_minus_eta)) * (1 - sqrt(q) * t)**2
        integral2 += weights[i] * f
    integral2 *= 0.5 * (b2 - a2)
    arcterm = -atan(sqrt(eta) / sqrt(one_minus_eta)) / pi
    q_hat_term2 = alpha * (1 - epsilon) * (arcterm - integral1 + integral2 / V**2)
    q_hat = q_hat_term1 + q_hat_term2
    
    # V_hat
    V_hat_term1 = alpha / (2 * V) * (safe_erf(1 / sqrt2q) - safe_erf((1 - V) / sqrt2q))
    a = (1 - V) / sqrt(q)
    b = 1 / sqrt(q)
    integral = 0.0
    for i in range(len(nodes)):
        t = 0.5 * (b - a) * nodes[i] + 0.5 * (b + a)
        f = safe_exp(-t**2 / 2) / sqrt(2 * pi) * safe_erf(sqrt(eta) * t / sqrt(2 * one_minus_eta))
        integral += weights[i] * f
    integral *= 0.5 * (b - a)
    V_hat_term2 = alpha * (1 - epsilon) / V * integral
    V_hat = V_hat_term1 + V_hat_term2
    
    return m_hat, q_hat, V_hat

@njit
def compute_hinge_conjugates_kernel_zerolambda(alpha, epsilon, m, q, V, nodes, weights):
    # Utils
    eta = m**2 / q
    one_minus_eta = 1.0 - eta
    sqrt2q = sqrt(2 * q)
    sqrt_2pi = sqrt(2.0 * pi)
    sqrt_2q_one_minus_eta = sqrt(2.0 * q * one_minus_eta)

    # m_hat
    m_hat_term1 = sqrt_2pi * (erf(1.0 / sqrt_2q_one_minus_eta) + 1)
    m_hat_term2 = 2 * sqrt(q) * sqrt(one_minus_eta) * exp(-1 / (2.0 * q * one_minus_eta))
    m_hat = alpha * (1 - epsilon) / (2 * pi * V) * (m_hat_term1 + m_hat_term2)
        
    # q_hat
    q_hat_term1 = (1 + q) * (1 + erf(1 / sqrt2q)) / 2 + sqrt(q / (2 * pi)) * exp(-1 / (2 * q))
    a1 = -10
    b1 = 1 / sqrt(q)
    integral1 = 0.0
    for i in range(len(nodes)):
        t = 0.5 * (b1 - a1) * nodes[i] + 0.5 * (b1 + a1)
        f = exp(-t**2 / 2) / sqrt_2pi * erf(sqrt(eta) * t / sqrt(2 * one_minus_eta)) * (1 - sqrt(q) * t)**2
        integral1 += weights[i] * f
    integral1 *= 0.5 * (b1 - a1)
    q_hat_term2 = (1 - epsilon) * integral1
    q_hat = alpha / V**2 * (q_hat_term1 + q_hat_term2)
        
    # V_hat
    V_hat_term1 = (1 + erf(1 / sqrt2q)) / 2
    a2 = -10
    b2 = 1 / sqrt(q)
    integral2 = 0.0
    for i in range(len(nodes)):
        t = 0.5 * (b2 - a2) * nodes[i] + 0.5 * (b2 + a2)
        f = exp(-t**2 / 2) / sqrt_2pi * erf(sqrt(eta) * t / sqrt(2 * one_minus_eta))
        integral2 += weights[i] * f
    integral2 *= 0.5 * (b2 - a2)
    V_hat_term2 = (1 - epsilon) * integral2
    V_hat = alpha / V * (V_hat_term1 + V_hat_term2)

    return m_hat, q_hat, V_hat