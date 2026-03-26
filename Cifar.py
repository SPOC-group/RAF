import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklear.svm import SVC

RNG = np.random.default_rng(0)


CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# Parameters
alpha = ...
eps = ...
n_test = ...
n_repeats = ...

class_pos = 3    
class_neg = 5   
class_mem = 1   

lambdas = ...
etas = ...

n_scale_pairs = ...


def flatten_images(X):
    return X.reshape(X.shape[0], -1).astype(np.float64)


def sample_without_replacement(indices, n, rng):
    indices = np.asarray(indices)
    if len(indices) < n:
        raise ValueError(f"Requested {n} samples, but only {len(indices)} available.")
    return rng.choice(indices, size=n, replace=False)


def pairwise_sq_dists(XA, XB):
    XA = np.asarray(XA, dtype=np.float64)
    XB = np.asarray(XB, dtype=np.float64)

    normA = np.sum(XA * XA, axis=1, keepdims=True)
    normB = np.sum(XB * XB, axis=1, keepdims=True).T
    D2 = normA + normB - 2.0 * (XA @ XB.T)
    return np.maximum(D2, 0.0)


def rbf_kernel_from_sq_dists(D2, eta_eff):
    return np.exp(-eta_eff * D2, dtype=np.float64)


def krr_predict_from_eigendecomp(K_test, U, s, y_train, lam):
    Uy = U.T @ y_train
    coeff = Uy / (s + lam)
    alpha_dual = U @ coeff

    f_train = U @ (s * coeff)
    f_test = K_test @ alpha_dual
    return f_train, f_test


def svm_predict_precomputed(K_train, K_test, y_train, lam, tol=1e-3, cache_size_mb=1000):
    C = 1.0 / lam

    clf = SVC(
        C=C,
        kernel="precomputed",
        shrinking=True,
        tol=tol,
        cache_size=cache_size_mb,
        max_iter=-1,
    )
    clf.fit(K_train, y_train)

    y_pred_train = clf.predict(K_train)
    y_pred_test = clf.predict(K_test)
    return y_pred_train, y_pred_test


def sign_predictions(scores):
    return np.where(scores >= 0.0, 1, -1)


def classification_error(y_true, y_pred):
    return 1.0 - np.mean(y_true == y_pred)


def fit_zscore(X_train_raw):
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_zscore(X, mean, std):
    return (X - mean) / std


def estimate_median_sqdist(X, rng, n_pairs=20000):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]

    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)

    same = (i == j)
    while np.any(same):
        j[same] = rng.integers(0, n, size=np.sum(same))
        same = (i == j)

    diff = X[i] - X[j]
    d2 = np.sum(diff * diff, axis=1)
    d2 = np.maximum(d2, 0.0)

    median_d2 = np.median(d2)
    return max(median_d2, 1e-12)


def summarize_kernel(K, name="K"):
    n = K.shape[0]
    diag_mean = np.mean(np.diag(K))
    offdiag_sum = K.sum() - np.trace(K)
    offdiag_mean = offdiag_sum / (n * (n - 1)) if n > 1 else 0.0
    print(f"{name}: mean(diag)={diag_mean:.6e}, mean(offdiag)={offdiag_mean:.6e}")


# Load CIFAR10
(X_train_full, y_train_full), (X_test_full, y_test_full) = cifar10.load_data()
y_train_full = y_train_full.reshape(-1)
y_test_full = y_test_full.reshape(-1)

X_train_full = flatten_images(X_train_full)
X_test_full = flatten_images(X_test_full)

d = X_train_full.shape[1]
n_total = int(round(alpha * d))
n_struct = int(round((1.0 - eps) * n_total))
n_mem = n_total - n_struct

if n_struct % 2 != 0:
    raise ValueError("For balanced structured classes, n_struct must be even.")
if n_test % 2 != 0:
    raise ValueError("For balanced structured test set, n_test must be even.")

n_struct_pos = n_struct // 2
n_struct_neg = n_struct // 2
n_test_pos = n_test // 2
n_test_neg = n_test // 2


idx_train_pos = np.where(y_train_full == class_pos)[0]
idx_train_neg = np.where(y_train_full == class_neg)[0]
idx_train_mem = np.where(y_train_full == class_mem)[0]

idx_test_pos = np.where(y_test_full == class_pos)[0]
idx_test_neg = np.where(y_test_full == class_neg)[0]


results = {
    eta: {
        "E_mem_repeats": [],
        "E_gen_repeats": [],
        "E_mem_mean": None,
        "E_gen_mean": None,
        "E_mem_std": None,
        "E_gen_std": None,
        "eta_effs": [],
    }
    for eta in etas
}


for rep in range(n_repeats):

    sel_train_pos = sample_without_replacement(idx_train_pos, n_struct_pos, RNG)
    sel_train_neg = sample_without_replacement(idx_train_neg, n_struct_neg, RNG)
    sel_train_mem = sample_without_replacement(idx_train_mem, n_mem, RNG)

    sel_test_pos = sample_without_replacement(idx_test_pos, n_test_pos, RNG)
    sel_test_neg = sample_without_replacement(idx_test_neg, n_test_neg, RNG)


    X_train_raw = np.vstack([
        X_train_full[sel_train_pos],
        X_train_full[sel_train_neg],
        X_train_full[sel_train_mem],
    ]).astype(np.float64)

    y_train = np.concatenate([
        np.ones(n_struct_pos, dtype=np.int64),
        -np.ones(n_struct_neg, dtype=np.int64),
        RNG.choice([-1, 1], size=n_mem, replace=True).astype(np.int64),
    ]).astype(np.float64)

    mem_mask = np.zeros(n_total, dtype=bool)
    mem_mask[n_struct:] = True


    X_test_raw = np.vstack([
        X_test_full[sel_test_pos],
        X_test_full[sel_test_neg],
    ]).astype(np.float64)

    y_test = np.concatenate([
        np.ones(n_test_pos, dtype=np.int64),
        -np.ones(n_test_neg, dtype=np.int64),
    ]).astype(np.float64)

    mean, std = fit_zscore(X_train_raw)
    X_train = apply_zscore(X_train_raw, mean, std).astype(np.float64)
    X_test = apply_zscore(X_test_raw, mean, std).astype(np.float64)


    median_d2 = estimate_median_sqdist(X_train, RNG, n_pairs=n_scale_pairs)

    D2_train = pairwise_sq_dists(X_train, X_train)
    D2_test = pairwise_sq_dists(X_test, X_train)

    # Run etas
    for eta in etas:
        eta_eff = eta / median_d2

        K_train = rbf_kernel_from_sq_dists(D2_train, eta_eff)
        K_test = rbf_kernel_from_sq_dists(D2_test, eta_eff)

        summarize_kernel(K_train, name="K_train")

        s, U = np.linalg.eigh(K_train)
        s = np.clip(s, 0.0, None)

        mem_errs = []
        gen_errs = []

        for lam in lambdas:
            f_train, f_test = krr_predict_from_eigendecomp(K_test, U, s, y_train, lam)

            y_pred_train = sign_predictions(f_train)
            y_pred_test = sign_predictions(f_test)

            mem_err = classification_error(y_train[mem_mask], y_pred_train[mem_mask])
            gen_err = classification_error(y_test, y_pred_test)

            mem_errs.append(mem_err)
            gen_errs.append(gen_err)


        results[eta]["E_mem_repeats"].append(np.asarray(mem_errs, dtype=np.float64))
        results[eta]["E_gen_repeats"].append(np.asarray(gen_errs, dtype=np.float64))
        results[eta]["eta_effs"].append(eta_eff)


for eta in etas:
    E_mem_reps = np.stack(results[eta]["E_mem_repeats"], axis=0)
    E_gen_reps = np.stack(results[eta]["E_gen_repeats"], axis=0)

    results[eta]["E_mem_mean"] = E_mem_reps.mean(axis=0)
    results[eta]["E_gen_mean"] = E_gen_reps.mean(axis=0)
    results[eta]["E_mem_std"] = E_mem_reps.std(axis=0)
    results[eta]["E_gen_std"] = E_gen_reps.std(axis=0)


