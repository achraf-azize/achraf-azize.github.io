import numpy as np
from rlberry.utils.jit_setup import numba_jit
from rlberry.utils.metrics import metric_lp


@numba_jit
def metric_lp(x, y, p, scaling):
    """
    Returns the p-norm:  || (x-y)/scaling||_p

    Parameters
    ----------
    x : numpy.ndarray
        1d array
    y : numpy.ndarray
        1d array
    p : int
        norm parameter
    scaling : numpy.ndarray
        1d array
    """
    assert p >= 1
    assert x.ndim == 1
    assert y.ndim == 1
    assert scaling.ndim == 1

    d = len(x)
    diff = np.abs((x - y) / scaling)
    # p = infinity
    if p == np.inf:
        return diff.max()
    # p < infinity
    tmp = 0
    for ii in range(d):
        tmp += np.power(diff[ii], p)
    return np.power(tmp, 1.0 / p)


@numba_jit
def map_to_representative(state,
                          lp_metric,
                          representative_states,
                          n_representatives,
                          min_dist,
                          scaling,
                          accept_new_repr):
    """Map state to representative state. """
    dist_to_closest = np.inf
    argmin = -1
    for ii in range(n_representatives):
        dist = metric_lp(state, representative_states[ii, :],
                         lp_metric,
                         scaling)
        if dist < dist_to_closest:
            dist_to_closest = dist
            argmin = ii

    max_representatives = representative_states.shape[0]
    if (dist_to_closest > min_dist) \
            and (n_representatives < max_representatives) \
            and accept_new_repr:
        new_index = n_representatives
        representative_states[new_index, :] = state
        return new_index
    return argmin


@numba_jit
def update_value_and_get_action(
    state,
    hh,
    V,
    R_hat,
    P_hat,
    B_sa,
    gamma,
    v_max):
    """
    state : int
    hh : int
    V : np.ndarray
        shape (H, S)
    R_hat : np.ndarray
        shape (S, A)
    P_hat : np.ndarray
        shape (S, A, S)
    B_sa : np.ndarray
        shape (S, A)
    gamma : double
    v_max : np.ndarray
        shape (H,)
    """
    H = V.shape[0]
    S, A = R_hat.shape[-2:]
    best_action = 0
    max_val = 0
    previous_value = V[hh, state]

    for aa in range(A):
        q_aa = R_hat[state, aa] + B_sa[state, aa]

        if hh < H-1:
            for sn in range(S):
                q_aa += gamma*P_hat[state, aa, sn]*V[hh+1, sn]

        if aa == 0 or q_aa > max_val:
            max_val = q_aa
            best_action = aa

    V[hh, state] = max_val
    V[hh, state] = min(v_max, V[hh, state])
    V[hh, state] = min(previous_value, V[hh, state])

    return best_action


@numba_jit
def kernel_func(z, kernel_type):
    """
    Returns a kernel function to the real value z.

    Kernel types:

    "uniform"      : 1.0*(abs(z) <= 1)
    "triangular"   : max(0, 1 - abs(z))
    "gaussian"     : exp(-z^2/2)
    "epanechnikov" : max(0, 1-z^2)
    "quartic"      : (1-z^2)^2 *(abs(z) <= 1)
    "triweight"    : (1-z^2)^3 *(abs(z) <= 1)
    "tricube"      : (1-abs(z)^3)^3 *(abs(z) <= 1)
    "cosine"       : cos( z * (pi/2) ) *(abs(z) <= 1)
    "exp-n"        : exp(-abs(z)^n/2), for n integer

    Parameters
    ----------
    z : double
    kernel_type : string
    """
    if kernel_type == "uniform":
        return 1.0 * (np.abs(z) <= 1)
    elif kernel_type == "triangular":
        return (1.0 - np.abs(z)) * (np.abs(z) <= 1)
    elif kernel_type == "gaussian":
        return np.exp(-np.power(z, 2.0) / 2.0)
    elif kernel_type == "epanechnikov":
        return (1.0 - np.power(z, 2.0)) * (np.abs(z) <= 1)
    elif kernel_type == "quartic":
        return np.power((1.0 - np.power(z, 2.0)), 2.0) * (np.abs(z) <= 1)
    elif kernel_type == "triweight":
        return np.power((1.0 - np.power(z, 2.0)), 3.0) * (np.abs(z) <= 1)
    elif kernel_type == "tricube":
        return np.power((1.0 - np.power(np.abs(z), 3.0)), 3.0) * (np.abs(z) <= 1)
    elif kernel_type == "cosine":
        return np.cos(z * np.pi / 2) * (np.abs(z) <= 1)
    elif "exp-" in kernel_type:
        exponent = _str_to_int(kernel_type.split("-")[1])
        return np.exp(-np.power(np.abs(z), exponent) / 2.0)
    else:
        raise NotImplementedError("Unknown kernel type.")


@numba_jit
def _str_to_int(s):
    """
    Source: https://github.com/numba/numba/issues/5650#issuecomment-623511109
    """
    final_index, result = len(s) - 1, 0
    for i, v in enumerate(s):
        result += (ord(v) - 48) * (10 ** (final_index - i))
    return result
