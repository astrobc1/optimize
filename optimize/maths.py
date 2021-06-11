# Maths
import numpy as np
import numba

@numba.njit(nogil=True)
def compute_rms(residuals):
    """Computes the rms of an array of residuals.

    Args:
        residuals (np.ndarray): The array of residuals.

    Returns:
        float: The RMS objective.
    """
    return np.sqrt(np.nansum(residuals**2) / len(residuals))

@numba.njit(nogil=True)
def compute_chi2(residuals, errors):
    """Computes the chi2 of an array of residuals.

    Args:
        residuals (np.ndarray): The array of residuals.
        errors (np.ndarray): The array of errors.

    Returns:
        float: The chi2 objective.
    """
    return np.nansum((residuals / errors)**2)

@numba.njit(nogil=True)
def compute_redchi2(residuals, errors, n_dof):
    """Computes the reduced chi2 of an array of residuals.

    Args:
        residuals (np.ndarray): The array of residuals.
        errors (np.ndarray): The array of errors.
        n_dof (int): The degrees of freedom.

    Returns:
        float: The reduced chi2 objective.
    """
    return compute_chi2(residuals, errors) / n_dof

@numba.njit(nogil=True)
def compute_stationary_dist_matrix(x1, x2):
    """Computes the distance matrix, D_ij = |x_i - x_j|.

    Args:
        x1 (np.ndarray): The first vec to use (x_i).
        x2 (np.ndarray): The second vec to use (x_j).

    Returns:
        np.ndarray: The distance matrix D_ij.
    """
    n1 = len(x1)
    n2 = len(x2)
    out = np.zeros(shape=(n1, n2), dtype=numba.float64)
    for i in range(n1):
        for j in range(n2):
            out[i, j] = np.abs(x1[i] - x2[j])
    return out