import numpy as np

def _prepare_data_for_autocorrelation(x, L = 1):
    """
    Parameters:
            x: NumPy array of shape (N, d) or SciPy sparse array
    """
    N = x.shape[0]

    if L < 1 or L >= N:
        raise ValueError("Lag must be a positive integer, but lower than length of time-serie.")
    
    M = N - L # number of pairs
    Su = x[0:M] # unshifted serie
    Sl = x[L:(M+L)] # lagged serie

    return Su, Sl, M

def calculate_pearson_correlation(x, L = 1):
    Su, Sl, M = _prepare_data_for_autocorrelation(x = x, L = L)

    # Expected shape (d, )
    mean_Su = np.mean(Su, axis = 0)
    mean_Sl = np.mean(Sl, axis = 0)
    mean_Su_sq = np.mean(Su ** 2, axis = 0)
    mean_Sl_sq = np.mean(Sl ** 2, axis = 0)
    mean_SuSl = np.mean(Su * Sl, axis = 0)

    var_Su_c = mean_Su_sq - mean_Su ** 2
    var_Sl_c = mean_Sl_sq - mean_Sl ** 2
    cov_c = mean_SuSl - mean_Su * mean_Sl

    var_Su = np.sum(var_Su_c)
    var_Sl = np.sum(var_Sl_c)
    cov = np.sum(cov_c)

    if (var_Su == 0) or (var_Sl == 0):
        return float('nan')

    corr = cov / np.sqrt(var_Su * var_Sl)
    return corr

def calculate_cosine_correlation(x, L = 1, sparse: bool = True, norms = None):
    Su, Sl, M = _prepare_data_for_autocorrelation(x = x, L = L)

    if norms is not None:
        norm_Su = norms[0:M]
        norm_Sl = norms[L:(M+L)]
    else:
        if sparse:
            norm_Su = np.array([np.sqrt(s.multiply(s).sum())for s in Su])
            norm_Sl = np.array([np.sqrt(s.multiply(s).sum())for s in Sl])
        else:
            norm_Su = np.linalg.norm(Su, axis = 1)
            norm_Sl = np.linalg.norm(Sl, axis = 1)

    if sparse:
        dot_products = np.array([(sui.multiply(sli)).sum() for sui, sli in zip(Su, Sl)], dtype = np.float64)
    else:
       dot_products = np.einsum('ij, ij -> i', Su, Sl)

    # Exclude pairs with zero norm
    non_zero_norms = (norm_Su > 0) & (norm_Sl > 0)
    corr = np.zeros(M)
    corr[non_zero_norms] = dot_products[non_zero_norms] / (norm_Su[non_zero_norms] * norm_Sl[non_zero_norms])

    return np.mean(corr[non_zero_norms])
