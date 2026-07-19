import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def e_value_from_p_value(p_value):
    """
    Convert a p-value to an e-value.
    For a p-value p, an e-value is 1/p (or 1/p if p > 0).
    """
    if p_value <= 0:
        return np.inf
    return 1.0 / p_value

def simple_e_value(returns, null_mean=0.0):
    """
    Compute a simple e-value for testing H0: mean = null_mean.
    Uses the likelihood ratio: E = exp( (n/2) * ( (mean - null_mean)^2 / var ) )
    Under the null, E[E] ≤ 1 (by e-value property).
    """
    if len(returns) < 3:
        return 1.0
    n = len(returns)
    mean = np.mean(returns)
    var = np.var(returns, ddof=1)
    if var == 0:
        return 1.0
    # Likelihood ratio e-value
    stat = (mean - null_mean)**2 / (var / n)
    ev = np.exp(0.5 * stat)
    return ev

def e_process(returns, window=20):
    """
    Compute an e-process: a sequence of e-values that allows optional stopping.
    This is the cumulative product of e-values.
    """
    if len(returns) < window:
        return np.array([1.0])
    e_values = []
    for i in range(window, len(returns) + 1):
        segment = returns[i-window:i]
        ev = simple_e_value(segment)
        e_values.append(ev)
    # Cumulative product (e-process)
    e_process = np.cumprod(e_values)
    return e_process

def e_score(returns, macro_df, window=20):
    """
    Compute per-ETF e-value score.
    Higher e-value = stronger evidence against null (positive expected return).
    """
    if len(returns) < window + 5 or macro_df is None or len(macro_df) < window + 5:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < window + 5:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)[-1]
    # Compute e-process
    ep = e_process(returns, window)
    if len(ep) == 0:
        return 1.0
    # Score = final e-value of the e-process (or log e-value for numerical stability)
    final_e = ep[-1]
    # Apply macro factor modulation
    final_e = final_e * (1 + macro_factor * 0.5)
    # Clip to reasonable range
    final_e = max(0.1, min(100.0, final_e))
    return float(final_e)
