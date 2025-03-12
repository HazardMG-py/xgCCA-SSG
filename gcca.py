import torch as th

def select_subspace(R, threshold=0.05, percentile=10):
    """
    Greedy subspace selection (non-differentiable, optional fallback).
    Args:
        R: Cross-correlation matrix [D, D]
        threshold: Correlation threshold for edge inclusion
        percentile: Percentile of rows/cols to remove per iteration
    Returns:
        selected: List of indices forming the subspace
    """
    D = R.shape[0]
    M = (th.abs(R) > threshold).float()
    rows, cols = list(range(D)), list(range(D))

    def density(rows, cols):
        if not rows or not cols:
            return 0.0
        return M[rows][:, cols].sum().item() / (len(rows) * len(cols))

    improved = True
    while improved:
        improved = False
        row_sums = M[rows][:, cols].sum(1)
        col_sums = M[rows][:, cols].sum(0)
        row_cutoff = th.quantile(row_sums, percentile / 100)
        col_cutoff = th.quantile(col_sums, percentile / 100)

        new_rows = [r for r in rows if row_sums[rows.index(r)] >= row_cutoff]
        new_cols = [c for c in cols if col_sums[cols.index(c)] >= col_cutoff]
        if density(new_rows, new_cols) > density(rows, cols):
            rows, cols = new_rows, new_cols
            improved = True

    selected = sorted(list(set(rows).intersection(cols)))
    return selected if selected else list(range(D))