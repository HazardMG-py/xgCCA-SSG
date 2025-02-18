import torch as th


def select_subspace(R, threshold=0.05):
    """
    Given a cross-correlation matrix R (of shape [D, D]),
    threshold it by keeping only entries with absolute value greater than `threshold`,
    then iteratively remove rows or columns to maximize the density of the remaining submatrix.

    Returns:
        selected_indices: a sorted list of feature indices (0-indexed) that form the informative subspace.
    """
    D = R.shape[0]
    # Create binary matrix: 1 if |R_ij| > threshold, else 0.
    M = (th.abs(R) > threshold).float()
    # Start with all feature indices.
    rows = list(range(D))
    cols = list(range(D))

    def current_density(rows, cols):
        if len(rows) == 0 or len(cols) == 0:
            return 0.0
        subM = M[rows][:, cols]
        return subM.sum().item() / (len(rows) * len(cols))

    curr_density = current_density(rows, cols)
    improved = True
    while improved:
        improved = False
        best_density = curr_density
        best_remove = None
        best_remove_type = None
        # Try removing each row
        for r in rows.copy():
            new_rows = [i for i in rows if i != r]
            if len(new_rows) == 0:
                continue
            candidate = current_density(new_rows, cols)
            if candidate > best_density:
                best_density = candidate
                best_remove = r
                best_remove_type = 'row'
        # Try removing each column
        for c in cols.copy():
            new_cols = [j for j in cols if j != c]
            if len(new_cols) == 0:
                continue
            candidate = current_density(rows, new_cols)
            if candidate > best_density:
                best_density = candidate
                best_remove = c
                best_remove_type = 'col'
        if best_remove is not None:
            if best_remove_type == 'row':
                rows.remove(best_remove)
            else:
                cols.remove(best_remove)
            curr_density = best_density
            improved = True
    # Use intersection of rows and columns as selected subspace
    selected = sorted(list(set(rows).intersection(set(cols))))
    if len(selected) == 0:
        selected = sorted(list(set(rows).union(set(cols))))
    return selected
