import torch
import numpy as np


def select_subspace(R, threshold=0.2, percentile=20):
    """
    Identifies biclique subspaces via iterative pruning of low-correlation features.

    Args:
        R (torch.Tensor): Cross-correlation matrix [D, D].
        threshold (float): Minimum absolute correlation to retain.
        percentile (int): Percentile cutoff for pruning rows/columns.

    Returns:
        list: Indices of features forming the biclique subspace.
    """
    D = R.shape[0]
    M = (torch.abs(R) > threshold).float()  # Binary mask
    rows = list(range(D))  # Row indices (features from view A)
    cols = list(range(D))  # Column indices (features from view B)

    def compute_density(r, c):
        if not r or not c:
            return 0.0
        return M[r][:, c].sum().item() / (len(r) * len(c))

    improved = True
    while improved:
        improved = False
        # Compute row/column sums within current subspace
        row_sums = M[rows][:, cols].sum(dim=1)  # Sum over columns
        col_sums = M[rows][:, cols].sum(dim=0)  # Sum over rows

        # Determine cutoffs based on percentile
        row_cutoff = np.percentile(row_sums.cpu().numpy(), percentile)
        col_cutoff = np.percentile(col_sums.cpu().numpy(), percentile)

        # Identify weak rows/columns
        weak_rows = [rows[i] for i, s in enumerate(row_sums) if s < row_cutoff]
        weak_cols = [cols[j] for j, s in enumerate(col_sums) if s < col_cutoff]

        # Propose new subspace
        new_rows = [i for i in rows if i not in weak_rows]
        new_cols = [j for j in cols if j not in weak_cols]
        new_density = compute_density(new_rows, new_cols)

        # Update if density improves
        if new_density > compute_density(rows, cols):
            rows, cols = new_rows, new_cols
            improved = True

    # Biclique = features surviving in both rows and columns
    biclique = sorted(list(set(rows).intersection(cols)))
    # Fallback: Use union if biclique is empty (avoid returning empty list)
    return biclique if biclique else sorted(list(set(rows + cols)))