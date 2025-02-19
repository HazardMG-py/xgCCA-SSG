def select_subspace(R, threshold=0.05, percentile=10):
    """
    Optimized version with batch removal of low-density rows/columns.
    """
    D = R.shape[0]
    M = (th.abs(R) > threshold).float()
    rows = list(range(D))
    cols = list(range(D))

    def current_density(rows, cols):
        if not rows or not cols:
            return 0.0
        subM = M[rows][:, cols]
        return subM.sum().item() / (len(rows) * len(cols))

    improved = True
    while improved:
        improved = False

        # Compute row/column sums
        row_sums = M[rows][:, cols].sum(1)  # Sum over remaining columns
        col_sums = M[rows][:, cols].sum(0)  # Sum over remaining rows

        # Find rows/columns to remove (bottom 10% by default)
        row_cutoff = np.percentile(row_sums.cpu().numpy(), percentile)
        col_cutoff = np.percentile(col_sums.cpu().numpy(), percentile)

        rows_to_remove = [rows[i] for i in range(len(rows)) if row_sums[i] < row_cutoff]
        cols_to_remove = [cols[j] for j in range(len(cols)) if col_sums[j] < col_cutoff]

        # Try removing batch
        new_rows = [i for i in rows if i not in rows_to_remove]
        new_cols = [j for j in cols if j not in cols_to_remove]
        candidate_density = current_density(new_rows, new_cols)

        if candidate_density > current_density(rows, cols):
            rows = new_rows
            cols = new_cols
            improved = True

    selected = sorted(list(set(rows).intersection(cols)))
    return selected if len(selected) > 0 else sorted(rows + cols)