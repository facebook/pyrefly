def transform(
    raw_X, n_features: int, dtype, alternate_sign: bool | int = 1, seed: int = 0
) -> tuple[int, list[int], list[int], list[float]]:
    """Guts of FeatureHasher.transform.
    Returns
    -------
    n_samples : integer
    indices, indptr, values : lists
        For constructing a scipy.sparse.csr_matrix.
    """
