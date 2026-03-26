"""Functions for introducing translational invariance in 2nd order force constants."""

from collections import defaultdict
from typing import Optional

import numpy as np
import scipy
from scipy.sparse import csr_array

from symfc.utils.cutoff_tools import FCCutoff
from symfc.utils.solver_funcs import get_batch_slice
from symfc.utils.utils import get_indep_atoms_by_lat_trans
from symfc.utils.utils_O2 import _get_atomic_lat_trans_decompr_indices

try:
    from symfc.utils.matrix import dot_product_sparse
except ImportError:
    pass


def _find_blocks_from_sparse_columns(c: csr_array) -> dict:
    """Find connected components from column co-occurrence in sparse matrix.

    Two columns are connected if they appear as nonzero entries in the same
    row. This is equivalent to the block structure of C.T @ C but much faster
    to compute since C is much sparser.

    Uses union-find (disjoint set) data structure.
    """
    n_cols = c.shape[1]
    parent = np.arange(n_cols)
    rank = np.zeros(n_cols, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    # For each row, union all nonzero columns together
    for i in range(c.shape[0]):
        start, end = c.indptr[i], c.indptr[i + 1]
        if end - start < 2:
            continue
        cols = c.indices[start:end]
        first = cols[0]
        for j in range(1, len(cols)):
            union(first, cols[j])

    # Flatten parents (full path compression)
    for i in range(n_cols):
        parent[i] = find(i)

    # Group by root
    group = defaultdict(list)
    for i in range(n_cols):
        group[parent[i]].append(i)

    return dict(group)


def optimize_batch_size_sum_rules_O2(natom: int, n_batch: int):
    """Calculate batch size for constructing projector for sum rules."""
    if n_batch > natom:
        raise ValueError("n_batch must be smaller than N.")
    batch_size = natom * (natom // n_batch)
    return batch_size


def _auto_n_batch_sum_rules_O2(natom: int) -> int:
    """Auto-select n_batch for sum rules projector construction.

    n_batch=2 is empirically optimal for large systems: it halves the size
    of the expensive C.T @ C sparse product while keeping overhead low.
    """
    return 2


def compressed_projector_sum_rules_O2(
    trans_perms: np.ndarray,
    n_a_compress_mat: csr_array,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    n_batch: int = 1,
    use_mkl: bool = False,
    return_blocks: bool = False,
) -> "csr_array | tuple[csr_array, dict]":
    r"""Return projection matrix for translational sum rule.

    Calculate a compressed projector for translational sum rules
    efficiently using independent atom with respect to lattice translations.
    This compression is achieved using C_trans and n_a_compress_mat,
    without the need to allocate C_trans. The implementation utilizes
    get_atomic_lat_trans_decompr_indices_O3 to ensure efficient memory usage.

    Return
    ------
    Compressed projector I - P^(c).
    I - P^(c)
    = n_a_compress_mat.T @ C_trans.T
      @ [I - C_sum^(c) @ C_sum^(c).T] @ C_trans @ n_a_compress_mat
    = I - [n_a_compress_mat.T @ C_trans.T @ C_sum^(c)]
          @ [C_sum^(c).T @ C_trans @ n_a_compress_mat]

    Algorithm
    ---------
    1. C_sum^(c).T = [I, I, I, ...] of size (27N, 27N^2).
       I denotes the unit matrix of size (27N, 27N).
       C_sum^(c).T is composed of N unit matrices.
       In this representation, the translational sum rules are given by
       \sum_i FC2(i, j, a, b) = 0.

    2. To divide the computation of a compressed projector into several batches,
       C_sum^(c) and C_trans are permuted from the index order of (i, j, a, b)
       to (a, b, j, i).
       This is represented by C_sum^(c).T @ C_trans = C_sum^(c).T @ S.T @ S @ C_trans,
       where S denotes the permutation matrix that changes the index order to
       (a, b, j, i). Using this permutation, the translational sum rules are
       represented as
       C_sum^(c).T @ S.T = [
           [1_N.T, 0_N.T, 0_N.T, ...]
           [0_N.T, 1_N.T, 0_N.T, ...]
           [0_N.T, 0_N.T, 1_N.T, ...]
           ...
       ],
       where 1_N and 0_N are column vectors of size N with all elements
       equal to one and zero, respectively.
       (Example) C_sum^(c).T @ S.T = [
                    [1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 ...]
                    [0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 ...]
                    [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 ...]
                    ...
                 ]
        In this function, the permutation is achieved by using matrix reshapes.

    3. Set C_trans.T @ C_sum^(c) @ C_sum^(c).T @ C_trans
       = [(C_trans.T @ S.T) @ (S @ C_sum^(c))] @ [(C_sum^(c).T @ S.T) @ (S @ C_trans)]
       =   [T_1, T_2, ..., T_N33]
         @ (S @ C_sum^(c)) @ (C_sum^(c).T @ S.T)
         @ [T_1, T_2, ..., T_N33].T
       = \sum_i t_i @ t_i.T,
       where t_i = \sum_c T_i[:, c].
       t_i is represented by c_sum_cplmt.T in this function.
       T_i is the submatrix of size (N, n_aN33) of permuted C_trans.

    4. Compute P^(c) = \sum_i (n_a_compress_mat.T @ t_i) @ (t_i.T @ n_a_compress_mat)

    5. Compute P = I - P^(c)
    """
    n_lp, natom = trans_perms.shape
    NN9 = natom**2 * 9
    NN = natom**2

    proj_size = n_a_compress_mat.shape[1]  # type: ignore
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    if atomic_decompr_idx is None:
        atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, natom)).T.reshape(-1) * 9

    indep_atoms = get_indep_atoms_by_lat_trans(trans_perms)
    nonzero = np.zeros((natom, natom), dtype=bool)
    nonzero[indep_atoms, :] = True
    nonzero = nonzero.reshape(-1)
    if fc_cutoff is not None:
        nonzero_c = fc_cutoff.nonzero_atomic_indices_fc2()
        nonzero_c = nonzero_c.reshape((natom, natom)).T.reshape(-1)
        nonzero = nonzero & nonzero_c

    # Auto-select batch count for large systems
    if n_batch == 1 and natom > 256:
        n_batch = _auto_n_batch_sum_rules_O2(natom)
    batch_size = optimize_batch_size_sum_rules_O2(natom, n_batch=n_batch)
    ab = np.arange(9)

    all_c_compressed = []  # Collect for block structure detection

    def _compute_batch(begin, end):
        """Compute one batch of the sum rule projector."""
        size = end - begin
        size_vector = size * 9
        size_row = size_vector // natom

        nonzero_b = nonzero[begin:end]
        size_data = np.count_nonzero(nonzero_b) * 9
        if size_data == 0:
            return None

        decompr_idx_b = decompr_idx[begin:end][nonzero_b]
        c_sum_cplmt = csr_array(
            (
                np.ones(size_data, dtype="double"),
                (
                    np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 9)],
                    (ab[:, None] + decompr_idx_b[None, :]).reshape(-1),
                ),
            ),
            shape=(size_row, NN9 // n_lp),
            dtype="double",
        )
        c_sum_cplmt = dot_product_sparse(
            c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl
        )
        if return_blocks:
            all_c_compressed.append(c_sum_cplmt)
        return dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    batches = list(zip(*get_batch_slice(NN, batch_size), strict=True))

    if len(batches) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(batches)) as executor:
            results = list(executor.map(
                lambda b: _compute_batch(b[0], b[1]), batches
            ))
        for result in results:
            if result is not None:
                proj_cplmt += result
    else:
        for begin, end in batches:
            result = _compute_batch(begin, end)
            if result is not None:
                proj_cplmt += result

    proj_cplmt /= natom
    if not return_blocks:
        proj = scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt
    else:
        # Skip expensive I-P on full matrix; pass complement projector directly.
        # The I-P will be applied per-block in eigsh_projector_sumrule (much cheaper).
        proj = proj_cplmt

    if return_blocks:
        # Find block structure from column co-occurrence in C matrices
        # Much faster than connected_components on the large C.T @ C
        from scipy.sparse import vstack as sparse_vstack

        c_all = sparse_vstack(all_c_compressed, format="csr")
        blocks = _find_blocks_from_sparse_columns(c_all)
        del c_all
        return proj, blocks
    return proj


def compressed_projector_sum_rules_O2_stable(
    trans_perms: np.ndarray,
    n_a_compress_mat: csr_array,
    atomic_decompr_idx: Optional[np.ndarray] = None,
    fc_cutoff: Optional[FCCutoff] = None,
    n_batch: int = 1,
    use_mkl: bool = False,
) -> csr_array:
    r"""Return projection matrix for translational sum rule.

    Calculate a compressed projector for translational sum rules.
    This compression is achieved using C_trans and n_a_compress_mat,
    without the need to allocate C_trans. The implementation utilizes
    get_atomic_lat_trans_decompr_indices_O3 to ensure efficient memory usage.

    Return
    ------
    Compressed projector I - P^(c).
    I - P^(c)
    = n_a_compress_mat.T @ C_trans.T
      @ [I - C_sum^(c) @ C_sum^(c).T] @ C_trans @ n_a_compress_mat
    = I - [n_a_compress_mat.T @ C_trans.T @ C_sum^(c)]
          @ [C_sum^(c).T @ C_trans @ n_a_compress_mat]

    Algorithm
    ---------
    1. C_sum^(c).T = [I, I, I, ...] of size (27N, 27N^2).
       I denotes the unit matrix of size (27N, 27N).
       C_sum^(c).T is composed of N unit matrices.
       In this representation, the translational sum rules are given by
       \sum_i FC2(i, j, a, b) = 0.

    2. To divide the computation of a compressed projector into several batches,
       C_sum^(c) and C_trans are permuted from the index order of (i, j, a, b)
       to (a, b, j, i).
       This is represented by C_sum^(c).T @ C_trans = C_sum^(c).T @ S.T @ S @ C_trans,
       where S denotes the permutation matrix that changes the index order to
       (a, b, j, i). Using this permutation, the translational sum rules are
       represented as
       C_sum^(c).T @ S.T = [
           [1_N.T, 0_N.T, 0_N.T, ...]
           [0_N.T, 1_N.T, 0_N.T, ...]
           [0_N.T, 0_N.T, 1_N.T, ...]
           ...
       ],
       where 1_N and 0_N are column vectors of size N with all elements
       equal to one and zero, respectively.
       (Example) C_sum^(c).T @ S.T = [
                    [1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 ...]
                    [0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 ...]
                    [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 ...]
                    ...
                 ]
        In this function, the permutation is achieved by using matrix reshapes.

    3. Set C_trans.T @ C_sum^(c) @ C_sum^(c).T @ C_trans
       = [(C_trans.T @ S.T) @ (S @ C_sum^(c))] @ [(C_sum^(c).T @ S.T) @ (S @ C_trans)]
       =   [T_1, T_2, ..., T_N33]
         @ (S @ C_sum^(c)) @ (C_sum^(c).T @ S.T)
         @ [T_1, T_2, ..., T_N33].T
       = \sum_i t_i @ t_i.T,
       where t_i = \sum_c T_i[:, c].
       t_i is represented by c_sum_cplmt.T in this function.
       T_i is the submatrix of size (N, n_aN33) of permuted C_trans.

    4. Compute P^(c) = \sum_i (n_a_compress_mat.T @ t_i) @ (t_i.T @ n_a_compress_mat)

    5. Compute P = I - P^(c)
    """
    n_lp, natom = trans_perms.shape
    NN9 = natom**2 * 9
    NN = natom**2

    proj_size = n_a_compress_mat.shape[1]  # type: ignore
    proj_cplmt = csr_array((proj_size, proj_size), dtype="double")

    if atomic_decompr_idx is None:
        atomic_decompr_idx = _get_atomic_lat_trans_decompr_indices(trans_perms)

    decompr_idx = atomic_decompr_idx.reshape((natom, natom)).T.reshape(-1) * 9
    if fc_cutoff is not None:
        nonzero = fc_cutoff.nonzero_atomic_indices_fc2()
        nonzero = nonzero.reshape((natom, natom)).T.reshape(-1)

    batch_size = optimize_batch_size_sum_rules_O2(natom, n_batch=n_batch)
    ab = np.arange(9)
    for begin, end in zip(*get_batch_slice(NN, batch_size), strict=True):
        size = end - begin
        size_vector = size * 9
        size_row = size_vector // natom

        if fc_cutoff is None:
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_vector, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom),
                        (ab[:, None] + decompr_idx[begin:end][None, :]).reshape(-1),
                    ),
                ),
                shape=(size_row, NN9 // n_lp),
                dtype="double",
            )
        else:
            nonzero_b = nonzero[begin:end]
            decompr_idx_b = decompr_idx[begin:end][nonzero_b]
            size_data = np.count_nonzero(nonzero_b) * 9
            c_sum_cplmt = csr_array(
                (
                    np.ones(size_data, dtype="double"),
                    (
                        np.repeat(np.arange(size_row), natom)[np.tile(nonzero_b, 9)],
                        (ab[:, None] + decompr_idx_b[None, :]).reshape(-1),
                    ),
                ),
                shape=(size_row, NN9 // n_lp),
                dtype="double",
            )

        c_sum_cplmt = dot_product_sparse(c_sum_cplmt, n_a_compress_mat, use_mkl=use_mkl)
        proj_cplmt += dot_product_sparse(c_sum_cplmt.T, c_sum_cplmt, use_mkl=use_mkl)

    proj_cplmt /= n_lp * natom
    return scipy.sparse.identity(proj_cplmt.shape[0]) - proj_cplmt
