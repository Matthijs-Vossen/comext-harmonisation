"""Solve LT conversion weights using OSQP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import osqp
from scipy import sparse

from .matrices import GroupMatrices
from .shares import EstimationShares
from ..concordance.groups import ConcordanceGroups
from ..weights.schema import WEIGHT_COLUMNS

@dataclass(frozen=True)
class SolverDiagnostics:
    group_id: str
    period: str
    status: str
    objective: float
    n_vars: int
    n_obs: int
    min_weight: float
    max_weight: float
    max_row_sum_dev: float


def _build_allowed_links(
    edges: pd.DataFrame,
    *,
    from_codes: List[str],
    to_codes: List[str],
    from_col: str,
    to_col: str,
) -> Dict[int, List[int]]:
    from_index = {code: idx for idx, code in enumerate(from_codes)}
    to_index = {code: idx for idx, code in enumerate(to_codes)}
    allowed_by_to: Dict[int, List[int]] = {idx: [] for idx in range(len(to_codes))}

    for row in edges.itertuples(index=False):
        from_code = getattr(row, from_col)
        to_code = getattr(row, to_col)
        if from_code not in from_index or to_code not in to_index:
            continue
        k_idx = from_index[from_code]
        s_idx = to_index[to_code]
        allowed_by_to[s_idx].append(k_idx)

    for s_idx, k_list in allowed_by_to.items():
        allowed_by_to[s_idx] = sorted(set(k_list))

    return allowed_by_to


def _build_qp_matrices(
    *,
    X: sparse.csr_matrix,
    Y: sparse.csr_matrix,
    allowed_by_to: Dict[int, List[int]],
) -> Tuple[sparse.csc_matrix, np.ndarray, sparse.csc_matrix, np.ndarray, np.ndarray, List[int], List[int]]:
    k_size = X.shape[1]
    s_size = Y.shape[1]

    M = (X.T @ X).toarray()
    N = (X.T @ Y).toarray()

    blocks = []
    q: List[float] = []
    var_k: List[int] = []
    var_s: List[int] = []

    for s_idx in range(s_size):
        k_indices = allowed_by_to.get(s_idx, [])
        if not k_indices:
            continue
        block = 2.0 * M[np.ix_(k_indices, k_indices)]
        blocks.append(sparse.csc_matrix(block))
        q.extend((-2.0 * N[k_indices, s_idx]).tolist())
        for k_idx in k_indices:
            var_k.append(k_idx)
            var_s.append(s_idx)

    if not blocks:
        P = sparse.csc_matrix((0, 0))
        q_vec = np.array([])
    else:
        P = sparse.block_diag(blocks, format="csc")
        q_vec = np.array(q, dtype=float)

    rows = np.array(var_k, dtype=int)
    cols = np.arange(len(var_k), dtype=int)
    data = np.ones(len(var_k), dtype=float)
    Aeq = sparse.coo_matrix((data, (rows, cols)), shape=(k_size, len(var_k))).tocsc()

    A = sparse.vstack([Aeq, sparse.eye(len(var_k), format="csc")], format="csc")

    l = np.concatenate([np.ones(k_size), np.zeros(len(var_k))])
    u = np.concatenate([np.ones(k_size), np.full(len(var_k), np.inf)])

    return P, q_vec, A, l, u, var_k, var_s


def _solve_group(
    *,
    group: GroupMatrices,
    groups: ConcordanceGroups,
    direction: str,
    vintage_a_year: str,
    vintage_b_year: str,
) -> Tuple[pd.DataFrame, SolverDiagnostics]:
    edges = groups.edges[
        (groups.edges["period"] == group.period) & (groups.edges["group_id"] == group.group_id)
    ]

    if direction == "a_to_b":
        from_codes = group.codes_a
        to_codes = group.codes_b
        X = group.matrix_a
        Y = group.matrix_b
        from_col = "vintage_a_code"
        to_col = "vintage_b_code"
        from_year = vintage_a_year
        to_year = vintage_b_year
    else:
        from_codes = group.codes_b
        to_codes = group.codes_a
        X = group.matrix_b
        Y = group.matrix_a
        from_col = "vintage_b_code"
        to_col = "vintage_a_code"
        from_year = vintage_b_year
        to_year = vintage_a_year

    allowed_by_to = _build_allowed_links(
        edges,
        from_codes=from_codes,
        to_codes=to_codes,
        from_col=from_col,
        to_col=to_col,
    )

    P, q, A, l, u, var_k, var_s = _build_qp_matrices(X=X, Y=Y, allowed_by_to=allowed_by_to)

    solver = osqp.OSQP()
    setup_kwargs = dict(
        P=P,
        q=q,
        A=A,
        l=l,
        u=u,
        verbose=False,
        eps_abs=1e-8,
        eps_rel=1e-8,
        polishing=True,
        scaling=20,
        max_iter=200000,
    )
    solver.setup(**setup_kwargs)
    try:
        result = solver.solve(raise_error=True)
    except Exception as exc:
        raise RuntimeError(
            "OSQP failed for "
            f"group_id={group.group_id} period={group.period} direction={direction} "
            f"from_year={from_year} to_year={to_year}"
        ) from exc

    weights = result.x
    status = result.info.status

    k_size = len(from_codes)
    row_sums = np.zeros(k_size)
    for idx, k_idx in enumerate(var_k):
        row_sums[k_idx] += weights[idx]

    max_row_sum_dev = float(np.max(np.abs(row_sums - 1.0))) if k_size else 0.0

    weight_rows = []
    for idx, (k_idx, s_idx) in enumerate(zip(var_k, var_s)):
        weight_rows.append(
            {
                "period": group.period,
                "from_vintage_year": from_year,
                "to_vintage_year": to_year,
                "from_code": from_codes[k_idx],
                "to_code": to_codes[s_idx],
                "group_id": group.group_id,
                "weight": float(weights[idx]),
            }
        )

    weight_df = pd.DataFrame(weight_rows)
    if not weight_df.empty:
        weight_df = weight_df[WEIGHT_COLUMNS]

    diagnostics = SolverDiagnostics(
        group_id=group.group_id,
        period=group.period,
        status=status,
        objective=float(result.info.obj_val),
        n_vars=len(var_k),
        n_obs=X.shape[0],
        min_weight=float(np.min(weights)) if len(weights) else 0.0,
        max_weight=float(np.max(weights)) if len(weights) else 0.0,
        max_row_sum_dev=max_row_sum_dev,
    )

    return weight_df, diagnostics


def estimate_weights(
    *,
    estimation: EstimationShares,
    matrices: Dict[str, GroupMatrices],
    groups: ConcordanceGroups,
    direction: str | None = None,
    max_workers: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate conversion weights for all groups in the matrices."""
    direction = direction or estimation.direction

    weight_tables = []
    diagnostics = []

    group_ids = list(matrices.keys())

    def _solve_for_group(group_id: str) -> tuple[pd.DataFrame, SolverDiagnostics]:
        group = matrices[group_id]
        return _solve_group(
            group=group,
            groups=groups,
            direction=direction,
            vintage_a_year=estimation.vintage_a_year,
            vintage_b_year=estimation.vintage_b_year,
        )

    if max_workers is None or max_workers <= 1 or len(group_ids) <= 1:
        for group_id in group_ids:
            weights, diag = _solve_for_group(group_id)
            weight_tables.append(weights)
            diagnostics.append(diag.__dict__)
    else:
        results: Dict[str, tuple[pd.DataFrame, SolverDiagnostics]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_solve_for_group, group_id): group_id for group_id in group_ids}
            for future in as_completed(futures):
                group_id = futures[future]
                results[group_id] = future.result()
        for group_id in sorted(results.keys()):
            weights, diag = results[group_id]
            weight_tables.append(weights)
            diagnostics.append(diag.__dict__)

    weights_df = (
        pd.concat(weight_tables, ignore_index=True)
        if weight_tables
        else pd.DataFrame(columns=WEIGHT_COLUMNS)
    )
    diagnostics_df = pd.DataFrame(diagnostics)

    return weights_df, diagnostics_df
