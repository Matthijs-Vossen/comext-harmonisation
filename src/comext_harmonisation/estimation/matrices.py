"""Build pair-by-code matrices for LT estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from scipy import sparse

from .shares import EstimationShares
from ..groups import ConcordanceGroups


@dataclass(frozen=True)
class GroupMatrices:
    group_id: str
    period: str
    pairs: pd.DataFrame
    codes_a: List[str]
    codes_b: List[str]
    matrix_a: sparse.csr_matrix
    matrix_b: sparse.csr_matrix
    dense_a: Optional[pd.DataFrame]
    dense_b: Optional[pd.DataFrame]


def _build_pair_index(pairs: pd.DataFrame) -> Tuple[Dict[Tuple[str, str], int], pd.MultiIndex]:
    tuples = list(zip(pairs["REPORTER"], pairs["PARTNER"]))
    index = pd.MultiIndex.from_tuples(tuples, names=["REPORTER", "PARTNER"])
    return {pair: idx for idx, pair in enumerate(tuples)}, index


def _build_sparse_matrix(
    df: pd.DataFrame,
    *,
    pair_index: Dict[Tuple[str, str], int],
    code_col: str,
    codes: List[str],
) -> sparse.csr_matrix:
    if not codes:
        return sparse.csr_matrix((len(pair_index), 0))
    if df.empty:
        return sparse.csr_matrix((len(pair_index), len(codes)))

    col_index = {code: idx for idx, code in enumerate(codes)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for row in df.itertuples(index=False):
        pair = (row.REPORTER, row.PARTNER)
        rows.append(pair_index[pair])
        cols.append(col_index[getattr(row, code_col)])
        data.append(row.share)

    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(len(pair_index), len(codes)), dtype=float
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def _dense_from_sparse(matrix: sparse.csr_matrix, index: pd.MultiIndex, columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix.toarray(), index=index, columns=columns)


def build_group_matrices(
    estimation: EstimationShares,
    *,
    groups: ConcordanceGroups,
    dense: bool = False,
    max_workers: int | None = None,
) -> Dict[str, GroupMatrices]:
    """Build pair-by-code matrices for each ambiguous group."""
    if estimation.shares_a.empty and estimation.shares_b.empty:
        return {}

    edges = groups.edges[groups.edges["period"] == estimation.period]
    matrices: Dict[str, GroupMatrices] = {}

    group_totals = estimation.group_totals
    if "skip_reason" in group_totals.columns:
        group_totals = group_totals[group_totals["skip_reason"] == ""]

    group_ids = sorted(group_totals["group_id"].unique())

    def _build_group_matrix(group_id: str) -> GroupMatrices:
        a_codes = (
            edges.loc[edges["group_id"] == group_id, "vintage_a_code"]
            .dropna()
            .unique()
            .tolist()
        )
        b_codes = (
            edges.loc[edges["group_id"] == group_id, "vintage_b_code"]
            .dropna()
            .unique()
            .tolist()
        )
        a_codes.sort()
        b_codes.sort()

        shares_a = estimation.shares_a[estimation.shares_a["group_id"] == group_id]
        shares_b = estimation.shares_b[estimation.shares_b["group_id"] == group_id]

        pairs = pd.concat(
            [
                shares_a[["REPORTER", "PARTNER"]],
                shares_b[["REPORTER", "PARTNER"]],
            ],
            ignore_index=True,
        ).drop_duplicates()
        pairs = pairs.sort_values(["REPORTER", "PARTNER"]).reset_index(drop=True)

        pair_index, pair_multi = _build_pair_index(pairs)

        matrix_a = _build_sparse_matrix(
            shares_a,
            pair_index=pair_index,
            code_col="vintage_a_code",
            codes=a_codes,
        )
        matrix_b = _build_sparse_matrix(
            shares_b,
            pair_index=pair_index,
            code_col="vintage_b_code",
            codes=b_codes,
        )

        dense_a = _dense_from_sparse(matrix_a, pair_multi, a_codes) if dense else None
        dense_b = _dense_from_sparse(matrix_b, pair_multi, b_codes) if dense else None

        return GroupMatrices(
            group_id=group_id,
            period=estimation.period,
            pairs=pairs,
            codes_a=a_codes,
            codes_b=b_codes,
            matrix_a=matrix_a,
            matrix_b=matrix_b,
            dense_a=dense_a,
            dense_b=dense_b,
        )

    if max_workers is None or max_workers <= 1 or len(group_ids) <= 1:
        for group_id in group_ids:
            matrices[group_id] = _build_group_matrix(group_id)
        return matrices

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_build_group_matrix, group_id): group_id for group_id in group_ids}
        results: Dict[str, GroupMatrices] = {}
        for future in as_completed(futures):
            group_id = futures[future]
            results[group_id] = future.result()

    for group_id in group_ids:
        matrices[group_id] = results[group_id]

    return matrices
