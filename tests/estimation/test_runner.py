import pandas as pd
import pytest
from scipy import sparse

from comext_harmonisation.estimation.matrices import GroupMatrices
from comext_harmonisation.estimation.runner import (
    run_weight_estimation_for_period,
    run_weight_estimation_for_period_multi,
)
from comext_harmonisation.estimation.shares import EstimationShares
from comext_harmonisation.groups import build_concordance_groups
from comext_harmonisation.weights import WEIGHT_COLUMNS


def _make_groups(period: str) -> tuple:
    edges = pd.DataFrame(
        [
            {
                "period": period,
                "vintage_a_year": period[:4],
                "vintage_b_year": period[4:],
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
            {
                "period": period,
                "vintage_a_year": period[:4],
                "vintage_b_year": period[4:],
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000012",
            },
        ]
    )
    groups = build_concordance_groups(edges)
    group_id = groups.group_summary.loc[0, "group_id"]
    return groups, group_id


def _make_estimation(period: str, group_id: str) -> EstimationShares:
    shares_cols = [
        "period",
        "vintage_year",
        "REPORTER",
        "PARTNER",
        "group_id",
        "vintage_a_code",
        "vintage_b_code",
        "value_eur",
        "share",
    ]
    empty_shares = pd.DataFrame(columns=shares_cols)
    totals = pd.DataFrame(
        [
            {
                "period": period,
                "group_id": group_id,
                "total_value_eur_a": 1.0,
                "total_value_eur_b": 1.0,
                "n_rows_a": 1,
                "n_rows_b": 1,
                "n_pairs_a": 1,
                "n_pairs_b": 1,
                "skip_reason": "",
            }
        ]
    )
    skipped = pd.DataFrame(columns=["period", "group_id", "skip_reason"])
    return EstimationShares(
        period=period,
        direction="a_to_b",
        measure="VALUE_EUR",
        vintage_a_year=period[:4],
        vintage_b_year=period[4:],
        shares_a=empty_shares.copy(),
        shares_b=empty_shares.copy(),
        group_totals=totals,
        skipped_groups=skipped,
    )


def _make_matrices(period: str, group_id: str) -> dict[str, GroupMatrices]:
    pairs = pd.DataFrame(
        [
            {"REPORTER": "NL", "PARTNER": "BE"},
            {"REPORTER": "NL", "PARTNER": "FR"},
        ]
    )
    matrix_a = sparse.csr_matrix([[1.0], [0.0]])
    matrix_b = sparse.csr_matrix([[0.6, 0.4], [0.0, 1.0]])
    return {
        group_id: GroupMatrices(
            group_id=group_id,
            period=period,
            pairs=pairs,
            codes_a=["00000001"],
            codes_b=["00000011", "00000012"],
            matrix_a=matrix_a,
            matrix_b=matrix_b,
            dense_a=None,
            dense_b=None,
        )
    }


def test_run_weight_estimation_for_period_writes_outputs(tmp_path, monkeypatch):
    period = "20092010"
    groups, group_id = _make_groups(period)
    estimation = _make_estimation(period, group_id)
    matrices = _make_matrices(period, group_id)

    weights = pd.DataFrame(
        [
            {
                "period": period,
                "from_vintage_year": period[:4],
                "to_vintage_year": period[4:],
                "from_code": "00000001",
                "to_code": "00000011",
                "group_id": group_id,
                "weight": 1.0,
            }
        ]
    )[WEIGHT_COLUMNS]
    diagnostics = pd.DataFrame(
        [
            {
                "group_id": group_id,
                "period": period,
                "measure": "VALUE_EUR",
                "status": "solved",
                "objective": -0.1,
                "n_vars": 1,
                "n_obs": 2,
                "min_weight": 1.0,
                "max_weight": 1.0,
                "max_row_sum_dev": 0.0,
            }
        ]
    )
    deterministic = pd.DataFrame(
        [
            {
                "period": period,
                "from_vintage_year": period[:4],
                "to_vintage_year": period[4:],
                "from_code": "00000002",
                "to_code": "00000012",
                "group_id": group_id,
                "weight": 1.0,
            }
        ]
    )[WEIGHT_COLUMNS]

    import comext_harmonisation.estimation.runner as runner

    monkeypatch.setattr(runner, "load_concordance_groups", lambda **kwargs: groups)
    monkeypatch.setattr(runner, "prepare_estimation_shares_for_period", lambda **kwargs: estimation)
    monkeypatch.setattr(runner, "build_group_matrices", lambda *args, **kwargs: matrices)
    monkeypatch.setattr(runner, "estimate_weights", lambda **kwargs: (weights, diagnostics))
    monkeypatch.setattr(runner, "build_deterministic_mappings", lambda *args, **kwargs: deterministic)

    outputs = run_weight_estimation_for_period(
        period=period,
        direction="a_to_b",
        measure="VALUE_EUR",
        output_dir=tmp_path,
    )

    assert outputs.weights_path.exists()
    assert outputs.deterministic_path.exists()
    assert outputs.diagnostics_path.exists()
    assert outputs.summary_csv_path is not None
    assert outputs.summary_csv_path.exists()

    summary = pd.read_csv(outputs.summary_csv_path)
    assert summary.loc[0, "n_groups_total"] == 1
    assert summary.loc[0, "n_groups_with_data"] == 1
    assert summary.loc[0, "n_groups_solved"] == 1
    assert summary.loc[0, "n_groups_failed"] == 0
    assert summary.loc[0, "n_weight_rows_ambiguous"] == 1
    assert summary.loc[0, "n_weight_rows_deterministic"] == 1
    assert summary.loc[0, "total_pairs"] == 2
    assert summary.loc[0, "total_obs"] == 2

    group_diag = pd.read_csv(outputs.diagnostics_path)
    assert group_diag.loc[0, "n_pairs"] == 2
    assert group_diag.loc[0, "n_codes_a"] == 1
    assert group_diag.loc[0, "n_codes_b"] == 2


def test_run_weight_estimation_for_period_fail_fast(tmp_path, monkeypatch):
    period = "20092010"
    groups, group_id = _make_groups(period)
    estimation = _make_estimation(period, group_id)
    matrices = _make_matrices(period, group_id)

    diagnostics = pd.DataFrame(
        [
            {
                "group_id": group_id,
                "period": period,
                "measure": "VALUE_EUR",
                "status": "primal infeasible",
                "objective": 0.0,
                "n_vars": 1,
                "n_obs": 2,
                "min_weight": 0.0,
                "max_weight": 0.0,
                "max_row_sum_dev": 1.0,
            }
        ]
    )

    import comext_harmonisation.estimation.runner as runner

    monkeypatch.setattr(runner, "load_concordance_groups", lambda **kwargs: groups)
    monkeypatch.setattr(runner, "prepare_estimation_shares_for_period", lambda **kwargs: estimation)
    monkeypatch.setattr(runner, "build_group_matrices", lambda *args, **kwargs: matrices)
    monkeypatch.setattr(runner, "estimate_weights", lambda **kwargs: (pd.DataFrame(columns=WEIGHT_COLUMNS), diagnostics))
    monkeypatch.setattr(runner, "build_deterministic_mappings", lambda *args, **kwargs: pd.DataFrame(columns=WEIGHT_COLUMNS))

    with pytest.raises(RuntimeError):
        run_weight_estimation_for_period(
            period=period,
            direction="a_to_b",
            measure="VALUE_EUR",
            output_dir=tmp_path,
        )


def test_run_weight_estimation_for_period_multi_writes_combined_summary(tmp_path, monkeypatch):
    period = "20092010"
    direction = "a_to_b"

    def _fake_run(*, measure: str, **kwargs):
        summary = pd.DataFrame(
            [
                {
                    "period": period,
                    "direction": direction,
                    "measure": measure,
                    "n_groups_total": 1,
                    "n_groups_with_data": 1,
                    "n_groups_solved": 1,
                    "n_groups_failed": 0,
                    "n_groups_skipped": 0,
                    "n_weight_rows_ambiguous": 1,
                    "n_weight_rows_deterministic": 1,
                    "total_pairs": 2,
                    "total_obs": 2,
                    "max_row_sum_dev_min": 0.0,
                    "max_row_sum_dev_max": 0.0,
                    "max_row_sum_dev_mean": 0.0,
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "ended_at": "2026-01-01T00:00:01+00:00",
                    "elapsed_seconds": 1.0,
                }
            ]
        )
        return type("Out", (), {"summary": summary, "summary_csv_path": None})()

    import comext_harmonisation.estimation.runner as runner

    monkeypatch.setattr(runner, "run_weight_estimation_for_period", _fake_run)

    results = run_weight_estimation_for_period_multi(
        period=period,
        direction=direction,
        measures=["VALUE_EUR", "QUANTITY_KG"],
        output_dir=tmp_path,
    )

    assert len(results) == 2
    assert all(result.summary_csv_path is None for result in results)
    assert not (tmp_path / "summaries" / f"run_summary_{period}_{direction}_value_eur.csv").exists()
    assert not (tmp_path / "summaries" / f"run_summary_{period}_{direction}_quantity_kg.csv").exists()
    summary_path = tmp_path / "summaries" / f"run_summary_{period}_{direction}.csv"
    assert summary_path.exists()
    summary_txt_path = tmp_path / "summaries" / f"run_summary_{period}_{direction}.txt"
    assert summary_txt_path.exists()
    combined = pd.read_csv(summary_path)
    assert set(combined["measure"]) == {"VALUE_EUR", "QUANTITY_KG"}
