from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import comext_harmonisation.estimation.runner as runner


def _edge(vintage_a_code, vintage_b_code, period="20002001", vintage_a_year="2000", vintage_b_year="2001"):
    return {
        "period": period,
        "vintage_a_year": vintage_a_year,
        "vintage_b_year": vintage_b_year,
        "vintage_a_code": vintage_a_code,
        "vintage_b_code": vintage_b_code,
    }


def _trade_row(product, value, partner="BE", flow="1"):
    return {
        "REPORTER": "NL",
        "PARTNER": partner,
        "TRADE_TYPE": "I",
        "PRODUCT_NC": product,
        "FLOW": flow,
        "VALUE_EUR": value,
    }


# LT_REF: Sec3 correspondence + Eq1 integration
def test_run_weight_estimation_integration_deterministic_and_ambiguous(
    tmp_path: Path, monkeypatch
) -> None:
    period = "20002001"
    edges = pd.DataFrame(
        [
            # Ambiguous component (full bipartite 2x2) -> estimated by solver.
            _edge("00000001", "00000011"),
            _edge("00000001", "00000012"),
            _edge("00000002", "00000011"),
            _edge("00000002", "00000012"),
            # Fully deterministic component -> deterministic mapping output.
            _edge("00000003", "00000013"),
        ]
    )

    monkeypatch.setattr(runner, "read_concordance_xls", lambda *_args, **_kwargs: edges)

    annual_dir = tmp_path / "annual"
    annual_dir.mkdir(parents=True, exist_ok=True)

    data_2000 = pd.DataFrame(
        [
            _trade_row("00000001", 70.0, partner="BE"),
            _trade_row("00000002", 30.0, partner="BE"),
            _trade_row("00000001", 20.0, partner="FR"),
            _trade_row("00000002", 80.0, partner="FR"),
            _trade_row("00000003", 40.0, partner="BE"),
        ]
    )
    data_2001 = pd.DataFrame(
        [
            _trade_row("00000011", 48.0, partner="BE"),
            _trade_row("00000012", 52.0, partner="BE"),
            _trade_row("00000011", 28.0, partner="FR"),
            _trade_row("00000012", 72.0, partner="FR"),
            _trade_row("00000013", 40.0, partner="BE"),
        ]
    )

    data_2000.to_parquet(annual_dir / "comext_2000.parquet", index=False)
    data_2001.to_parquet(annual_dir / "comext_2001.parquet", index=False)

    outputs = runner.run_weight_estimation_for_period(
        period=period,
        direction="a_to_b",
        measure="VALUE_EUR",
        concordance_path=tmp_path / "dummy_concordance.xls",
        annual_base_dir=annual_dir,
        output_dir=tmp_path / "out",
        fail_on_status=True,
    )

    assert outputs.weights_path.exists()
    assert outputs.deterministic_path.exists()
    assert outputs.diagnostics_path.exists()
    assert outputs.summary_csv_path is not None and outputs.summary_csv_path.exists()

    # Ambiguous solver output should cover only the 2x2 ambiguous component.
    amb = outputs.weights_ambiguous.sort_values(["from_code", "to_code"]).reset_index(drop=True)
    assert set(amb["from_code"]) == {"00000001", "00000002"}
    assert set(amb["to_code"]) == {"00000011", "00000012"}
    assert len(amb) == 4
    assert (amb["weight"] >= -1e-6).all()
    row_sums = amb.groupby("from_code")["weight"].sum()
    assert np.allclose(row_sums.values, np.ones_like(row_sums.values), atol=1e-6)

    expected = {
        ("00000001", "00000011"): 0.6,
        ("00000001", "00000012"): 0.4,
        ("00000002", "00000011"): 0.2,
        ("00000002", "00000012"): 0.8,
    }
    for row in amb.itertuples(index=False):
        assert np.isclose(row.weight, expected[(row.from_code, row.to_code)], atol=1e-5)

    # Deterministic output should include the separate 1:1 mapping.
    det = outputs.weights_deterministic
    pairs = set(zip(det["from_code"], det["to_code"]))
    assert ("00000003", "00000013") in pairs
    det_row = det[(det["from_code"] == "00000003") & (det["to_code"] == "00000013")].iloc[0]
    assert np.isclose(det_row["weight"], 1.0, atol=1e-12)

    # Summary sanity: one ambiguous group solved, deterministic rows present.
    summary_row = outputs.summary.iloc[0]
    assert int(summary_row["n_groups_total"]) == 1
    assert int(summary_row["n_groups_with_data"]) == 1
    assert int(summary_row["n_groups_solved"]) == 1
    assert int(summary_row["n_weight_rows_ambiguous"]) == 4
    assert int(summary_row["n_weight_rows_deterministic"]) >= 1
