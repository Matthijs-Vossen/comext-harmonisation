from pathlib import Path

import pandas as pd

from comext_harmonisation.analysis.config import (
    LinkDistributionConfig,
    LinkDistributionOutputConfig,
    LinkDistributionPathsConfig,
    LinkDistributionScopeConfig,
)
from comext_harmonisation.analysis.link_distribution import runner as ld_runner
from comext_harmonisation.concordance.groups import build_concordance_groups


def _make_config(tmp_path: Path, *, scope_mode: str = "revised_only") -> LinkDistributionConfig:
    output_dir = tmp_path / "outputs"
    return LinkDistributionConfig(
        paths=LinkDistributionPathsConfig(
            concordance_path=Path("dummy.xls"),
            concordance_sheet=None,
            annual_base_dir=tmp_path / "annual",
            output_dir=output_dir,
        ),
        scope=LinkDistributionScopeConfig(mode=scope_mode),
        output=LinkDistributionOutputConfig(
            summary_csv=output_dir / "summary.csv",
            focal_codes_csv=output_dir / "focal_codes.csv",
        ),
    )


def test_link_distribution_runner_writes_expected_outputs(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    edges = pd.DataFrame(
        {
            "period": ["20062007"] * 3 + ["20072008"] * 3,
            "vintage_a_year": ["2006"] * 3 + ["2007"] * 3,
            "vintage_b_year": ["2007"] * 3 + ["2008"] * 3,
            "vintage_a_code": ["A1", "A1", "X1", "B1", "B2", "Y1"],
            "vintage_b_code": ["B1", "B2", "Y1", "C1", "C2", "Z1"],
        }
    )
    groups = build_concordance_groups(edges)
    monkeypatch.setattr(ld_runner, "load_concordance_groups", lambda **_: groups)

    outputs = ld_runner.run_link_distribution_analysis(config)

    assert Path(outputs["summary_csv"]).exists()
    assert Path(outputs["focal_codes_csv"]).exists()

    summary = pd.read_csv(outputs["summary_csv"])
    focal_codes = pd.read_csv(outputs["focal_codes_csv"])

    assert set(summary["scope_label"]) == {"revised_only"}
    assert set(focal_codes["scope_label"]) == {"revised_only"}
    assert set(summary["direction"]) == {"a_to_b", "b_to_a"}
    assert set(focal_codes["direction"]) == {"a_to_b", "b_to_a"}
    assert set(summary["period"].astype(str)) == {"20062007", "20072008"}

    summary_2006 = summary.loc[
        (summary["period"] == 20062007)
        & (summary["focal_side"] == "vintage_a")
    ].set_index("relationship")["n_focal_codes"]
    assert summary_2006.to_dict() == {"1:1": 1, "1:n": 1}
    unknown_2006 = summary.loc[
        (summary["period"] == 20062007)
        & (summary["focal_side"] == "vintage_a")
    ].set_index("relationship")["unknown_conversion_weight"]
    assert unknown_2006.to_dict() == {"1:1": False, "1:n": True}

    summary_2007 = summary.loc[
        (summary["period"] == 20062007)
        & (summary["focal_side"] == "vintage_b")
    ].set_index("relationship")["n_focal_codes"]
    assert summary_2007.to_dict() == {"1:1": 1, "m:1": 2}
    unknown_2007 = summary.loc[
        (summary["period"] == 20062007)
        & (summary["focal_side"] == "vintage_b")
    ].set_index("relationship")["unknown_conversion_weight"]
    assert unknown_2007.to_dict() == {"1:1": False, "m:1": True}

    totals = summary.groupby(["period", "focal_side"], as_index=False)["n_focal_codes"].sum()
    expected_totals = (
        focal_codes.groupby(["period", "focal_side"], as_index=False)["focal_code"]
        .count()
        .rename(columns={"focal_code": "expected"})
    )
    merged = totals.merge(expected_totals, on=["period", "focal_side"], how="inner")
    assert merged["n_focal_codes"].tolist() == merged["expected"].tolist()


def test_link_distribution_runner_rejects_non_adjacent_period(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    edges = pd.DataFrame(
        {
            "period": ["20062008"],
            "vintage_a_year": ["2006"],
            "vintage_b_year": ["2008"],
            "vintage_a_code": ["A1"],
            "vintage_b_code": ["B1"],
        }
    )
    groups = build_concordance_groups(edges)
    monkeypatch.setattr(ld_runner, "load_concordance_groups", lambda **_: groups)

    try:
        ld_runner.run_link_distribution_analysis(config)
    except ValueError as exc:
        assert "adjacent annual breaks" in str(exc)
    else:
        raise AssertionError("Expected non-adjacent periods to be rejected")


def test_link_distribution_runner_adds_observed_identities(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, scope_mode="observed_universe_implied_identities")
    edges = pd.DataFrame(
        {
            "period": ["20062007"] * 3,
            "vintage_a_year": ["2006"] * 3,
            "vintage_b_year": ["2007"] * 3,
            "vintage_a_code": ["A1", "A1", "X1"],
            "vintage_b_code": ["B1", "B2", "Y1"],
        }
    )
    groups = build_concordance_groups(edges)
    monkeypatch.setattr(ld_runner, "load_concordance_groups", lambda **_: groups)
    monkeypatch.setattr(
        ld_runner,
        "_available_periods_for_observed_universe",
        lambda *_args, **_kwargs: {"20062007"},
    )
    monkeypatch.setattr(
        ld_runner,
        "build_code_universe_from_annual",
        lambda **_: {2006: {"A1", "X1", "U1"}, 2007: {"B1", "B2", "Y1", "U1"}},
    )

    outputs = ld_runner.run_link_distribution_analysis(config)
    summary = pd.read_csv(outputs["summary_csv"])
    focal_codes = pd.read_csv(outputs["focal_codes_csv"])

    assert set(summary["scope_label"]) == {"observed_universe_implied_identities"}
    assert set(focal_codes["scope_label"]) == {"observed_universe_implied_identities"}

    summary_2006 = summary.loc[
        (summary["period"] == 20062007)
        & (summary["focal_side"] == "vintage_a")
    ].set_index("relationship")["n_focal_codes"]
    assert summary_2006.to_dict() == {"1:1": 2, "1:n": 1}

    summary_2007 = summary.loc[
        (summary["period"] == 20062007)
        & (summary["focal_side"] == "vintage_b")
    ].set_index("relationship")["n_focal_codes"]
    assert summary_2007.to_dict() == {"1:1": 2, "m:1": 2}

    identity_rows = focal_codes.loc[focal_codes["group_id"].astype(str).str.contains("_identity_")]
    assert set(identity_rows["focal_code"]) == {"U1"}
