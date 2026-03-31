from pathlib import Path

import pandas as pd

from comext_harmonisation.analysis.chained_link_distribution import runner as cld_runner
from comext_harmonisation.analysis.config import (
    ChainedLinkDistributionConfig,
    ChainedLinkDistributionOutputConfig,
    ChainedLinkDistributionPathsConfig,
    ChainedLinkDistributionPlotConfig,
    ChainedLinkDistributionScopeConfig,
    ChainedLinkDistributionYearsConfig,
)
from comext_harmonisation.concordance.groups import build_concordance_groups


def _make_config(tmp_path: Path, *, scope_mode: str) -> ChainedLinkDistributionConfig:
    output_dir = tmp_path / "out"
    return ChainedLinkDistributionConfig(
        years=ChainedLinkDistributionYearsConfig(
            backward_anchor=2002,
            forward_anchor=2000,
        ),
        paths=ChainedLinkDistributionPathsConfig(
            concordance_path=Path("dummy.xls"),
            concordance_sheet=None,
            annual_base_dir=tmp_path / "annual",
            output_dir=output_dir,
        ),
        scope=ChainedLinkDistributionScopeConfig(mode=scope_mode),
        output=ChainedLinkDistributionOutputConfig(
            summary_csv=output_dir / "summary.csv",
        ),
        plot=ChainedLinkDistributionPlotConfig(
            output_path=output_dir / "fig.png",
            bar_output_path=output_dir / "fig_bars.png",
            title="Thesis figure",
            use_latex=False,
            latex_preamble="",
        ),
    )


def _toy_groups():
    edges = pd.DataFrame(
        {
            "period": ["20002001", "20002001", "20012002", "20012002", "20012002"],
            "vintage_a_year": ["2000", "2000", "2001", "2001", "2001"],
            "vintage_b_year": ["2001", "2001", "2002", "2002", "2002"],
            "vintage_a_code": ["A", "X", "B", "B", "Y"],
            "vintage_b_code": ["B", "Y", "C1", "C2", "Z"],
        }
    )
    return build_concordance_groups(edges)


def test_chained_link_distribution_runner_outputs(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, scope_mode="observed_universe_implied_identities")
    groups = _toy_groups()
    monkeypatch.setattr(cld_runner, "load_concordance_groups", lambda **_: groups)
    monkeypatch.setattr(
        cld_runner,
        "build_code_universe_from_annual",
        lambda **_: {
            2000: {"A", "X", "U"},
            2001: {"B", "Y", "U"},
            2002: {"C1", "C2", "Z", "U"},
        },
    )
    plot_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cld_runner,
        "plot_chained_link_distribution_panels",
        lambda **kwargs: plot_calls.append(kwargs),
    )
    bar_plot_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cld_runner,
        "plot_chained_link_distribution_bar_panels",
        lambda **kwargs: bar_plot_calls.append(kwargs),
    )

    outputs = cld_runner.run_chained_link_distribution_analysis(config)

    assert outputs["output_plot"] == config.plot.output_path
    assert outputs["output_plot_bars"] == config.plot.bar_output_path
    assert outputs["summary_csv"] == config.output.summary_csv
    assert config.output.summary_csv.exists()
    assert len(plot_calls) == 1
    assert len(bar_plot_calls) == 1
    assert plot_calls[0]["output_path"] == config.plot.output_path
    assert bar_plot_calls[0]["output_path"] == config.plot.bar_output_path
    assert plot_calls[0]["title"] == config.plot.title
    assert bar_plot_calls[0]["title"] == config.plot.title

    summary = pd.read_csv(config.output.summary_csv)
    assert set(summary["panel_direction"]) == {"backward", "forward"}
    assert set(summary["scope_mode"]) == {"observed_universe_implied_identities"}

    forward_2001 = summary.loc[
        (summary["panel_direction"] == "forward") & (summary["compare_year"] == 2001)
    ].set_index("relationship")["n_anchor_codes"]
    assert forward_2001.to_dict() == {"1:1": 3, "m:1": 0, "1:n": 0, "m:n": 0}

    forward_2002 = summary.loc[
        (summary["panel_direction"] == "forward") & (summary["compare_year"] == 2002)
    ].set_index("relationship")["n_anchor_codes"]
    assert forward_2002.to_dict() == {"1:1": 2, "m:1": 0, "1:n": 1, "m:n": 0}

    backward_2001 = summary.loc[
        (summary["panel_direction"] == "backward") & (summary["compare_year"] == 2001)
    ].set_index("relationship")["n_anchor_codes"]
    assert backward_2001.to_dict() == {"1:1": 2, "m:1": 2, "1:n": 0, "m:n": 0}

    totals = summary.groupby(["panel_direction", "compare_year"], as_index=False)[
        "n_anchor_codes"
    ].sum()
    assert set(totals["n_anchor_codes"]) == {3, 4}
    assert (
        summary.groupby(["panel_direction", "compare_year"], as_index=False)["share_anchor_codes"]
        .sum()["share_anchor_codes"]
        .round(8)
        .eq(1.0)
        .all()
    )


def test_chained_link_distribution_observed_scope_increases_one_to_one(monkeypatch, tmp_path: Path) -> None:
    groups = _toy_groups()
    monkeypatch.setattr(cld_runner, "load_concordance_groups", lambda **_: groups)
    monkeypatch.setattr(
        cld_runner,
        "build_code_universe_from_annual",
        lambda **_: {
            2000: {"A", "X", "U"},
            2001: {"B", "Y", "U"},
            2002: {"C1", "C2", "Z", "U"},
        },
    )
    monkeypatch.setattr(cld_runner, "plot_chained_link_distribution_panels", lambda **_: None)
    monkeypatch.setattr(cld_runner, "plot_chained_link_distribution_bar_panels", lambda **_: None)

    revised_cfg = _make_config(tmp_path / "revised", scope_mode="revised_only")
    observed_cfg = _make_config(tmp_path / "observed", scope_mode="observed_universe_implied_identities")

    cld_runner.run_chained_link_distribution_analysis(revised_cfg)
    cld_runner.run_chained_link_distribution_analysis(observed_cfg)

    revised = pd.read_csv(revised_cfg.output.summary_csv)
    observed = pd.read_csv(observed_cfg.output.summary_csv)

    revised_one_to_one = int(
        revised.loc[
            (revised["panel_direction"] == "forward")
            & (revised["compare_year"] == 2002)
            & (revised["relationship"] == "1:1"),
            "n_anchor_codes",
        ].iloc[0]
    )
    observed_one_to_one = int(
        observed.loc[
            (observed["panel_direction"] == "forward")
            & (observed["compare_year"] == 2002)
            & (observed["relationship"] == "1:1"),
            "n_anchor_codes",
        ].iloc[0]
    )
    assert observed_one_to_one > revised_one_to_one
