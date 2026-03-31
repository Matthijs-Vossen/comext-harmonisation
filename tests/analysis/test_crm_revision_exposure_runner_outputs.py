from pathlib import Path

import pandas as pd

from comext_harmonisation.analysis.config import (
    CrmRevisionExposureConfig,
    CrmRevisionExposureOutputConfig,
    CrmRevisionExposurePathsConfig,
    CrmRevisionExposurePlotConfig,
    CrmRevisionExposureScopeConfig,
    CrmRevisionExposureYearsConfig,
)
from comext_harmonisation.analysis.crm_revision_exposure import runner as cre_runner
from comext_harmonisation.concordance.groups import build_concordance_groups


def _make_config(tmp_path: Path, crm_codes_path: Path) -> CrmRevisionExposureConfig:
    output_dir = tmp_path / "out"
    return CrmRevisionExposureConfig(
        years=CrmRevisionExposureYearsConfig(
            anchor_year=2001,
            backward_end_year=2000,
            forward_end_year=2003,
            benchmark_backward_years=(2000,),
            benchmark_forward_years=(2002, 2003),
        ),
        paths=CrmRevisionExposurePathsConfig(
            concordance_path=Path("dummy.xls"),
            concordance_sheet=None,
            annual_base_dir=tmp_path / "annual",
            crm_codes_path=crm_codes_path,
            output_dir=output_dir,
        ),
        scope=CrmRevisionExposureScopeConfig(mode="observed_universe_implied_identities"),
        output=CrmRevisionExposureOutputConfig(
            summary_csv=output_dir / "summary.csv",
            code_exposure_csv=output_dir / "code_exposure.csv",
            benchmark_summary_csv=output_dir / "benchmark_summary.csv",
        ),
        plot=CrmRevisionExposurePlotConfig(
            output_path=output_dir / "fig.png",
            title="CRM exposure",
            use_latex=False,
            latex_preamble="",
        ),
    )


def _toy_groups():
    edges = pd.DataFrame(
        {
            "period": [
                "20002001",
                "20002001",
                "20012002",
                "20012002",
                "20012002",
                "20022003",
                "20022003",
                "20022003",
            ],
            "vintage_a_year": [
                "2000",
                "2000",
                "2001",
                "2001",
                "2001",
                "2002",
                "2002",
                "2002",
            ],
            "vintage_b_year": [
                "2001",
                "2001",
                "2002",
                "2002",
                "2002",
                "2003",
                "2003",
                "2003",
            ],
            "vintage_a_code": ["A", "X", "B", "B", "Y", "C1", "C2", "Z"],
            "vintage_b_code": ["B", "Y", "C1", "C2", "Z", "D1", "D2", "W"],
        }
    )
    return build_concordance_groups(edges)


def test_crm_revision_exposure_runner_outputs(monkeypatch, tmp_path: Path) -> None:
    crm_codes_path = tmp_path / "crm_codes.csv"
    crm_codes_path.write_text(
        "\n".join(
            [
                "raw_material,cn_code_2023",
                "A,B",
                "B,B",
                "C,U",
            ]
        )
    )
    config = _make_config(tmp_path, crm_codes_path)
    groups = _toy_groups()

    monkeypatch.setattr(cre_runner, "load_concordance_groups", lambda **_: groups)
    monkeypatch.setattr(
        cre_runner,
        "build_code_universe_from_annual",
        lambda **_: {
            2000: {"A", "X", "U"},
            2001: {"B", "Y", "U"},
            2002: {"C1", "C2", "Z", "U"},
            2003: {"D1", "D2", "W", "U"},
        },
    )
    plot_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cre_runner,
        "plot_crm_revision_exposure_panels",
        lambda **kwargs: plot_calls.append(kwargs),
    )

    outputs = cre_runner.run_crm_revision_exposure_analysis(config)

    assert outputs["output_plot"] == config.plot.output_path
    assert outputs["summary_csv"] == config.output.summary_csv
    assert outputs["code_exposure_csv"] == config.output.code_exposure_csv
    assert outputs["benchmark_summary_csv"] == config.output.benchmark_summary_csv
    assert len(plot_calls) == 1
    assert plot_calls[0]["output_path"] == config.plot.output_path

    summary = pd.read_csv(config.output.summary_csv)
    code_exposure = pd.read_csv(config.output.code_exposure_csv)
    benchmark = pd.read_csv(config.output.benchmark_summary_csv)

    assert set(summary["panel_direction"]) == {"backward", "forward"}
    assert set(summary["population"]) == {"all_anchor_codes", "crm_anchor_codes"}
    assert set(summary["metric"]) == {
        "remained_strict_1_to_1",
        "ever_non_1_to_1_step",
        "ever_unknown_weight_step",
    }

    forward_2002_all = summary.loc[
        (summary["panel_direction"] == "forward")
        & (summary["compare_year"] == 2002)
        & (summary["population"] == "all_anchor_codes")
        & (summary["metric"] == "ever_non_1_to_1_step")
    ].iloc[0]
    assert int(forward_2002_all["n_codes"]) == 1
    assert int(forward_2002_all["total_codes"]) == 3
    assert float(forward_2002_all["share_codes"]) == 1 / 3

    forward_2003_crm_unknown = summary.loc[
        (summary["panel_direction"] == "forward")
        & (summary["compare_year"] == 2003)
        & (summary["population"] == "crm_anchor_codes")
        & (summary["metric"] == "ever_unknown_weight_step")
    ].iloc[0]
    assert int(forward_2003_crm_unknown["n_codes"]) == 1
    assert int(forward_2003_crm_unknown["total_codes"]) == 2
    assert float(forward_2003_crm_unknown["share_codes"]) == 0.5

    backward_2000_all = summary.loc[
        (summary["panel_direction"] == "backward")
        & (summary["compare_year"] == 2000)
        & (summary["population"] == "all_anchor_codes")
        & (summary["metric"] == "remained_strict_1_to_1")
    ].iloc[0]
    assert int(backward_2000_all["n_codes"]) == 3
    assert int(backward_2000_all["total_codes"]) == 3
    assert float(backward_2000_all["share_codes"]) == 1.0

    code_b_2002 = code_exposure.loc[
        (code_exposure["panel_direction"] == "forward")
        & (code_exposure["compare_year"] == 2002)
        & (code_exposure["anchor_code"] == "B")
    ].iloc[0]
    assert bool(code_b_2002["is_crm_code"]) is True
    assert bool(code_b_2002["touched_non_1_to_1_this_step"]) is True
    assert bool(code_b_2002["ever_unknown_weight_step"]) is True
    assert code_b_2002["final_relationship"] == "1:n"

    code_b_2003 = code_exposure.loc[
        (code_exposure["panel_direction"] == "forward")
        & (code_exposure["compare_year"] == 2003)
        & (code_exposure["anchor_code"] == "B")
    ].iloc[0]
    assert bool(code_b_2003["touched_non_1_to_1_this_step"]) is False
    assert bool(code_b_2003["ever_non_1_to_1_step"]) is True
    assert code_b_2003["final_relationship"] == "1:n"

    crm_count = int(
        code_exposure.loc[
            (code_exposure["panel_direction"] == "forward")
            & (code_exposure["compare_year"] == 2002),
            "is_crm_code",
        ].sum()
    )
    assert crm_count == 2

    assert set(benchmark["compare_year"]) == {2000, 2002, 2003}
