import pandas as pd

from comext_harmonisation.analysis.synthetic_persistence.runner import (
    CandidateTiming,
    _classify_candidate_status,
    _compute_code_evidence,
    _split_regime_segments_for_plot,
)


def test_code_evidence_computes_descriptive_metrics() -> None:
    rows = []
    for year in range(1, 7):
        is_synth = year <= 3
        rows.append(
            {
                "dimension": "prehistory",
                "set_name": "prehistory",
                "code": "85171300",
                "year": year,
                "share_conv": 0.01 if is_synth else 0.10,
                "is_synthetic_window": is_synth,
                "is_inlife_window": not is_synth,
            }
        )
    df = pd.DataFrame(rows)

    evidence = _compute_code_evidence(df)
    assert len(evidence) == 1
    row = evidence.iloc[0]
    assert row["synthetic_years"] == 3
    assert row["inlife_years"] == 3
    assert row["synthetic_peak_share"] == 0.01
    assert row["inlife_peak_share"] == 0.10
    assert row["peak_year"] == 4


def test_code_evidence_handles_zero_inlife_denominator() -> None:
    rows = []
    for year in range(1, 7):
        is_synth = year <= 3
        rows.append(
            {
                "dimension": "afterlife",
                "set_name": "afterlife",
                "code": "85281011",
                "year": year,
                "share_conv": 0.02 if is_synth else 0.0,
                "is_synthetic_window": is_synth,
                "is_inlife_window": not is_synth,
            }
        )
    df = pd.DataFrame(rows)

    evidence = _compute_code_evidence(df)
    assert len(evidence) == 1
    assert pd.isna(evidence.iloc[0]["cumulative_ratio_synth_to_inlife"])


def test_split_regime_segments_prehistory_bridges_boundary() -> None:
    frame = pd.DataFrame(
        [
            {"year": 1988, "share_conv": 0.001, "is_synthetic_window": True},
            {"year": 1989, "share_conv": 0.002, "is_synthetic_window": True},
            {"year": 1990, "share_conv": 0.004, "is_synthetic_window": False},
            {"year": 1991, "share_conv": 0.005, "is_synthetic_window": False},
        ]
    )
    observed, synthetic = _split_regime_segments_for_plot(frame)

    assert observed["year"].to_list() == [1989, 1990, 1991]
    assert synthetic["year"].to_list() == [1988, 1989, 1990]


def test_split_regime_segments_afterlife_bridges_boundary() -> None:
    frame = pd.DataFrame(
        [
            {"year": 1988, "share_conv": 0.006, "is_synthetic_window": False},
            {"year": 1989, "share_conv": 0.004, "is_synthetic_window": False},
            {"year": 1990, "share_conv": 0.002, "is_synthetic_window": True},
            {"year": 1991, "share_conv": 0.001, "is_synthetic_window": True},
        ]
    )
    observed, synthetic = _split_regime_segments_for_plot(frame)

    assert observed["year"].to_list() == [1988, 1989, 1990]
    assert synthetic["year"].to_list() == [1989, 1990, 1991]


def test_classify_candidate_status_outside_window_and_concept_flag() -> None:
    outside = CandidateTiming(
        obs_first_year=None,
        obs_last_year=None,
        concordance_intro_year=2025,
        concordance_sunset_year=None,
    )
    included, reason = _classify_candidate_status(
        dimension="afterlife",
        timing=outside,
        start_year=1988,
        end_year=2023,
    )
    assert included is False
    assert reason == "introduced_outside_window"

    concept_not_afterlife = CandidateTiming(
        obs_first_year=2011,
        obs_last_year=2023,
        concordance_intro_year=2011,
        concordance_sunset_year=None,
    )
    included, reason = _classify_candidate_status(
        dimension="afterlife",
        timing=concept_not_afterlife,
        start_year=1988,
        end_year=2023,
    )
    assert included is False
    assert reason == "concept_not_afterlife"
