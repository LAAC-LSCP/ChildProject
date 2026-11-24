"""metrics extraction pipelines (ChildProject.pipelines.metrics)"""
import numpy as np
import pandas as pd
import pytest
import yaml

from ChildProject.pipelines.metrics import (
    AclewMetrics,
    CustomMetrics,
    LenaMetrics,
    Metrics,
    MetricsPipeline,
    MetricsSpecificationPipeline,
    pipelines,
)

from conftest import TEST_DATA, TRUTH, import_static


def truth(name):
    return pd.read_csv(TRUTH / name, dtype={"child_id": str})


@pytest.fixture
def its(long_am, lena_segments):
    """valid dataset with LENA-like segments imported for sound.wav as sets lena_its / custom_its"""
    for set_name in ["lena_its", "custom_its"]:
        import_static(long_am, lena_segments, set_name, range_offset=100_000_000, fmt="its")
    return long_am.project


@pytest.fixture
def aclew(long_am, aclew_segments):
    """valid dataset with ACLEW-like segments imported for sound.wav as vtc / alice / vcm sets"""
    for set_name in ["aclew_vtc", "aclew_alice", "aclew_vcm"]:
        import_static(long_am, aclew_segments, set_name, range_offset=4000, fmt="rttm")
    return long_am.project


def metric_list(rows, columns=("callable", "set", "speaker")):
    return pd.DataFrame(np.array(rows, dtype=object), columns=list(columns))


# ============================================================ registry
def test_pipelines_registry():
    assert pipelines == {"custom": CustomMetrics, "lena": LenaMetrics, "aclew": AclewMetrics}


# ============================================================ Metrics: input checks
def test_metrics_unknown_callable(its):
    with pytest.raises(ValueError, match="made_up_function function is not defined"):
        Metrics(its, metric_list([["made_up_function", "custom_its", "CHI"]]))


def test_metrics_callable_must_be_function_or_name(its):
    with pytest.raises(ValueError, match="cannot be evaluated as a metric"):
        Metrics(its, metric_list([[42, "custom_its", "CHI"]]))


def test_metrics_accepts_a_callable(its):
    from ChildProject.pipelines.metricsFunctions import voc_speaker
    metrics = Metrics(its, metric_list([[voc_speaker, "custom_its", "CHI"]])).extract()
    assert "voc_chi" in metrics.columns


def test_metrics_list_must_be_dataframe(its):
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        Metrics(its, "tests/data/list_metrics.csv")


def test_metrics_list_requires_columns(its):
    with pytest.raises(ValueError, match=r"must contain atleast the columns \[callable,set\]"):
        Metrics(its, pd.DataFrame({"callable": ["voc_speaker"]}))


def test_metrics_unknown_set(its):
    with pytest.raises(ValueError, match="annotation set 'unknown' was not found"):
        Metrics(its, metric_list([["voc_speaker", "unknown", "CHI"]]))


def test_metrics_unknown_unit(its):
    with pytest.raises(ValueError, match="<language> is not specified in this dataset"):
        Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"]]), by="language")


def test_metrics_duplicated_names(its):
    with pytest.raises(ValueError, match="the metric name <voc_chi> is used multiple times"):
        Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"], ["voc_speaker", "custom_its", "CHI"]]))


@pytest.mark.parametrize("change,value", [("name", "voc_mal_ph_its"), ("name", "voc_fem_ph_its"), ("speaker", "FEM")])
def test_metrics_list_with_conflicting_names(its, change, value):
    parameters = pd.read_csv(TEST_DATA / "list_metrics.csv")
    parameters.iloc[0, parameters.columns.get_loc(change)] = value
    with pytest.raises(ValueError):
        Metrics(its, parameters)


@pytest.mark.parametrize("kwargs", [{"by": "recording_filename"}, {"period": "1h"}, {"recordings": ["sound.wav"]},
                                    {"from_time": "10:00:00"}, {"to_time": "10:00:00"}])
def test_metrics_segments_option_is_exclusive(its, kwargs):
    with pytest.raises(AssertionError, match="can not be combined"):
        Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"]]), segments="tests/data/segments.csv",
                **dict({"by": "segments"}, **kwargs))


@pytest.mark.parametrize("option,value", [("from_time", "10:00"), ("to_time", "10h")])
def test_metrics_invalid_time(its, option, value):
    with pytest.raises(ValueError, match=f"invalid value for {option}"):
        Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"]]), **{option: value})


def test_metrics_empty_segments(its):
    with pytest.raises(AssertionError, match="should not be empty"):
        Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"]]), by="segments",
                segments=pd.DataFrame(columns=["recording_filename", "segment_onset", "segment_offset"]))


def test_metrics_unknown_requested_columns_are_ignored(its, capsys):
    metrics = Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"]]), rec_cols="nope,date_iso",
                      child_cols="nope").extract()
    out = capsys.readouterr().out
    assert "requested column <nope> does not exist in recordings.csv" in out
    assert "requested column <nope> does not exist in children.csv" in out
    assert "date_iso" in metrics.columns and "nope" not in metrics.columns


# ============================================================ Metrics: extraction
def test_metrics_one_row_per_recording(its):
    metrics = Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"]])).extract()
    assert metrics["recording_filename"].tolist() == ["sound.wav", "video.mp4", "sound2.wav"]


def test_metrics_unannotated_recordings_have_no_value(its):
    metrics = Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"]])).extract().set_index("recording_filename")
    assert metrics.loc["sound.wav", "duration_custom_its"] == 100_000_000
    assert metrics.loc["sound2.wav", "duration_custom_its"] == 0
    assert pd.isna(metrics.loc["sound2.wav", "voc_chi"])


def test_metrics_by_child(its):
    metrics = Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"]]), by="child_id").extract()
    assert metrics["child_id"].tolist() == ["1"]


def test_metrics_restricted_recordings(its):
    metrics = Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"]]), recordings=["sound2.wav"]).extract()
    assert metrics["recording_filename"].tolist() == ["sound2.wav"]


def test_metrics_period_rows(its):
    metrics = Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"]]), period="6h",
                      recordings=["sound.wav"]).extract()
    assert metrics["period_start"].tolist() == ["00:00:00", "06:00:00", "12:00:00", "18:00:00"]
    assert metrics["period_end"].tolist() == ["06:00:00", "12:00:00", "18:00:00", "00:00:00"]


def test_metrics_rec_cols_not_unique_for_unit_are_NA(its):
    metrics = Metrics(its, metric_list([["voc_speaker", "custom_its", "CHI"]]), by="child_id",
                      rec_cols="date_iso,experiment").extract()
    assert pd.isna(metrics.loc[0, "date_iso"])
    assert metrics.loc[0, "experiment"] == "test"


def test_metrics_with_pool(its):
    metrics = Metrics(its, pd.read_csv(TEST_DATA / "list_metrics.csv"), threads=2).extract()
    pd.testing.assert_frame_equal(metrics, truth("custom_metrics.csv"), check_like=True)


def test_metrics_matches_truth(its):
    metrics = Metrics(its, pd.read_csv(TEST_DATA / "list_metrics.csv")).extract()
    pd.testing.assert_frame_equal(metrics, truth("custom_metrics.csv"), check_like=True)


def test_custom_metrics_from_csv(its):
    metrics = CustomMetrics(its, TEST_DATA / "list_metrics.csv").extract()
    pd.testing.assert_frame_equal(metrics, truth("custom_metrics.csv"), check_like=True)


def test_lena_metrics(its):
    metrics = LenaMetrics(its, set="lena_its", period="1h", from_time="10:00:00", to_time="16:00:00").extract()
    pd.testing.assert_frame_equal(metrics, truth("lena_metrics.csv"), check_like=True)


def test_lena_metrics_unknown_set(its):
    with pytest.raises(ValueError, match="annotation set 'unknown' was not found"):
        LenaMetrics(its, set="unknown")


def test_lena_metrics_time_range_without_period(its):
    full = LenaMetrics(its, set="lena_its", recordings=["sound.wav"]).extract()
    ranged = LenaMetrics(its, set="lena_its", recordings=["sound.wav"], from_time="10:00:00", to_time="11:00:00").extract()
    assert ranged.loc[0, "voc_chi_ph"] != full.loc[0, "voc_chi_ph"]


def test_lena_metrics_period_outside_time_range_is_empty(its):
    metrics = LenaMetrics(its, set="lena_its", period="12h", from_time="10:00:00", to_time="11:00:00",
                          recordings=["sound.wav"]).extract()
    assert metrics.set_index("period_start").loc["12:00:00", "duration_lena_its"] == 0


def test_aclew_metrics(aclew):
    metrics = AclewMetrics(aclew, by="child_id", rec_cols="date_iso", child_cols="experiment,child_dob",
                           vtc="aclew_vtc", alice="aclew_alice", vcm="aclew_vcm").extract()
    pd.testing.assert_frame_equal(metrics, truth("aclew_metrics.csv"), check_like=True)


def test_aclew_metrics_unknown_vtc_set(aclew):
    with pytest.raises(ValueError):
        AclewMetrics(aclew, vtc="unknown")


def test_aclew_metrics_without_alice_and_vcm(aclew, capsys):
    metrics = AclewMetrics(aclew, vtc="aclew_vtc", alice="none", vcm="none").extract()
    out = capsys.readouterr().out
    assert "The ALICE set ('none') was not found" in out and "The vcm set ('none') was not found" in out
    assert "wc_adu_ph" not in metrics.columns and "lp_n" not in metrics.columns


def test_aclew_metrics_include_std(aclew):
    metrics = AclewMetrics(aclew, vtc="aclew_vtc", alice="aclew_alice", vcm="aclew_vcm", include_std=True).extract()
    for column in ["std_voc_dur_chi", "std_cry_voc_dur_chi", "std_can_voc_dur_chi", "std_non_can_voc_dur_chi"]:
        assert column in metrics.columns


def test_metrics_by_segments(aclew):
    metrics = Metrics(aclew, metric_list([
        ["voc_speaker", "aclew_vtc", "FEM"], ["voc_speaker", "aclew_vtc", "CHI"], ["voc_speaker_ph", "aclew_vtc", "FEM"],
        ["voc_speaker_ph", "aclew_vtc", "CHI"], ["wc_speaker_ph", "aclew_alice", "FEM"], ["lp_n", "aclew_vcm", pd.NA],
        ["lp_dur", "aclew_vcm", pd.NA]]), by="segments", rec_cols="date_iso", child_cols="experiment,child_dob",
        segments=TEST_DATA / "segments.csv").extract()
    expected = truth("segments_metrics.csv")
    expected.columns = [c.replace("segments_", "aclew_") for c in expected.columns]
    pd.testing.assert_frame_equal(metrics, expected, check_like=True)


def test_metrics_by_segments_dataframe(aclew):
    segments = pd.read_csv(TEST_DATA / "segments.csv")
    metrics = Metrics(aclew, metric_list([["voc_speaker", "aclew_vtc", "CHI"]]), by="segments", segments=segments).extract()
    assert metrics[["segment_onset", "segment_offset"]].values.tolist() == [[0, 1000], [1500, 4000]]


def test_metrics_peaks(aclew):
    metrics = Metrics(aclew, metric_list([
        ["voc_speaker", "aclew_vtc", "FEM", pd.NA], ["voc_speaker", "aclew_vtc", "CHI", pd.NA],
        ["peak_voc_speaker", "aclew_vtc", "FEM", 1000], ["peak_voc_speaker", "aclew_vtc", "CHI", 3600000],
        ["voc_speaker_ph", "aclew_vtc", "FEM", pd.NA], ["voc_speaker_ph", "aclew_vtc", "CHI", pd.NA],
        ["wc_adu_ph", "aclew_alice", pd.NA, pd.NA], ["peak_wc_adu", "aclew_alice", pd.NA, 1000],
        ["simple_CTC", "aclew_vtc", pd.NA, pd.NA], ["peak_simple_CTC", "aclew_vtc", pd.NA, 1000]],
        columns=("callable", "set", "speaker", "period_time"))).extract()
    expected = truth("peak_metrics.csv")
    expected.columns = [c.replace("peak_vtc", "aclew_vtc").replace("peak_alice", "aclew_alice") for c in expected.columns]
    pd.testing.assert_frame_equal(metrics, expected, check_like=True)


# ============================================================ MetricsPipeline
def test_metrics_pipeline_writes_metrics(its, tmp_path):
    output = tmp_path / "pipeline_metrics.csv"
    MetricsPipeline().run(its.path, output, "lena", set="lena_its", by="recording_filename")
    expected = truth("pipeline_metrics.csv")
    expected.columns = [c.replace("pipe_its", "lena_its") for c in expected.columns]
    pd.testing.assert_frame_equal(pd.read_csv(output, dtype={"child_id": str}), expected, check_like=True)


def test_metrics_pipeline_writes_parameters(its, tmp_path, capsys):
    pipeline = MetricsPipeline()
    pipeline.run(its.path, tmp_path / "m.csv", "lena", set="lena_its")
    parameters = yaml.safe_load(open(pipeline.parameters_path))
    assert parameters["parameters"]["pipeline"] == "lena"
    assert parameters["parameters"]["set"] == "lena_its"
    assert {"callable": "voc_speaker_ph", "set": "lena_its", "speaker": "FEM"} in parameters["parameters"]["metrics_list"]
    assert "Your dataset is not currently a git repository" in capsys.readouterr().out


def test_metrics_pipeline_unknown_pipeline(its, tmp_path):
    with pytest.raises(NotImplementedError, match="invalid pipeline 'nope'"):
        MetricsPipeline().run(its.path, tmp_path / "m.csv", "nope")


# ============================================================ MetricsSpecificationPipeline
def write_spec(tmp_path, project, **overrides):
    spec = yaml.safe_load((TEST_DATA / "parameters_metrics.yml").read_text())
    spec.update({"path": str(project.path), "destination": str(tmp_path / "specs_metrics.csv")})
    for metric in spec["metrics_list"]:
        metric["set"] = "custom_its"
    spec.update(overrides)
    path = tmp_path / "spec.yml"
    path.write_text(yaml.dump(spec))
    return path


def specs_truth():
    expected = truth("specs_metrics.csv")
    expected.columns = [c.replace("specs_its", "custom_its") for c in expected.columns]
    return expected


def test_specification_pipeline(its, tmp_path):
    pipeline = MetricsSpecificationPipeline()
    pipeline.run(write_spec(tmp_path, its))
    pd.testing.assert_frame_equal(pd.read_csv(pipeline.destination, dtype={"child_id": str}), specs_truth(), check_like=True)


def test_specification_pipeline_rerun_from_exported_parameters(its, tmp_path):
    pipeline = MetricsSpecificationPipeline()
    pipeline.run(write_spec(tmp_path, its))
    pipeline.run(pipeline.parameters_path)
    pd.testing.assert_frame_equal(pd.read_csv(pipeline.destination, dtype={"child_id": str}), specs_truth(), check_like=True)


@pytest.mark.parametrize("missing", ["path", "destination", "metrics_list"])
def test_specification_pipeline_required_keys(its, tmp_path, missing):
    spec = write_spec(tmp_path, its)
    content = yaml.safe_load(spec.read_text())
    del content[missing]
    spec.write_text(yaml.dump(content))
    with pytest.raises(ValueError, match=f"must contain .*'{missing}' key"):
        MetricsSpecificationPipeline().run(spec)


def test_specification_pipeline_empty_file(tmp_path):
    (tmp_path / "empty.yml").write_text("")
    with pytest.raises(ValueError, match="could not find any parameters"):
        MetricsSpecificationPipeline().run(tmp_path / "empty.yml")


def test_specification_pipeline_invalid_yaml(tmp_path):
    (tmp_path / "bad.yml").write_text("a: [b\n")
    with pytest.raises(yaml.YAMLError, match="parsing of the parameters file"):
        MetricsSpecificationPipeline().run(tmp_path / "bad.yml")


def test_specification_pipeline_metrics_list_must_be_a_list(its, tmp_path):
    with pytest.raises(ValueError, match="must be a list of elements"):
        MetricsSpecificationPipeline().run(write_spec(tmp_path, its, metrics_list=5))


def test_specification_pipeline_unknown_parameter(its, tmp_path):
    with pytest.raises(ValueError, match="Unrecognized parameter found"):
        MetricsSpecificationPipeline().run(write_spec(tmp_path, its, colour="blue"))


def test_specification_pipeline_ignores_pipeline_key(its, tmp_path):
    pipeline = MetricsSpecificationPipeline()
    pipeline.run(write_spec(tmp_path, its, pipeline="lena"))
    assert pipeline.metrics.shape[0] == 12  # 1 child x 12 periods of 2h
