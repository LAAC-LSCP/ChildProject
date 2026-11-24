"""ChildProject.pipelines.anonymize"""
import json
import re

import pytest

from ChildProject.pipelines.anonymize import AnonymizationPipeline

RAW = "annotations/{}/raw"


@pytest.fixture
def anonymized(project):
    AnonymizationPipeline().run(project.path, "new_its", "new_its_anonymized")
    return (project.path / RAW.format("new_its_anonymized") / "example_lena_new.its").read_text()


def write_its(project, set_name, content, name="file.its"):
    folder = project.path / RAW.format(set_name)
    (folder / name).parent.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(content)
    return folder / name


def test_output_file_is_created_for_every_input_file(project):
    write_its(project, "multi", "<ITS fileName=\"a\">\n", "sub/a.its")
    write_its(project, "multi", "<ITS fileName=\"b\">\n", "b.its")
    AnonymizationPipeline().run(project.path, "multi", "multi_anon")
    out = project.path / RAW.format("multi_anon")
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.its")) == ["b.its", "sub/a.its"]


def test_files_without_extension_are_anonymized_too(project):
    write_its(project, "noext", "<ITS fileName=\"secret\">\n", "file")
    AnonymizationPipeline().run(project.path, "noext", "noext_anon")
    assert (project.path / RAW.format("noext_anon") / "file").exists()


@pytest.mark.parametrize("pattern", [
    r'<PrimaryChild [^>]*DOB="1000-01-01"',
    r'<ChildInfo [^>]*dob="1000-01-01"',
    r'<ITS fileName="new_filename_1001"',
])
def test_identifying_values_are_replaced(anonymized, pattern):
    assert re.search(pattern, anonymized)


def test_dates_are_replaced_but_times_are_kept(anonymized):
    assert 'startClockTime="1000-01-01T13:12:36Z"' in anonymized
    assert "2019-12-08" not in anonymized


def test_original_identifiers_are_gone(anonymized):
    assert "2018-10-24" not in anonymized
    assert "20191220_123229_024880" not in anonymized


def test_non_identifying_content_is_kept(anonymized, project):
    original = (project.path / RAW.format("new_its") / "example_lena_new.its").read_text()
    assert anonymized.count("\n") == original.count("\n")
    assert 'Gender="F"' in anonymized


def test_input_set_is_not_modified(project):
    path = project.path / RAW.format("new_its") / "example_lena_new.its"
    before = path.read_text()
    AnonymizationPipeline().run(project.path, "new_its", "anon")
    assert path.read_text() == before


def test_same_input_and_output_set(project):
    with pytest.raises(Exception, match="input_set and output_set should not be equal"):
        AnonymizationPipeline().run(project.path, "new_its", "new_its")


def test_existing_destination(project):
    with pytest.raises(Exception, match="destination .* already exists"):
        AnonymizationPipeline().run(project.path, "new_its", "old_its")


def test_custom_replacements(project, tmp_path):
    write_its(project, "custom", '<Child id="C123" DOB="2018-01-01" Gender="F"/>\n')
    config = tmp_path / "replacements.json"
    config.write_text(json.dumps({"Child": {"id": "ANON"}}))
    AnonymizationPipeline().run(project.path, "custom", "custom_anon", replacements_json_dict=str(config))
    content = (project.path / RAW.format("custom_anon") / "file.its").read_text()
    assert content == '<Child id="ANON" DOB="2018-01-01" Gender="F"/>\n'


def test_custom_replacement_full_value_when_only_time_is_false(project, tmp_path):
    write_its(project, "custom", '<Bar startClockTime="2019-12-08T13:12:36Z"/>\n')
    config = tmp_path / "replacements.json"
    config.write_text(json.dumps({"Bar": {"startClockTime": [{"replace_value": "X"}, {"only_time": "false"}]}}))
    AnonymizationPipeline().run(project.path, "custom", "custom_anon", replacements_json_dict=str(config))
    assert (project.path / RAW.format("custom_anon") / "file.its").read_text() == '<Bar startClockTime="X"/>\n'


def test_node_names_match_on_word_boundary(project):
    write_its(project, "boundary", '<ChildInfoExtra dob="2018-10-24"/>\n')
    AnonymizationPipeline().run(project.path, "boundary", "boundary_anon")
    assert "2018-10-24" in (project.path / RAW.format("boundary_anon") / "file.its").read_text()


def test_setup_parser():
    import argparse
    parser = argparse.ArgumentParser()
    AnonymizationPipeline.setup_parser(parser)
    args = parser.parse_args(["path", "--input-set", "a", "--output-set", "b"])
    assert (args.path, args.input_set, args.output_set, args.replacements_json_dict) == ("path", "a", "b", "")
