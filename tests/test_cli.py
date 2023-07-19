import subprocess


def cli(cmd):
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    exit_code = process.wait()
    return stdout, stderr, exit_code


def test_validate():
    stdout, stderr, exit_code = cli(
        ["child-project", "validate", "examples/valid_raw_data"]
    )
    assert exit_code == 0


def test_overview():
    stdout, stderr, exit_code = cli(
        ["child-project", "overview", "examples/valid_raw_data"]
    )
    assert exit_code == 0


def test_import_annotations():
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "import-annotations",
            "examples/valid_raw_data",
            "--annotations",
            "examples/valid_raw_data/annotations/input_short.csv",
        ]
    )
    assert exit_code == 0

def test_compute_durations():
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "compute-durations",
            "examples/valid_raw_data"
        ]
    )
    assert exit_code == 0

def test_explain():
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "explain",
            "examples/valid_raw_data",
            "notes"
        ]
    )
    assert exit_code == 0

    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "explain",
            "examples/valid_raw_data",
            "non-existent-variable"
        ]
    )
    assert exit_code == 0
    
def test_compare_recordings():
    stdout, stderr, exit_code = cli(
        [
            "child-project",
            "compare-recordings",
            "examples/valid_raw_data",
            "sound.wav",
            "sound2.wav",
            "--interval",
            "10"
        ]
    )
    assert exit_code == 0