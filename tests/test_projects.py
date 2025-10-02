from ChildProject.projects import ChildProject
import pandas as pd
import pytest
import shutil
import os
import filecmp
from pathlib import Path

TEST_DIR = os.path.join("output", "projects")
PATH = Path(TEST_DIR)

@pytest.fixture(scope="function")
def project(request):
    if os.path.exists(TEST_DIR):
#        shutil.copytree(src="examples/valid_raw_data", dst="output/annotations")
        shutil.rmtree(TEST_DIR)
    shutil.copytree(src="examples/valid_raw_data", dst=TEST_DIR, symlinks=True)
    
    project = ChildProject(TEST_DIR)
    yield project

def test_enforce_dtypes():
    project = ChildProject("examples/valid_raw_data", enforce_dtypes=True)
    project.read()

    assert project.recordings["child_id"].dtype.kind == "O"
    assert project.children["child_id"].dtype.kind == "O"

    project = ChildProject("examples/valid_raw_data", enforce_dtypes=False)
    project.read()

    assert project.recordings["child_id"].dtype.kind == "i"
    assert project.children["child_id"].dtype.kind == "i"
    
@pytest.mark.parametrize("idis,rshape,cshape,drshape,dcshape",
                         [(True,2,1,1,1),
                         (False,3,2,0,0),
                         ])
def test_ignore_discarded(idis, rshape, cshape, drshape, dcshape):
    project = ChildProject("examples/valid_raw_data", ignore_discarded=idis)
    project.read()
    
    assert project.recordings.shape[0] == rshape
    assert project.discarded_recordings.shape[0] == drshape
    assert project.children.shape[0] == cshape
    assert project.discarded_children.shape[0] == dcshape


@pytest.mark.parametrize("old,new,error,exists,indexed",
                         [('sound.wav','new_sound.wav', None, True, True),
                          ('sound.wav','sound2.wav', AssertionError, True, True),
                          ('soundx.wav','new_sound.wav', None, False, True),
                          ('soundx.wav','new_sound.wav', None, False, False),
                         ])
def test_rename_recording(project, old, new, error, exists, indexed):
    project.read()
    if indexed:
        if old not in project.recordings['recording_filename'].unique():
            project.recordings = pd.concat([project.recordings, pd.DataFrame({'recording_filename':[old],
                                                                      'experiment':['test'],
                                                                      'start_time':['9:00'],
                                                                      'date_iso':['2020-01-01'],
                                                                      'child_id': ['1'],
                                                                      'recording_device_type':['usb'],
                                                                      })])
        series = project.recordings.set_index('recording_filename', drop=True).loc[old].copy()
    if error:
        with pytest.raises(error):
            project.rename_recording(old, new)

    else:
        project.rename_recording(old, new)

        if indexed:
            new_series = project.recordings.set_index('recording_filename', drop=True).loc[new].copy()
            pd.testing.assert_series_equal(series, new_series, check_names=False)
        assert not project.get_recording_path(old).exists()
        if exists:
            assert project.get_recording_path(new).exists()


def test_compute_ages():
    project = ChildProject("examples/valid_raw_data")
    project.read()

    project.recordings["age"] = project.compute_ages()
    project.recordings["age_days"] = project.compute_ages(age_format='days')
    project.recordings["age_weeks"] = project.compute_ages(age_format='weeks')
    project.recordings["age_years"] = project.compute_ages(age_format='years')

    truth = pd.read_csv("tests/truth/ages.csv", dtype={'child_id': 'string'}, dtype_backend='numpy_nullable',
                        ).set_index("line")

    pd.testing.assert_frame_equal(
        project.recordings[["child_id", "age", "age_days", "age_weeks", "age_years"]],
        truth[["child_id", "age", "age_days", "age_weeks", "age_years"]],
        check_like=True, check_index_type=False, check_dtype=False
    )

@pytest.mark.parametrize("error,chi_lines,rec_lines", 
                         [(None,[],[]),
                         (None,['test2,3,2018-02-02,0'],[]),
                         ])
def test_projects_read(project, error, chi_lines, rec_lines):
    
    if chi_lines:
        chi_path = os.path.join(TEST_DIR, "metadata","children.csv")
        with open(chi_path, "a") as f:
            for l in chi_lines:
                f.write(str(l) + '\n')
    if rec_lines:
        rec_path = os.path.join(TEST_DIR, "metadata", "recordings.csv")
        with open(rec_path, "a") as f:
            for l in chi_lines:
                f.write(str(l) + '\n')
    if error:
        with pytest.raises(error):
            project.read()
    else:
        project.read()


@pytest.mark.parametrize("new_recs,keep_discarded,skip_validation,error",
                         [(pd.DataFrame({'experiment':['test_xp','test_xp'], 'child_id':['C01','C02'], 'recording_filename':['rec1.wav','rec2.wav'], 'start_time':['NA','05:32'], 'date_iso':['2021-03-05','2023-11-29'],
                                        'recording_device_type':['usb','unknown']}).convert_dtypes(), False, False, ValueError),
                          (pd.DataFrame({'experiment': ['test','test'], 'child_id':['C01','C02'], 'recording_filename': ['rec1.wav','rec2.wav'], 'start_time': ['NA','05:32'], 'date_iso': ['2021-03-05','2023-11-29'],
                                       'recording_device_type': ['usb','unknown']}).convert_dtypes(), False, True, None),
                          (pd.DataFrame({'experiment': ['test','test'], 'child_id':['C01','C02'], 'recording_filename': ['rec1.wav','rec2.wav'], 'start_time': ['NA','05:32'], 'date_iso': ['2021-03-05','2023-11-29'],
                                       'recording_device_type': ['usb','unknown']}).convert_dtypes(), True, True, None),
                          (pd.DataFrame({'experiment': ['test','test'], 'child_id':['1','1'], 'recording_filename': ['rec1.wav','rec2.wav'], 'start_time': ['NA','05:32'], 'date_iso': ['2015-02-28','2023-11-29'],
                                       'recording_device_type': ['usb','unknown']}).convert_dtypes(), True, False, ValueError),
                          (pd.DataFrame({'experiment': ['test','test'], 'child_id':['1','1'], 'recording_filename': ['rec1.wav','rec2.wav'], 'start_time': ['NA','05:32'], 'date_iso': ['2021-03-05','2023-11-29'],
                                       'recording_device_type': ['usb','unknown']}).convert_dtypes(), True, False, None),
                         ])
def test_write_recordings(project, new_recs, keep_discarded, skip_validation, error):
    project.read()
    discarded = project.discarded_recordings
    project.recordings = new_recs

    if error:
        with pytest.raises(error):
            project.write_recordings(keep_discarded=keep_discarded, skip_validation=skip_validation)
    else:
        project.write_recordings(keep_discarded=keep_discarded, skip_validation=skip_validation)

        project.read()
        new_recs = new_recs.assign(discard='0')
        new_recs['discard'] = new_recs['discard'].astype('string')
        pd.testing.assert_frame_equal(project.recordings.dropna(axis=1).reset_index(drop=True), new_recs.reset_index(drop=True), check_like=True)
        if keep_discarded:
            pd.testing.assert_frame_equal(project.discarded_recordings.dropna(axis=1).reset_index(drop=True), discarded.reset_index(drop=True), check_like=True)
        else:
            assert project.discarded_recordings.shape[0] == 0

@pytest.mark.parametrize("new_chis,keep_discarded,skip_validation,error",
                         [(pd.DataFrame({'experiment': ['test_xp'], 'child_id': ['1'], 'child_dob': ['2015-05-04']}).convert_dtypes(), False, False, ValueError),
                          (pd.DataFrame({'experiment': ['test'], 'child_id': ['1'], 'child_dob': ['2025-05-04']}).convert_dtypes(), True, False, ValueError),
                          (pd.DataFrame({'experiment': ['test'], 'child_id': ['1'], 'child_dob': ['2017-05-04']}).convert_dtypes(), True, False, None),
                          (pd.DataFrame({'experiment': ['test'], 'child_id': ['1'], 'child_dob': ['2025-05-04']}).convert_dtypes(), True, True, None),
                          (pd.DataFrame({'experiment': ['test'], 'child_id': ['1'], 'child_dob': ['2025-05-04']}).convert_dtypes(), False, True, None),
                         ])
def test_write_children(project, new_chis, keep_discarded, skip_validation, error):
    project.read()
    discarded = project.discarded_children.copy()
    project.children = new_chis

    if error:
        print(project.validate(current_metadata=True))
        with pytest.raises(error):
            project.write_children(keep_discarded=keep_discarded, skip_validation=skip_validation)
    else:
        print(project.validate(current_metadata=True))
        project.write_children(keep_discarded=keep_discarded, skip_validation=skip_validation)

        project.read(accumulate=False)
        new_chis['discard'] = '0'
        new_chis['discard'] = new_chis['discard'].astype('string')
        pd.testing.assert_frame_equal(project.children.dropna(axis=1).reset_index(drop=True),
                                      new_chis.reset_index(drop=True), check_like=True)
        if keep_discarded:
            print(project.discarded_children)
            print(discarded)
            pd.testing.assert_frame_equal(project.discarded_children.reset_index(drop=True),
                                          discarded.reset_index(drop=True), check_like=True)
        else:
            assert project.discarded_children.shape[0] == 0

def test_dict_summary(project):
    project.read()
    summary = project.dict_summary()
    assert summary == {'recordings': {'count': 2, 'duration': 8000, 'first_date': '2020-04-20', 'last_date': '2020-04-21', 'discarded': 1, 'devices': {'usb': {'count': 2, 'duration': 8000}}}, 'children': {'count': 1, 'min_age': 3.6139630390143735, 'max_age': 3.646817248459959, 'M': None, 'F': None, 'languages': {}, 'monolingual': None, 'multilingual': None, 'normative': None, 'non-normative': None}}

@pytest.mark.parametrize("file_path,dst_file,dst_path,file_type,overwrite,error",
     [(PATH / 'metadata/children/0_test.csv', 'rec008.wav', PATH / 'recordings/raw/rec008.wav', 'recording', False, None),
    (PATH / 'metadata/children/0_test.csv', 'rec008', PATH / 'recordings/raw/rec008', 'recording', False, None),
    (PATH / 'metadata/children/0_test.csv', Path('rec008.wav'), PATH / 'recordings/raw/rec008.wav', 'recording', False, None),
    (PATH / 'metadata/children/0_test.csv', 'metrics.csv', PATH / 'extra/metrics.csv', 'extra', False, None),
    (PATH / 'metadata/children/0_test.csv', 'sound.wav', None, 'recording', False, FileExistsError),
    (PATH / 'made_up_file.txt', 'sound5.wav', None, 'recording', False, FileNotFoundError),
    (PATH / 'metadata/children/0_test.csv', '/etc/sound.wav', None, 'recording', False, AssertionError),
    (PATH / 'metadata/children/0_test.csv', 'sound5.wav', None, 'made_up_type', False, ValueError),
    (PATH / 'metadata/children/0_test.csv', 'metadata.xlsx', PATH / 'metadata/metadata.xlsx', 'metadata', False, None),
    (PATH / 'metadata/children/0_test.csv', 'children.csv', PATH / 'metadata/children.csv', 'metadata', True, None),
    (PATH / 'metadata/children/0_test.csv', 'README.md', PATH / 'README.md', 'raw', False, None),
    (PATH / 'metadata/children/0_test.csv', 'scripts/any_script.py', PATH / 'scripts/any_script.py', 'raw', True, None),
    (PATH / 'metadata/children/0_test.csv', '../other_place', None, 'raw', False, AssertionError),
    (PATH / 'metadata/children/0_test.csv', 'fake_readme.md', None, 'raw', True, AssertionError),
    (str(PATH / 'metadata/children/0_test.csv'), 'scripts/new_subfolder/any_script.py', PATH / 'scripts/new_subfolder/any_script.py', 'raw', False, None),
      ])
def test_add_project_file(project, file_path, dst_file, dst_path, file_type, overwrite, error):
    if error is not None:
        with pytest.raises(error):
            project.add_project_file(file_path, dst_file, file_type, overwrite)
    else:
        project.add_project_file(file_path, dst_file, file_type, overwrite)
        assert filecmp.cmp(file_path, dst_path)

@pytest.mark.parametrize("file,dst,file_type,error",
     [('sound.wav', PATH / 'recordings/raw/sound.wav', 'recording', None),
    ('sound6.wav', None, 'recording', FileNotFoundError),
    ('children.csv', PATH / 'recordings/raw/rec008.wav', 'metadata', None),
    ('/sound.wav', None, 'recording', AssertionError),
    ('../../../../sound.wav', None, 'recording', AssertionError),
    ('fake_readme.md', None, 'raw', AssertionError),
      ])
def test_remove_project_file(project, file, dst, file_type, error):
    if error is not None:
        with pytest.raises(error):
            project.remove_project_file(file, file_type)
    else:
        project.remove_project_file(file, file_type)
        assert not dst.exists()
