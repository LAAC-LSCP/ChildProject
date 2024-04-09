# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

- eaf_builder replicating the subtree structure of recordings is not happening anymore. And individual files are not placed in individual subfolder anymore

## [0.1.2] 2024-03-14

### Added

- add the derived annotation pipeline
- audio conversion has now a 'standard' conversion pipeline that will convert files to mono channel 16kHz pcm_s16le (no need to convert channels and then sampling rate and no need to know the options)
- add the simple_CTC metric to the list of available metrics

### Changed

- migrated the project to a pyproject.toml implementation
- the output of the CLI in the terminal is now handled by the logger module and not by print statements
- validating a dataset now results in warnings for broken symlinks and no errors anymore (#425)
- validation with recordings existing but for which mediainfo can't read the sample rate no longer fail but outputs a warning.
- recording_device_type in recordings.csv now accepts izyrec
- rework of the list of available metrics functions: now has a absolute value and per_hour decorator, and a peak_ option

### Fixed

- zooniverse chunks upload was failing if the dataset column was missing in the csv
- eaf_builder with speaker_id NA no longer fails (#438)
- name of files in rttm that contain only digits in their name now work correctly with the filter ($457)
- periodic sampler having extra NA rows when using explode

## [0.1.1] 2023-05-11

### Added

- validation of the annotation index checks for annotation period outside of audio duration

### Changed

- ignore_discarded is now the default behaviour of childproject projects, robustness was added to the discard column
- annotation index validation and annotation importation now checks for the range_offset being greater than duration and oiutputs an error when it is the case
- projects now check for unicity of the experiment column in children and recording csv, read() fails when not unique
- zooniverse uploads : uploads extra metadata to zooniverse (name of the audio clip, dataset it belongs to)
- zooniverse uploads : the subject_set id is stored in the chunk csv file, as the subject_set name (display name) is susceptible to change
- zooniverse uploads : the upload now handles SIGINT and SIGTERM signals to save progression of the upload to the csv before exiting (useful when a job needs to be interrupted
- allow once again get_within_time_range to take str arguments as times
- add arguments to choose the format of compute_ages project method

### Fixed

- discard column in recordings.csv and children.csv now works properly
- metrics pipeline now checks the converted name for unicity even if a specific name was given
- rename set also renames the merged_from column
- rename set accepts subsets location without failing

## [0.1.0] 2023-02-20

### Added

- Windows automated tests (some functions were edited to be windows compatible). TO REMEMBER: type int in windows default to int32 instead of int64, can lead to big int turned into negative values
- Check calculated ages during corpus validation

### Changed

- pandas version restricted to avoid errors of future releases , (1.1.0 (assert_frame_equal check_less_precise) to 1.5.0 (last checked version))
- no usage of sox command anymore, remove sox dependency
- merging annotations now sets the format to 'NA' instead of a blank value.

### Fixed

- replace exit() with raise ValueError() to comply with Exception propagation (metrics pipeline)
- fixed ignore-errors in zooniverse upload_chunks
- fixed calculate_shift to correctly reshape to single channel

## [0.0.7] 2022-09-14

### Fixed

- missing column merged_from in annotations.csv does not fail anymore

## [0.0.6] 2022-09-13

### Added

- Start times can include seconds (e.g. 12:34:59) while still accepting the old format (This change will allow other columns to accept multiple formats easily).
- `child-project --version` command
- `merge_sets` in `AnnotationManager` method now accepts arguments [full_set_merge,skip_existing,recording_filter] to carry out partial merges
- `child-project metrics` added the `--segments` command to extract metrics from a dataFrame of segments
- metrics <voc_speaker> <lena_CTC> and <lena_CVC>

### Changed

- `metrics` pipeline's options `--from --to` require a HH:MM:SS format now.
- `merge-annotations` command fails when the output_set already exists or if the required sets don't exist


### Fixed

- eafbuilder attributes a default time-aligneable ling-type to created tiers to avoid random attribution that can lead to wrong behaviour and crashes
- 'imported_at' column in annotations.csv did not have a new correct format (in a set)
- metrics avg_cry_... avg_can_... and avg_non_can_... were not calculated correctly
- metrics lp_n lp_dur use lena columns in priority
- metrics lena-CTC and lena-CVC are added correctly and added to the output of the lena pipeline
- praat-parselmouth is now in the setup file so the dependency get installed automatically by conda

### Dropped

- `child-project process --split` --split option dropped as there is no further need of reducing long audios (>15hs)

## [0.0.5] - 2022-07-25

### Added

 - `--spectrogram` option in the `zooniverse extract-chunks` pipeline to generate an image of a spectrogram that will help citizen-scientists with the classification on zooniverse.
 - `child-project compare-recordings` command added to allow users to prin a divergence score. This will help identify audio files that are just duplicates of others (and possibily have different codecs/sampling rate/number of channels)
 - `--import-speech-from` command added to the eaf builder to integrate previous annotations to the eaf file (e.g. VTC segments) to facilitate annotation process

### Changed

- `metrics` pipeline, reworked to be more flexible. Performance hit with it.
    - Old pipelines still exist
    - new usage of `--period` option on every pipeline and for eveery metric.
    - Usage of a csv file to specify the list of metrics wanted
    - Ease of adding new metrics to the supported list
    - Outputs a yml parameter file that can be reused to compute the same metrics and keep a trace of what was run.
- changes to standard annotation value (addressee, vcm_type etc)

### Fixed

- importation of empty file now correctly generates an empty converted file
- `--period` option correctly works with other units than `recording_filename`

### Dropped

- Support for python 3.6

## [0.0.4] - 2022-02-02

### Added 

 - Conversation sampler
 - `get_within_ranges` function to retrieve all annotations that match the desired portions of the recordings
 - `--import-speech-from` option for the EAF annotation builder to pre-fill annotations based on any previously imported set of annotations
 - `compute_ages` function to compute the age of the subject child for each recording
 - `lena_speaker` for the LENA its converter
 - `lena_speaker` aggregated metrics for the LenaMetrics pipeline
 - Improved AclewMetrics and LenaMetrics performance
 - Improved error handling (dataframes sanity checks)

### Changed

 - More flexible high-volubility sampler

### Fixed
 
 - Fixed pipelines crashes in presence of NA values in `recording_filename`
 - `RandomVocalizationSampler` crash fix

## [0.0.3] - 2021-10-06

### Fixed

 - Fixed exceptions thrown by child-project CLI

## [0.0.2] - 2021-09-29

### Added 

 - CSV importer to register pre-exisiting CSV annotations into the index without performing any conversion
 - Enable Zooniverse pipeline's `retrieve-classifications` to match classifications to the original chunks metadata
 - `get_within_time_range` method to clip a list of annotations within a given time-range
 - `get_segments_timestamps` method to calculate the onset and offset timestamps of each segment from an annotation
 - `--from-time`/`--to-time` option for metrics extraction
 - Time-unit aggregated metrics, supporting custom time periods.
 - optional `--recordings` option to apply the audio processors to specific recordings only
 - allow `child-project validate` to check custom recordings profiles and/or annotation sets
 - `--ignore-errors` switch for Zooniverse pipeline's `upload-chunks`
 - `enforce_dtypes` option for `ChildProject` in order to enforce the dtype of certain metadata columns (e.g. session_id, child_id) to their expected dtype

### Fixed

 - Fixed skip-existing argument of the basic audio processor
 - Fixed a crash-bug that occured while extracting metrics from recordings with no FEM/MAL/CHI/OCH segment
 - Made `pyannote-agreement` an optional dependency
 - Added dependency constraints to fix some installation issues.

## [0.0.1] - 2021-07-14

### Added

- First proper release of the package.

[unreleased]: https://github.com/LAAC-LSCP/ChildProject/compare/v0.0.7...HEAD
[0.0.7]: https://github.com/LAAC-LSCP/ChildProject/releases/tag/v0.0.7
[0.0.6]: https://github.com/LAAC-LSCP/ChildProject/releases/tag/v0.0.6
[0.0.5]: https://github.com/LAAC-LSCP/ChildProject/releases/tag/v0.0.5
[0.0.4]: https://github.com/LAAC-LSCP/ChildProject/releases/tag/v0.0.4
[0.0.3]: https://github.com/LAAC-LSCP/ChildProject/releases/tag/v0.0.3
[0.0.2]: https://github.com/LAAC-LSCP/ChildProject/releases/tag/v0.0.2
[0.0.1]: https://github.com/LAAC-LSCP/ChildProject/releases/tag/v0.0.1
