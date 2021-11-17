# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added 

 - `lena_speaker` for the LENA its converter
 - Improved error handling (dataframes sanity checks)

### Fixed
 
 - Fixed pipelines crashes in presence of NA values in `recording_filename`

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

[unreleased]: https://github.com/LAAC-LSCP/ChildProject/compare/v0.0.3...HEAD
[0.0.3]: https://github.com/LAAC-LSCP/ChildProject/releases/tag/v0.0.3
[0.0.2]: https://github.com/LAAC-LSCP/ChildProject/releases/tag/v0.0.2
[0.0.1]: https://github.com/LAAC-LSCP/ChildProject/releases/tag/v0.0.1
