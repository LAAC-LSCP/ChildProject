# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added 

 - CSV importer to register pre-exisiting CSV annotations into the index without performing any conversion
 - Enable Zooniverse pipeline's `retrieve-classifications` to match classifications to the original chunks metadata
 - `get_within_time_range` method to clip a list of annotations within a given time-range
 - `get_segments_timestamps` method to calculate the onset and offset timestamps of each segment from an annotation
 - `--from-time`/`--to-time` option for metrics extraction
 - Time-unit aggregated metrics, supporting custom time periods.
 - optional `--recordings` option to apply the audio processors to specific recordings only
 - allow `child-project validate` to check custom recordings profiles and/or annotation sets

### Fixed

 - Fixed skip-existing argument of the basic audio processor
 - Fixed a crash-bug that occured while extracting metrics from recordings with no FEM/MAL/CHI/OCH segment

## [0.0.1] - 2021-07-14

### Added

- First proper release of the package.

[unreleased]: https://github.com/LAAC-LSCP/ChildProject/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/LAAC-LSCP/ChildProject/releases/tag/v0.0.1
