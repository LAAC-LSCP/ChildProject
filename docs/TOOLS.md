- [Current tools](#current-tools)
  - [Validate raw data](#validate-raw-data)
  - [Convert recordings](#convert-recordings)
    - [Multi-core audio conversion with slurm on a cluster](#multi-core-audio-conversion-with-slurm-on-a-cluster)
  - [Compute recordings duration](#compute-recordings-duration)
  - [Import annotations](#import-annotations)
    - [Single importation](#single-importation)
    - [Merge annotation sets](#merge-annotation-sets)
    - [Remove an annotation set](#remove-an-annotation-set)
    - [Bulk importation](#bulk-importation)
  - [Zooniverse](#zooniverse)


# Current tools

## Validate raw data

This is typically done (repeatedly!) in the process of importing your data into our format for the first time, but you should also do this whenever you make a change to the dataset.

Looks for errors and inconsistency in the metadata, or for missing audios. The validation will pass if the [formatting instructions](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html) are met.

```
child-project validate /path/to/dataset
```


## Convert recordings

Converts all recordings in a dataset to a given encoding. Converted audios are stored into `converted_recordings/$name`.


```
child-project convert /path/to/dataset --name=16kHz --format=wav --sampling=16000 --codec=pcm_s16le
```


We typically run the following, to split long sound files every 15 hours, because the software we use for human annotation (ELAN, Praat) works better with audio that is maximally 15h long:

```
child-project convert /path/to/dataset --name=16kHz --split=15:00:00 --format=wav --sampling=16000 --codec=pcm_s16le
```


### Multi-core audio conversion with slurm on a cluster

If you have access to a cluster with slurm, you can use a command like the one below to batch-convert your recordings. Please note that you may need to change some details depending on your cluster (eg cpus per task). If needed, refer to the [slurm user guide](https://slurm.schedmd.com/quickstart.html)

```
sbatch --mem=64G --time=5:00:00 --cpus-per-task=4 --ntasks=1 -o namibia.txt child-project convert /path/to/dataset --name standard --format WAV --codec pcm_s16le --sampling 16000 --threads 4`
```

## Compute recordings duration

Compute recordings duration and store in into a column named 'duration' in the metadata.

```
child-project compute-durations [--force] /path/to/dataset
```

## Import annotations

Annotations can be imported one by one or in bulk. Annotation importation does the following :

1. Convert all input annotations from their original format (e.g. rttm, eaf, textgrid..) into the CSV format defined [here](https://laac-lscp.github.io/ChildRecordsData/FORMATTING.html#annotations-format) and stores them into `annotations/`.
2. Registers them to the annotation index at `metadata/annotations.csv`

### Single importation

Use this to import a single annotation file.

```
child-project import-annotations /path/to/dataset --set eaf --recording_filename sound.wav --time_seek 0 --raw_filename example.eaf --range_onset 0 --range_offset 300 --format eaf
```

### Merge annotation sets

```
child-project merge-annotations /path/to/dataset --left-set vtc --right-set alice --left-columns speaker_id,ling_type,speaker_type,vcm_type,lex_type,mwu_type,addresseee,transcription --right-columns phonemes,syllables,words --output-set alice_vtc
```

### Remove an annotation set

```
child-project remove-annotations /path/to/dataset --set vtc
```

### Bulk importation

Use this to do bulk importation of many annotation files.

```
child-project import-annotations /path/to/dataset --annotations /path/to/dataframe.csv
```

The input dataframe `/path/to/dataframe.csv` must have one entry per annotation to import, according to the format specified [here](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html#annotation-importation-input-format).


## Zooniverse

Find all the instructions on how-to use Zooniverse together with child-project [here](http://laac-lscp.github.io/ChildRecordsData/ZOONIVERSE.html).
