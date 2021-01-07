- [Introduction](#introduction)
- [Dataset format and structure](#dataset-format-and-structure)
- [Installation](#installation)
- [Installing datasets](#installing-datasets)
- [Working with the data](#working-with-the-data)
  - [Validate raw data](#validate-raw-data)
  - [Convert recordings](#convert-recordings)
    - [Multi-core audio conversion with slurm on a cluster](#multi-core-audio-conversion-with-slurm-on-a-cluster)
  - [Import annotations](#import-annotations)
    - [Single importation](#single-importation)
    - [Bulk importation](#bulk-importation)
  - [Zooniverse](#zooniverse)
  - [Compute recordings duration](#compute-recordings-duration)

## Introduction

ChildRecordData provides specifications and tools for the storage and management of day-long recordings of children and their associated meta-data and annotations.

![structure](http://laac-lscp.github.io/ChildRecordsData/images/structure.png "File organization structure")

### Dataset format and structure

See the [formatting instructions and specifications](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html)

### Available tools

These are introduced in more detail below, but in a nutshell we provide tools and a procedure to:

- Validate raw data
- Convert your raw recordings into a standardized format
- Add recording duration to your metadata
- Import raw annotations (from ELAN, Praat, csv, rttm) into our standardized format
- Add clips to an annotation pipeline in Zooniverse, and retrieve the ensuing annotations

## Installation

You'll need to install our package and git-annex, following the instructions we provide next.

### 1. Install our package via pip
  
The package can be installed using pip :

```
pip install git+https://github.com/LAAC-LSCP/ChildRecordsData.git
```

(The package requires python 3, so make sure `pip` points to a python3 installation, otherwise try pip3.)

If you are having permissions issues - or any other issue -, you can create and activate a python environment beforehand :

```
python3.6 -m venv ~/ChildProjectVenv
source ~/ChildProjectVenv/bin/activate
pip install git+https://github.com/LAAC-LSCP/ChildRecordsData.git
```

### 2. Install git-annex

The package also requires git-annex in order to retrieve the datasets. It can be installed via `apt install git-annex` or `brew install git-annex`. It is encouraged to use a recent version of git-annex (8.2+).

## Getting some data

You can either have some data of your own that you would like to use the package on, or you may know of some datasets that are already in this format that you'd like to reuse.

It may be easier to start with an extant dataset. Here is the list that we know exists. Please note that the large majority of these data are NOT public, and thus if you cannot retrieve them, this means you need to get in touch with the data managers.

### Extant datasets

Instructions to download extant datasets can be found [here](http://laac-lscp.github.io/ChildRecordsData/REUSE.html).

#### Public data sets (TODO)

We have prepared a public data set for testing purposes which is based on the [VanDam Public Daylong HomeBank Corpus](https://homebank.talkbank.org/access/Public/VanDam-Daylong.html); VanDam, Mark (2018). VanDam Public Daylong HomeBank Corpus. doi:10.21415/T5388S.



#### From the [LAAC team](https://lscp.dec.ens.fr/en/research/teams-lscp/language-acquisition-across-cultures)


| Name | Authors | Location | Recordings | Audio length (hours) | Status |
|------|---------|----------|------------|----------------------|--------|
{% for project in projects -%}
| **{{project.name}}** | {{project.authors}} | [{{project.location}}]({{project.location}}) | {{project.recordings}} | {{project.duration|round|int}} | {{project.status}} | 
{% endfor %}


#### Other private datasets

We know of no other private datasets at present

## Converting a dataset into ChildRecordsData format

If you have your own dataset, you can convert it into our format using these  
[formatting instructions and specifications](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html)

Once you have done so, you can use the package's tools to:

- Validate raw data
- Convert your raw recordings into a standardized format
- Add recording duration to your metadata
- Import raw annotations (from ELAN, Praat, csv, rttm) into our standardized format
- Add clips to an annotation pipeline in Zooniverse, and retrieve the ensuing annotations

We provide detailed instructions next.

### Validate raw data

This is typically done (repeatedly!) in the process of importing your data into our format for the first time, but you should also do this whenever you make a change to the dataset.

Looks for errors and inconsistency in the metadata, or for missing audios. The validation will pass if the [formatting instructions](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html) are met.

```
child-project validate /path/to/dataset
```


### Convert recordings

Converts all recordings in a dataset to a given encoding. Converted audios are stored into `converted_recordings/$name`.


```
child-project convert /path/to/dataset --name=16kHz --format=wav --sampling=16000 --codec=pcm_s16le
```


We typically run the following, to split long sound files every 15 hours, because the software we use for human annotation (ELAN, Praat) works better with audio that is maximally 15h long:

```
child-project convert /path/to/dataset --name=16kHz --split=15:00:00 --format=wav --sampling=16000 --codec=pcm_s16le
```


#### Multi-core audio conversion with slurm on a cluster

If you have access to a cluster with slurm, you can use a command like the one below to batch-convert your recordings. Please note that you may need to change some details depending on your cluster (eg cpus per task). If needed, refer to the [slurm user guide](https://slurm.schedmd.com/quickstart.html)

```
sbatch --mem=64G --time=5:00:00 --cpus-per-task=4 --ntasks=1 -o namibia.txt child-project convert /path/to/dataset --name standard --format WAV --codec pcm_s16le --sampling 16000 --threads 4`
```

### Compute recordings duration

Compute recordings duration and store in into a column named 'duration' in the metadata.

```
child-project compute-durations [--force] /path/to/dataset
```

### Import annotations

Annotations can be imported one by one or in bulk. Annotation importation does the following :

1. Convert all input annotations from their original format (e.g. rttm, eaf, textgrid..) into the CSV format defined [here](https://laac-lscp.github.io/ChildRecordsData/FORMATTING.html#annotations-format) and stores them into `annotations/`.
2. Registers them to the annotation index at `metadata/annotations.csv`

#### Single importation

```
child-project import-annotations /path/to/dataset --set eaf --recording_filename sound.wav --time_seek 0 --raw_filename example.eaf --range_onset 0 --range_offset 300 --format eaf
```

#### Bulk importation

```
child-project import-annotations /path/to/dataset --annotations /path/to/dataframe.csv
```

The input dataframe `/path/to/dataframe.csv` must have one entry per annotation to import, according to the format specified [here](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html#annotation-importation-input-format).

### Zooniverse

Find all the instructions on how-to use Zooniverse together with child-project [here](http://laac-lscp.github.io/ChildRecordsData/ZOONIVERSE.html).

