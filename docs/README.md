- [Introduction](#introduction)
- [Data formatting and structure](#data-formatting-and-structure)
- [Installation](#installation)
- [Installing datasets](#installing-datasets)
- [Working with the data](#working-with-the-data)
  - [Validate raw data](#validate-raw-data)
  - [Convert recordings](#convert-recordings)
    - [Multi-core audio conversion with slurm on a cluster](#multi-core-audio-conversion-with-slurm-on-a-cluster)
  - [Import annotations](#import-annotations)
    - [Single importation](#single-importation)
    - [Bulk importation](#bulk-importation)

## Introduction

ChildRecordData provides specifications and tools for the storage and management of day-long recordings of children and their associated meta-data and annotations. 

![structure](http://laac-lscp.github.io/ChildRecordsData/images/structure.png "File organization structure")

## Data formatting and structure

See the [formatting instructions and specifications](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html)

## Installation

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

## Installing datasets

The list of available datasets with instructions to download them can be found [here](http://laac-lscp.github.io/ChildRecordsData/PROJECTS.html).

## Working with the data

### Validate raw data

```
child-project validate /path/to/raw/data
```

### Convert recordings

```
child-project convert /path/to/project --name=16kHz --format=wav --sampling=16000 --codec=pcm_s16le
```

With audio splitting every 15 hours :

```
child-project convert /path/to/project --name=16kHz --split=15:00:00 --format=wav --sampling=16000 --codec=pcm_s16le
```

#### Multi-core audio conversion with slurm on a cluster

```
sbatch --mem=64G --time=5:00:00 --cpus-per-task=4 --ntasks=1 -o namibia.txt child-project convert ../data/namibia/ --name standard --format WAV --codec pcm_s16le --sampling 16000 --threads 4`
```

### Import annotations

Annotations can be imported one by one or in bulk.

#### Single importation

```
child-project import-annotations /path/to/project --set eaf --recording_filename sound.wav --time_seek 0 --raw_filename example.eaf --range_onset 0 --range_offset 300 --format eaf
```

#### Bulk importation

```
child-project import-annotations /path/to/project --annotations /path/to/dataframe.csv
```

The input dataframe `/path/to/dataframe.csv` must have one entry per annotation to import, according to the format specified [here](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html#annotation-importation-input-format).

