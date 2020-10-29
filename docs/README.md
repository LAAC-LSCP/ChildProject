- [Introduction](#introduction)
- [Data formatting and structure](#data-formatting-and-structure)
- [Installation](#installation)
  - [Installing the package](#installing-the-package)
- [Usage](#usage)
  - [Validate raw data](#validate-raw-data)
  - [Import raw data](#import-raw-data)
  - [Convert recordings](#convert-recordings)
    - [Multi-core audio conversion with slurm on a cluster](#multi-core-audio-conversion-with-slurm-on-a-cluster)
  - [Import annotations](#import-annotations)
- [Available data](#available-data)

## Introduction

ChildRecordData provides specifications and tools for the storage and management of day-long recordings of children and their associated meta-data and annotations. 

![structure](http://laac-lscp.github.io/ChildRecordsData/images/structure.png "File organization structure")

## Data formatting and structure

See the [formatting instructions and specifications](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html)

## Installation

```
python3.6 -m venv ~/ChildProjectVenv

git clone https://github.com/lucasgautheron/ChildRecordsData.git
cd ChildRecordsData
source ~/ChildProjectVenv/bin/activate
pip install -r requirements.txt
```

### Installing the package

If you want to import ChildProject modules into your code, you should install the package by doing :

```
pip install git+https://github.com/LAAC-LSCP/ChildRecordsData.git
```

## Usage

### Validate raw data

```
python validate_raw_data.py --source=/path/to/raw/data
```

### Import raw data

Copy all raw data files to the specified destination and creates the working tree.

```
python import_data.py --source=/path/to/raw/data --destination=/path/to/the/working/directory
```

### Convert recordings

```
python convert.py --source=/path/to/project --name=16kHz --format=wav --sampling=16000 --codec=pcm_s16le
```

With audio splitting every 15 hours :

```
python convert.py --source=/path/to/project --name=16kHz --split=15:00:00 --format=wav --sampling=16000 --codec=pcm_s16le
```

#### Multi-core audio conversion with slurm on a cluster

```
sbatch --mem=64G --time=5:00:00 --cpus-per-task=4 --ntasks=1 -o namibia.txt ./convert.py --source ../data/namibia/ --name standard --format WAV --codec pcm_s16le --sampling 16000 --threads 4`
```

### Import annotations

```
python import_annotations.py --source /path/to/project --annotations /path/to/dataframe.csv
```

The input dataframe `/path/to/dataframe.csv` must have one entry per annotation to import, according to the format specified [here](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html#annotation-importation-input-format).

## Available data

The list of available data can be found [here](http://laac-lscp.github.io/ChildRecordsData/PROJECTS.html).