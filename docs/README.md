- [Introduction](#introduction)
  - [Dataset format and structure](#dataset-format-and-structure)
  - [Available tools](#available-tools)
- [Installation](#installation)
  - [Install our package via pip](#install-our-package-via-pip)
  - [Install git-annex](#install-git-annex)
  - [Check the setup](#check-the-setup)
- [Getting some data](#getting-some-data)
  - [Extant datasets](#extant-datasets)
- [Converting a dataset into ChildRecordsData format](#converting-a-dataset-into-childrecordsdata-format)
- [Cheatsheet](#cheatsheet)
- [Using our tools](#using-our-tools)
- [Missing and planned features](#missing-and-planned-features)
- [Help-needed ?](#help-needed-)

## Introduction

Day-long (audio-)recordings of children are increasingly common, but there is no scientific standard formatting that can benefit the organization and analyses of such data. ChildRecordData provides standardizing specifications and tools for the storage and management of day-long recordings of children and their associated meta-data and annotations.

![structure](http://laac-lscp.github.io/ChildRecordsData/images/structure.png "File organization structure")

We assume that the data include three very different types:

1. Audio, of which we distinguish the raw audio extracted from the hardware; and a version that has been converted into a standardized format. These audios are the long-form ones. At the time being, we do not foresee including clips extracted from these long-form audios, and assume that any such process will generate some form of annotation that can then be re-cast temporally to the long-form audio.
2. Annotations, of which we again distinguish raw and standardized versions. At present, we can import from Praat's textgrid, ELAN's eaf, and VTC's rttm format.
3. Metadata corresponding to the children, recordings, and annotations, which will therefore also describe the converted recordings.

 [formatting instructions and specifications](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html)


### Dataset format and structure

We provide instructions for setting up the metadata in [formatting instructions and specifications](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html). Read on for instructions on how to get or produce data in this format.

### Available tools

Day-long audiorecordings are often collected using a LENA recorder, and analyzed with the LENA software. However, open source alternatives to the LENA commercial environment are emerging, some of which are shown in the following figure.

![structure](http://laac-lscp.github.io/ChildRecordsData/images/tools.png "Overview of some tools in the day-long recordings environment")

For instance, alternative hardware includes the babylogger and any other light-weight recording device with enough battery and storage to record over several hours.

Alternative automated analysis options include the [Voice Type Classifier](https://github.com/MarvinLvn/voice-type-classifier), which segments the audio into different talker types (key child, female adult, etc) and [ALICE](https://github.com/orasanen/ALICE), an automated linguistic unit counter.

As for manual annotation options, [ELAN](https://archive.mpi.nl/tla/elan) can be used, for instance employing the [ACLEW DAS annotation scheme](https://osf.io/b2jep/). Assignment of annotation to individuals and evaluation can be done using [Seshat](https://github.com/bootphon/seshat). Finally, [Zooniverse](zooniverse.org) can be used to crowd-source certain aspects of the classification with the help of citizen scientists.

In this context, we provide tools and a procedure to:

- Validate datasets (making sure that metadata, recordings and annotations are in the right place and format)
- Convert your raw recordings into the desired format
- Import raw annotations (from ELAN, Praat, csv, rttm from VTC and ALICE) into our standardized format
- Add clips to an annotation pipeline in Zooniverse, and retrieve the ensuing annotations

## Installation

You'll need to install our package and git-annex, following the instructions we provide next.

### Install our package via pip
  
The package can be installed using pip :

```
pip install git+https://github.com/LAAC-LSCP/ChildRecordsData.git
```

Important: the package requires Python 3+, so make sure `pip` points to a python3 installation, otherwise try pip3. You can do so by typing `pip --version`.

If you are having permissions issues - or any other issue -, you can try any of the following :

 - Use the `--user` flag:

```
pip install git+https://github.com/LAAC-LSCP/ChildRecordsData.git --user
```

 - Create and activate a virtual python environment beforehand. You will have to activate it everytime you need to use the package or datalad.

```
python3 -m venv ~/ChildProjectVenv
source ~/ChildProjectVenv/bin/activate
pip install git+https://github.com/LAAC-LSCP/ChildRecordsData.git
```

### Install git-annex

The package also requires git-annex in order to retrieve the datasets. It can be installed via `apt install git-annex` or `brew install git-annex`. It is encouraged to use a recent version of git-annex (8.2+).

### Check the setup

You can check the setup by issuing the following commands :

```
$ datalad --version
datalad 0.13.7
```

```
$ child-project --help
usage: child-project [-h]
                     {validate,import-annotations,merge-annotations,remove-annotations,import-data,stats,compute-durations,zooniverse,convert}
                     ...

positional arguments:
  {validate,import-annotations,merge-annotations,remove-annotations,import-data,stats,compute-durations,zooniverse,convert}

optional arguments:
  -h, --help            show this help message and exit
```

## Getting some data

You can either have some data of your own that you would like to use the package on, or you may know of some datasets that are already in this format that you'd like to reuse.

It may be easier to start with an extant dataset. Here is the list that we know exists. Please note that the large majority of these data are NOT public, and thus if you cannot retrieve them, this means you need to get in touch with the data managers.

### Extant datasets

Instructions to download extant datasets can be found [here](http://laac-lscp.github.io/ChildRecordsData/TUTORIAL_REUSE.html).

The list of extant datasets can be found [here](http://laac-lscp.github.io/ChildRecordsData/EXTANT.html).


## Converting a dataset into ChildRecordsData format

If you have your own dataset, you can convert it into our format according to our 
[specifications](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html). Once you have done so, you can use the package's tools. 

We provide a detailed tutorial to convert a dataset into our format and then publish it [here](http://laac-lscp.github.io/ChildRecordsData/TUTORIAL_VANDAM.html).

## Cheatsheet

We provid a [cheatsheet](http://laac-lscp.github.io/ChildRecordsData/CHEATSHEET.html) for the most common tasks.

## Using our tools

We provide tools to:

- Validate raw data
- Convert your raw recordings into a standardized format
- Add recording duration to your metadata
- Import raw annotations (from LENA, ELAN, Praat, csv, rttm) into our standardized format
- Add clips to an annotation pipeline in Zooniverse, and retrieve the ensuing annotations

We provide detailed instructions [here](http://laac-lscp.github.io/ChildRecordsData/TOOLS.html).

## Missing and planned features

- clarify link with R package

## Help-needed ?

If you need more help than this documentation provides, feel free to ask your question [here](https://github.com/LAAC-LSCP/ChildRecordsData/issues/new?assignees=&labels=help+wanted%2C+question&template=question.md&title=).
