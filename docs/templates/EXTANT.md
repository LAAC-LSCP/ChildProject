- [List of available projects](#list-of-available-projects)
- [How to use datasets](#how-to-use-datasets)
  - [How it works](#how-it-works)
  - [Installing datalad](#installing-datalad)
  - [Installing a dataset](#installing-a-dataset)
    - [Quick way (using child-project)](#quick-way-using-child-project)
    - [Datalad way (using only datalad)](#datalad-way-using-only-datalad)
  - [Downloading large files](#downloading-large-files)
  - [Updating a dataset](#updating-a-dataset)
  - [Contributing to a dataset](#contributing-to-a-dataset)
  - [Creating a new dataset](#creating-a-new-dataset)


# List of extant datasets

## Public data sets (TODO)

We have prepared a public data set for testing purposes which is based on the [VanDam Public Daylong HomeBank Corpus](https://homebank.talkbank.org/access/Public/VanDam-Daylong.html); VanDam, Mark (2018). VanDam Public Daylong HomeBank Corpus. doi:10.21415/T5388S.



## From the [LAAC team](https://lscp.dec.ens.fr/en/research/teams-lscp/language-acquisition-across-cultures)


| Name | Authors | Location | Recordings | Audio length (hours) | Status |
|------|---------|----------|------------|----------------------|--------|
{% for project in projects -%}
| **{{project.name}}** | {{project.authors}} | [{{project.location}}]({{project.location}}) | {{project.recordings}} | {{project.duration|round|int}} | {{project.status}} | 
{% endfor %}


## Other private datasets

We know of no other private datasets at present, but we hope one day to be able to use [datalad's search feature](http://docs.datalad.org/en/stable/generated/man/datalad-search.html)
