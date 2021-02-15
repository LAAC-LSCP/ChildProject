- [List of extant datasets](#list-of-extant-datasets)
  - [Public data sets](#public-data-sets)
  - [From the LAAC team](#from-the-laac-team)
  - [Other private datasets](#other-private-datasets)


# List of extant datasets

## Public data sets

We have prepared a [public dataset](https://github.com/LAAC-LSCP/vandam-daylong-demo) for testing purposes which is based on the [VanDam Public Daylong HomeBank Corpus](https://homebank.talkbank.org/access/Public/VanDam-Daylong.html); VanDam, Mark (2018). VanDam Public Daylong HomeBank Corpus. doi:10.21415/T5388S.


## From the [LAAC team](https://lscp.dec.ens.fr/en/research/teams-lscp/language-acquisition-across-cultures)


| Name | Authors | Location | Recordings | Audio length (hours) | Status |
|------|---------|----------|------------|----------------------|--------|
{% for project in projects -%}
| **{{project.name}}** | {{project.authors}} | [{{project.location}}]({{project.location}}) | {{project.recordings}} | {{project.duration|round|int}} | {{project.status}} | 
{% endfor %}


## Other private datasets

We know of no other private datasets at present, but we hope one day to be able to use [datalad's search feature](http://docs.datalad.org/en/stable/generated/man/datalad-search.html)
