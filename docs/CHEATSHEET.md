- [Cheatsheet](#cheatsheet)
  - [DataLad cheatsheet](#datalad-cheatsheet)
    - [Installing a dataset](#installing-a-dataset)
    - [Getting data](#getting-data)
    - [Getting updates](#getting-updates)
    - [Saving changes](#saving-changes)
    - [Publishing changes](#publishing-changes)
  - [ChildProject cheatsheet](#childproject-cheatsheet)

# Cheatsheet

## DataLad cheatsheet

### Installing a dataset

```bash
datalad install [-h] [-s SOURCE] [-d DATASET] [-g] [-D DESCRIPTION] [-r] [-R LEVELS] [--reckless [auto|ephemeral|shared-...]] [-J NJOBS] [PATH [PATH ...]]
```

Example:

```
datalad install -r git@github.com:LAAC-LSCP/datasets.git
```

*Note: some datasets might have additional installation instructions!*

More: [datalad install](http://docs.datalad.org/en/stable/generated/man/datalad-install.html)

### Getting data

```bash
datalad get [-h] [-s LABEL] [-d PATH] [-r] [-R LEVELS] [-n] [-D DESCRIPTION] [--reckless [auto|ephemeral|shared-...]] [-J NJOBS] [PATH [PATH ...]]
```

Example:

```
datalad get annotations/vtc
```

More: [datalad get](http://docs.datalad.org/en/stable/generated/man/datalad-get.html)

### Getting updates

```bash
datalad update --merge
```

More: [datalad update](http://docs.datalad.org/en/stable/generated/man/datalad-update.html)


### Saving changes

```bash
datalad save [-h] [-m MESSAGE] [-d DATASET] [-t ID] [-r] [-R LEVELS] [-u] [-F MESSAGE_FILE] [--to-git] [-J NJOBS] [PATH [PATH ...]]
```

Example:

```
datalad save metadata/children.csv -m "correcting children metadata"
```

`datalad save` is analoguous to doing `git add`+`git commit`. It will decide automatically whether to store the files in git or in the annex.

*Note: datalad save records the changes locally. They still have to be pulished - just like with git commit !*

More: [datalad save](http://docs.datalad.org/en/stable/generated/man/datalad-save.html)

### Publishing changes

```bash
datalad push [-h] [-d DATASET] [--to SIBLING] [--since SINCE] [--data {anything|nothing|auto|auto-if-wanted}] [-f {all|gitpush|checkdatapresent}] [-r] [-R LEVELS] [-J NJOBS] [PATH [PATH ...]]
```

Example:

```bash
datalad push
```

More: [datalad push](http://docs.datalad.org/en/stable/generated/man/datalad-push.html)


## ChildProject cheatsheet

