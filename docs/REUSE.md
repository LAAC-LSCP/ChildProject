- [How to reuse extant datasets](#how-to-reuse-extant-datasets)
  - [How it works](#how-it-works)
  - [Installing datalad](#installing-datalad)
  - [Installing a dataset](#installing-a-dataset)
    - [Setting up your ssh access to GitHub](#setting-up-your-ssh-access-to-github)
    - [Setting up your ssh access to Oberon](#setting-up-your-ssh-access-to-oberon)
    - [Quick way (using child-project)](#quick-way-using-child-project)
    - [Datalad way (using only datalad)](#datalad-way-using-only-datalad)
  - [Downloading large files](#downloading-large-files)
  - [Updating a dataset](#updating-a-dataset)
  - [Contributing to a dataset](#contributing-to-a-dataset)
  - [Creating a new dataset](#creating-a-new-dataset)


# How to reuse extant datasets

Our datasets rely on [datalad](https://www.datalad.org/). Datalad allows the versioning and distribution of large datasets.
Datalad relies on another tool called [git-annex](https://git-annex.branchable.com/), which itself is an extension of git providing support for large file versioning with a high flexibility. 

## How it works

Our dataset are distributed on two "siblings". Siblings are analogous to git and git-annex remotes.
The github remote doesn't include large files, only pointers refering to them. The large files are stored in a sibling hosted on the /scratch1 partition available through Oberon.

![structure](http://laac-lscp.github.io/ChildRecordsData/images/infrastructure.png "Dataset infrastructure")

## Installing datalad

*Important: datasets are hosted on github and oberon. This means you are required access to our private github repositories as well as ssh access to oberon. You will be prompted for credentials everytime you issue datalad commands, so we recommand using SSH keys and enabling the Keychain (append `~/.ssh/config` with `UseKeychain yes`).*

1. Install git-annex using `apt install git-annex` (linux) or `brew install git-annex` (mac). Git-annex is available by default on Oberon.
2. Install datalad with pip : `pip install datalad` (or  `pip install datalad --user` in case of permission issues)

If you are still having permission issues, consider using python virtual environments or conda. Otherwise, refer to your system administrator.

## Installing a dataset

### Setting up your ssh access to GitHub

This step shall only be done once for all.
Since our datasets are partially hosted on GitHub private repositories, authentication is needed to access them. I highly recommend that you use SSH authentication, for security and simplicity. I will only provide instructions for SSH authentication, and leave HTTPS authentication to consenting adults. Here are the steps:

1. Copy your SSH private key (usually located in `~/.ssh/id_rsa.pub`)
2. Go to [GitHub.com > Settings > SSH and GPG Keys](https://github.com/settings/keys)
3. Click on the green button 'New SSH key' and paste your public key where requested.

### Setting up your ssh access to Oberon

The actual content of the large files (e.g. recordings) is stored on the lab cluster (Oberon). Therefore, you need ssh access to the cluster to retrieve them.

If you can already ssh into Oberon with an alias by doing `ssh foberon` or `ssh oberon` or similar, skip this step. Otherwise, follow these instructions to setup ssh properly:

1. Add the following lines to your `~/.ssh/config` file:

```
Host frontal
   Hostname frontal-ssh.ens.fr
   User lgautheron

Host foberon
    Hostname 129.199.81.30
    User lgautheron 
    ProxyJump frontal

Host *
ForwardX11 yes
UseKeychain yes
```

2. Try to SSH into oberon by typing `ssh foberon`. You will be prompted for your credentials.

### Quick way (using child-project)

```
child-project import-data git@github.com:LAAC-LSCP/namibia-data.git --destination namibia-data --storage-hostname foberon
```

### Datalad way (using only datalad)

1. The first step is to clone the dataset from github, for instance :

```
datalad install git@github.com:LAAC-LSCP/namibia-data.git
cd namibia-data
```

2. The next step is configure the access to the cluster (where large files are stored). If you are installing the dataset on oberon, you just have to type :

```
datalad run-procedure setup
```

Otherwise, you need to specify the SSH alias for Oberon on your system. If you followed the previous steps to setup your SSH access to Oberon, it should be `foberon` :

```
datalad run-procedure setup foberon
```

That's it ! Your dataset is ready to go. By default, large files do not get downloaded automatically. See the next section for help with downloading those files.

## Downloading large files

Files can be retrieved using `datalad get [path]`. For instance, `datalad get recordings` will download all recordings.

## Updating a dataset

A dataset can be updated from the sources using `dataset update`.

## Contributing to a dataset

You can save local changes to a dataset with `datalad save [path] -m "commit message"`. For instance :

```
datalad save raw_annotations/vtc -m "adding vtc rttms"
```

These changes still have to be pushed, which can be done with :

```
datalad publish --to scratch1 --transfer-data all
```

## Creating a new dataset

This section is a work in progress.

```
DATASETNAME = "tsimane2017-data"
datalad create -c laac $DATASETNAME
cd $DATASETNAME
datalad create-sibling-github -s origin --github-organization LAAC-LSCP --access-protocol ssh $DATASETNAME
datalad create-sibling -s scratch1 "/scratch1/data/laac_data/$DATASETNAME"
echo "/scratch1/data/laac_data/$DATASETNAME" > .datalad/path
datalad run-procedure setup
datalad publish --to scratch1
```