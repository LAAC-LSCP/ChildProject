- [Converting a dataset](#converting-a-dataset)
  - [Set-up datalad and child-project](#set-up-datalad-and-child-project)
  - [Create a dataset](#create-a-dataset)
  - [Gather and sort the files](#gather-and-sort-the-files)
  - [Create the metadata](#create-the-metadata)
  - [Save the changes locally](#save-the-changes-locally)
  - [Processing](#processing)
  - [Publish the dataset](#publish-the-dataset)
    - [Where to publish my dataset ?](#where-to-publish-my-dataset-)
    - [Publish to a SSH server](#publish-to-a-ssh-server)
    - [Publish to GitHub](#publish-to-github)
      - [GitHub + SSH mirror to store the large files](#github--ssh-mirror-to-store-the-large-files)
    - [Publish on S3](#publish-on-s3)
    - [Publish on OSF](#publish-on-osf)

# Converting a dataset

This tutorial will guide you through the steps for the conversion of an existing dataset. We will use the [VanDam-Daylong dataset from HomeBank](https://homebank.talkbank.org/access/Public/VanDam-Daylong.html) as an example. 

## Set-up datalad and child-project

## Create a dataset

The first step is to create a new dataset named vandam-data :

```bash
datalad create vandam-data
cd vandam-data
```

So far, the dataset contains nothing but hidden files:

```bash
$ ls -A
.datalad	.git		.gitattributes
```

Now, we would like to get the data from https://, convert it to our standards, and then publish it.

## Gather and sort the files

The first step is to create the directories:

```bash
mkdir metadata
mkdir -p recordings/raw
mkdir annotations
mkdir extra
touch extra/.gitignore # Make sure the directory is present even though it's empty
```

Then, download the original data-set from HomeBank.

The audio first:

```bash
wget https://media.talkbank.org/homebank/Public/VanDam-Daylong/BN32/BN32_010007.mp3 -O recordings/raw/BN32_010007.mp3
```

Now let's get the annotations.

```bash
wget https://homebank.talkbank.org/data/Public/VanDam-Daylong.zip -O VanDam-Daylong.zip
unzip VanDam-Daylong.zip
rm VanDam-Daylong.zip
```

Let's explore the content of VanDam-Daylong:

```bash
$ ls -R VanDam-Daylong
0metadata.cdc	BN32

VanDam-Daylong/BN32:
0its		BN32_010007.cha

VanDam-Daylong/BN32/0its:
e20100728_143446_003489.its
```

 - `0metadata.cdc1` looks like some metadata file, so we will move it to `metadata/` :

```bash
mv VanDam-Daylong/0metadata.cdc metadata/
```

 - `BN32_010007.cha` contains some transcriptions. Let's create a set of annotations `cha` and move it there :
  
```bash
mkdir -p annotations/cha/raw
mv VanDam-Daylong/BN32/BN32_010007.cha annotations/cha/raw
```

 - `e20100728_143446_003489.its` contains diarization and other information such as word counts. Let's create another set of annotations for it. And for the sake of consistency, we'll rename it `BN32_010007.its`.
 
 ```bash
mkdir -p annotations/its/raw
mv VanDam-Daylong/BN32/0its/e20100728_143446_003489.its annotations/its/raw/BN32_010007.its
```

Now we've got all the files. Let's try to run the validation on the dataset:

```bash
$ child-project validate .

Traceback (most recent call last):
  File "/Users/acristia/anaconda3/bin/child-project", line 8, in <module>
    sys.exit(main())
  File "/Users/acristia/anaconda3/lib/python3.7/site-packages/ChildProject/cmdline.py", line 241, in main
    args.func(args)
  File "/Users/acristia/anaconda3/lib/python3.7/site-packages/ChildProject/cmdline.py", line 39, in validate
    errors, warnings = project.validate(args.ignore_files)
  File "/Users/acristia/anaconda3/lib/python3.7/site-packages/ChildProject/projects.py", line 102, in validate
    self.read()
  File "/Users/acristia/anaconda3/lib/python3.7/site-packages/ChildProject/projects.py", line 86, in read
    self.children = self.ct.read(lookup_extensions = ['.csv', '.xls', '.xlsx'])
  File "/Users/acristia/anaconda3/lib/python3.7/site-packages/ChildProject/tables.py", line 65, in read
    raise Exception("could not find table '{}'".format(self.path))
Exception: could not find table './metadata/children'
```

The validation fails, because the metadata is missing. We need to store the metadata about the children and the recordings according as specified in the [documentation](https://laac-lscp.github.io/ChildRecordsData/FORMATTING.html#metadata).


## Create the metadata

Let's start with the recordings metadata. `metadata/recordings.csv` should at least have the following columns: experiment, child_id, date_iso, start_time, recording_device_type, filename.
The .its file contains (`annotations/its/raw/BN32_010007.its`) precious information about when the recording started:

```xml
<Recording num="1" startClockTime="2010-07-24T11:58:16Z" endClockTime="2010-07-25T01:59:20Z" startTime="PT0.00S" endTime="PT50464.24S">
```

 Make sure that `metadata/recordings.csv` contains the following text:

```
experiment,child_id,date_iso,start_time,recording_device_type,filename
vandam-daylong,1,2010-07-24,11:58,lena,BN32_010007.mp3
```

(we have decided that the only child of the dataset should have ID '1')

Now the children metadata.
The only fields that are required are: experiment, child_id and child_dob.
The .its file also contains some information about the child:

```xml
<ChildInfo algorithmAge="P12M" gender="F" />
```

She was a 12 month old girl at the time of the recording. We can thus assign her a calculated date of birth: 2009-07-24. We will set `dob_criterion` to "extrapolated" to keep track of the fact that the date of birth was calculated from the approximate age at recording. We will also set `dob_accuracy` to 'month' for that child.


This is what `metadata/children.csv` should look like:

```
experiment,child_id,child_dob,dob_criterion,dob_accuracy
vandam-daylong,1,2009-07-24,extrapolated,month
```

We can now make sure that they are no errors by running the validation command again:

```bash
child-project validate .
```

No error occurs.

## Save the changes locally

A DataLad dataset is essentially a git repository, with the large files being handled by git-annex.
Some of the files (usually the small, text files such as metadata and scripts) ought to be versionned with git, and the larger files or binary files should be stored in the *annex*.

The rules to decide what files should be stored which way can be set in the `.gitattributes` file. You should fill it will the following content:

```
* annex.backend=MD5E
**/.git* annex.largefiles=nothing
scripts/* annex.largefiles=nothing
metadata/* annex.largefiles=nothing
recordings/converted/* annex.largefiles=((mimeencoding=binary))
```

These rules will version all the files under `scripts/` and `metadata/`, as well as the text files inside of `recordings/converted/`. By default, the other files will be put in the annex.

The changes can now be saved. This can be done with [datalad save](http://docs.datalad.org/en/stable/generated/man/datalad-save.html). `datalad save` is equivalent to a combination of `git add` and `git commit` in one go. It decides, based on the rules in `.gitattributes`, whether to store files with git or git-annex.

```
datalad save . -m "first commit"
```

However, so far, your changes remain local, and your dataset still needs to be published into a *sibling* to be shared with others.

## Processing

You can do some processing on the dataset. For instance, you can compute the duration of the recording, and update the metadata with this information. This is easily done with:

```bash
child-project compute-durations .
```

Now `metadata/recordings.csv` became:

```bash
$ cat metadata/recordings.csv 
experiment,child_id,date_iso,start_time,recording_device_type,filename,duration
vandam-daylong,1,2010-07-24,11:58,lena,BN32_010007.mp3,50464.512
```

You can also convert and index the its annotation:

```bash
child-project import-annotations . --set its \
  --recording_filename BN32_010007.mp3 \
  --time_seek 0 \
  --range_onset 0 \
  --range_offset 50464.512 \
  --raw_filename BN32_010007.its \
  --format its
```

And save the changes again:

```bash
datalad save . -m "its"
```

## Publish the dataset

### Where to publish my dataset ?

DataLad allows you to publish your datasets on a very wide range of platforms, each having their own advantages and limitations. It is also possible to publish to several platforms, as we do with our own datasets.

The table below summarises the features of a few storage supports. The solutions described here are by no mean exhaustive, but they are easy to generalize.

 - Platforms that support Git store the .git files and will allow you to clone the datasets from them with `datalad install`
 - Platforms that support Large Files will allow you to store and distribute the large or binary files that are stored with git-annex instead of the regular git files (such as scripts and metadata)

It is necessary to use a platform or a combination of platforms that supports both.

<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-amwm" colspan="3">Supports</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-fymr">Host</td>
    <td class="tg-7btt">Git</td>
    <td class="tg-7btt">Large Files</td>
    <td class="tg-7btt">Encryption</td>
  </tr>
  <tr>
    <td class="tg-0pky">SSH server</td>
    <td class="tg-0pky">Yes</td>
    <td class="tg-0pky">Yes</td>
    <td class="tg-0pky">No ?</td>
  </tr>
  <tr>
    <td class="tg-0pky">GitHub</td>
    <td class="tg-0pky">Yes</td>
    <td class="tg-0pky">No</td>
    <td class="tg-0pky">No</td>
  </tr>
  <tr>
    <td class="tg-0pky">Amazon S3</td>
    <td class="tg-0pky">No</td>
    <td class="tg-0pky">Yes</td>
    <td class="tg-0pky">Yes</td>
  </tr>
  <tr>
    <td class="tg-0lax">OSF.io</td>
    <td class="tg-0lax">Yes</td>
    <td class="tg-0lax">No*</td>
    <td class="tg-0lax">No</td>
  </tr>
</tbody>
</table>

### Publish to a SSH server

If you have access to a SSH server with enough storage capacity, you can use it to store and share the dataset.
This is done with the [datalad create-sibling](http://docs.datalad.org/en/stable/generated/man/datalad-create-sibling.html) command:

```bash
datalad create-sibling [-h] [-s [NAME]] [--target-dir PATH] [--target-url URL] [--target-pushurl URL] [--dataset DATASET] [-r] [-R LEVELS] [--existing MODE] [--shared {false|true|umask|group|all|world|everybody|0xxx}] [--group GROUP] [--ui {false|true|html_filename}] [--as-common-datasrc NAME] [--publish-by-default REFSPEC] [--publish-depends SIBLINGNAME] [--annex-wanted EXPR] [--annex-group EXPR] [--annex-groupwanted EXPR] [--inherit] [--since SINCE] [SSHURL]
```

For instance, you can create it (this is only to be done once) by issuing:


```bash
datalad create-sibling -s cluster --annex-wanted 'include=*' <ssh-server>:/remote/path/to/the/dataset
```

`cluster` is the name of the sibling, and `<ssh-server>:/remote/path/to/the/dataset` is the SSH url of its destination.
`--annex-wanted 'include=*'` implies that all large files will be published to this sibling by default.

Once the sibling has been created, the changes can be published:

```bash
datalad push --to cluster
```

That's it! People can now get your data from:

```bash
datalad install <ssh-server>:/remote/path/to/the/dataset
```

If `--annex-wanted` had not been set to `'include=*'`, the large files (i.e. annexed files) would not be published unless you asked for it explicitly with the `--data` flag:

```bash
datalad push --to cluster --data anything
```


### Publish to GitHub

You first need to create the repository, which can be done in a straightforward way from the command line with [datalad create-sibling-github](http://docs.datalad.org/en/stable/generated/man/datalad-create-sibling-github.html):

```bash
datalad create-sibling-github [-h] [--dataset DATASET] [-r] [-R LEVELS] [-s NAME] [--existing MODE] [--github-login NAME] [--github-organization NAME] [--access-protocol {https|ssh}] [--publish-depends SIBLINGNAME] [--private] [--dryrun] REPONAME
```

For instance:

```
datalad create-sibling-github -s origin --access-protocol ssh vandam-daylong-demo
```

`origin` will be the local name of the sibling, and `vandam-daylong-demo` the name of the GitHub repository.
Once the sibling has been created, you can publish the changes with [datalad push](http://docs.datalad.org/en/stable/generated/man/datalad-push.html):

```bash
datalad push --to origin
```

You should get a repository identical to [this one](https://github.com/LAAC-LSCP/vandam-daylong-demo). 

Users can now install your dataset from GitHub:

```
datalad install https://github.com/LAAC-LSCP/vandam-daylong-demo.git
```

PS: we recommand that you do `git push --set-upstream origin` to set upstream to the GitHub sibling. Users who install your dataset will not need to do this. 

#### GitHub + SSH mirror to store the large files

Now, let's assume you have already created a SSH sibling as well for your dataset, and that it is named `cluster`.
You can make sure that all changes to `github` are published to `cluster` as well, by setting the `publish-depends` property of the github sibling:

```bash
datalad siblings configure -s origin --publish-depends cluster
```

Now, `datalad push --to origin` will publish the changes to both `cluster` and `github`.

However, when the users install your dataset from GitHub, they will not have access to the `cluster` sibling unless you make it available to them, which can be done this way :


```bash
git annex 
git annex initremote cluster type=git location=ssh://cluster.com/path/to/the/repository autoenable=true
git annex enableremote cluster
git remote add origin git@github.com:LAAC-LSCP/vandam-daylong-demo.git
```


### Publish on S3

You might not have access to a SSH server with enough storage capacity, or you might just not want to setup SSH keys to every user of your dataset. Fortunately, DataLad supports a [large number of storage providers](https://git-annex.branchable.com/special_remotes/) such as: Amazon S3, Dropbox, Google Cloud Storage, Microsoft Azure Blob Storage, as well as any FTP/SFTP server. Here, we will give instructions for Amazon S3. 

Like other *git annex special remotes*, Amazon S3 will not support the git files, only the large files. But you can use it along with GitHub or GitLab.

*For the sake of simplicity, we will not use encryption here, but git annex implements several [encryption schemes](https://git-annex.branchable.com/encryption/) which are easy to use.*

First, store your AWS credentiels into your environment variables, like this:

```bash
export AWS_ACCESS_KEY_ID="08TJMT99S3511WOZEP91"
export AWS_SECRET_ACCESS_KEY="s3kr1t"
```

You are now readyto create the s3 sibling. This is done directly through git-annex this time:

```bash
git annex initremote s3 chunk=100MiB type=S3 encryption=none datacenter=eu-west-3 embedcreds=no signature=v4
```

You can now publish the data with:

```bash
datalad push --to s3 --data anything
```

(Optional) You can set the S3 sibling to require that all large files should be stored on it:

```bash
datalad siblings configure -s s3 --annex-wanted 'include=*'
```

This will let DataLad publish all the large files automatically without setting `--data`:

```bash
datalad push --to s3
```

Let's assume your users will install the dataset from a GitHub repository. You should publish the information about the newly created S3 sibling on GitHub, which can be done with (provided you have set up your GitHub repository as described in the previous section):

```bash
datalad push --to github
```

Now, users will be able to get the data by issuing the following commands:

```bash
datalad install git@github.com:<your-username>/vandam-daylong-demo.git
git annex enableremote s3
datalad get *
```

With this configuration, they will need to setup their AWS credentials as you did. [But it is possible to configure the sibling so that the credentials are encrypted](https://git-annex.branchable.com/tips/using_Amazon_S3/) and stored in the repository, so all users with authorized private keys will be able to get the data without this step.

### Publish on OSF

DataLad has an [extension](http://docs.datalad.org/projects/osf/en/latest/generated/man/datalad-create-sibling-osf.html) to publish data on the [Open Science Framework](https://osf.io/).

This extension supports the following modes:

|   **Mode**   | **datalad install** | **large files** | **history** | **older files** | **human-readable project** |
|:------------:|:-------------------:|:---------------:|:-----------:|:---------------:|----------------------------|
| `annex`      |         Yes         |       Yes       |     Yes     |       Yes       |             No             |
| `export`     |         Yes         |       Yes       |     Yes     |        No       |             Yes            |
| `gitonly`    |         Yes         |        No       |     Yes     |        No       |             No             |
| `exportonly` |          No         |       Yes       |      No     |       Yes       |             Yes            |

The first step is to install the extension: 

```
pip install datalad-osf --upgrade
```

We decide to use the `export` mode - but you can decide which best suits your needs from the table above. We can now create the sibling:

```
datalad create-sibling-osf --title "VanDam Demo" \
  --mode export \
  -s osf \
  --category data \
  --tag reproducibility \
  --public
```

You will be prompted your credentials in the process, which will require access tokens to be created [from your osf.io account](https://osf.io/settings/tokens). 

And finally we can push the data. This is done in two steps: 

 1. publishing the .git files so people can clone the dataset directly from OSF

```
datalad push --to osf
```

 2. exporting a human-readable snapshot of the files to OSF

```
git-annex export HEAD --to osf-storage
```

