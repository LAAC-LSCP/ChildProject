# ChildRecordsData

## Formatting data

See the [formatting instructions](http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html)

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

#### Multi-core audio conversion with sbatch :

1. create `convert.sh`
```bash
#!/bin/bash
python convert.py --source ../data/namibia/ --name mp --format WAV --codec pcm_s16le --sampling 16000 --threads 4
```
2. run `$ chmod +x convert.sh`
3. run `$ sbatch --mem=64G --time=5:00:00 --cpus-per-task=4 --ntasks=1 -o namibia.txt ./convert.sh`
