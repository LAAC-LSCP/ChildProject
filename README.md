# ChildRecordsData

## Installation

```
python3.6 -m venv ~/ChildProjectVenv

git clone https://github.com/lucasgautheron/ChildRecordsData.git
cd ChildRecordsData
source ~/ChildProjectVenv/bin/activate
pip install -r requirements.txt
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