

## Raw data tree

```
project
│   children.csv
│
└───recordings
│   │   recordings.csv
│   │   recording1.wav
│
└───extra
    │   notes.txt
``` 

## children notebook

| Name | Description | Required ? | Format |
|------|-------------|------------|--------|
{% for column in children %}| {{column.name}} | {{column.description}} | {{'required' if column.required else 'optional'}} | none |
{% endfor %}



## recording notebook

| Name | Description | Required ? | Format |
|------|-------------|------------|--------|
{% for column in recordings %}| {{column.name}} | {{column.description}} | {{'required' if column.required else 'optional'}} | none |
{% endfor %}