

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

Can be either `children.csv`, `children.xls` or `children.xslx`.

| Name | Description | Required ? | Format |
|------|-------------|------------|--------|
{% for column in children -%}
| {{column.name}} | {{column.description}} | {{'required' if column.required else 'optional'}}
{%- if column.datetime %} | `{{column.datetime}}` (date/time) |
{%- elif column.regex %} | `{{column.regex}}` (regex) |
{%- elif column.filename %} | filename |
{%- else %} | - |
{%- endif %}
{% endfor %}



## recording notebook

Can be either `recordings/recordings.csv`, `recordings/recordings.xls` or `recordings/recordings.xslx`.

| Name | Description | Required ? | Format |
|------|-------------|------------|--------|
{% for column in recordings -%}
| {{column.name}} | {{column.description}} | {{'required' if column.required else 'optional'}}
{%- if column.datetime %} | `{{column.datetime}}` (date/time) |
{%- elif column.regex %} | `{{column.regex}}` (regex) |
{%- elif column.filename %} | filename |
{%- else %} | - |
{%- endif %}
{% endfor %}