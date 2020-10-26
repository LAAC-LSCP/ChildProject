

- [Source data formatting guidelines](#source-data-formatting-guidelines)
  - [Raw data tree](#raw-data-tree)
  - [children notebook](#children-notebook)
  - [recording notebook](#recording-notebook)
- [Annotations formatting](#annotations-formatting)
  - [Annotations format](#annotations-format)
  - [Annotations index](#annotations-index)
  - [Annotation importation input format](#annotation-importation-input-format)

# Source data formatting guidelines

## Raw data tree

```
project
│   children.csv
│
└───recordings
│   │   recordings.csv
│   │   recording1.wav
│
└───raw_annotations
│   │   child1.rttm
│   │   child1_3600.TextGrid
│
└───annotations
│   │   annotations.csv
│
└───extra
    │   notes.txt
``` 

## children notebook

Can be either `children.csv`, `children.xls` or `children.xslx`.

| Name | Description | Required ? | Format |
|------|-------------|------------|--------|
{% for column in children -%}
| **{{column.name}}** | {{column.description}} | {{'**required**' if column.required else 'optional'}}
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
| **{{column.name}}** | {{column.description}} | {{'**required**' if column.required else 'optional'}}
{%- if column.datetime %} | `{{column.datetime}}` (date/time) |
{%- elif column.regex %} | `{{column.regex}}` (regex) |
{%- elif column.filename %} | filename |
{%- else %} | - |
{%- endif %}
{% endfor %}

# Annotations formatting

## Annotations format

The package provides functions to convert any annotation into the following csv format, with one row per segment :

| Name | Description | Format |
|------|-------------|--------|
{% for column in annotation_segments -%}
| **{{column.name}}** | {{column.description}}
{%- if column.datetime %} | `{{column.datetime}}` (date/time) |
{%- elif column.regex %} | `{{column.regex}}` (regex) |
{%- elif column.filename %} | filename |
{%- else %} | - |
{%- endif %}
{% endfor %}

## Annotations index

Annotations are indexed in one unique dataframe located at `/annotations/annotations.csv`, with the following format :

| Name | Description | Format |
|------|-------------|--------|
{% for column in annotations -%}
| **{{column.name}}** | {{column.description}}
{%- if column.datetime %} | `{{column.datetime}}` (date/time) |
{%- elif column.regex %} | `{{column.regex}}` (regex) |
{%- elif column.filename %} | filename |
{%- else %} | - |
{%- endif %}
{% endfor %}

## Annotation importation input format

The annotations importation script and function take a dataframe of the following format as an input :

| Name | Description | Required ? | Format |
|------|-------------|------------|--------|
{% for column in input_annotations -%}
| **{{column.name}}** | {{column.description}} | {{'**required**' if column.required else 'optional'}}
{%- if column.datetime %} | `{{column.datetime}}` (date/time) |
{%- elif column.regex %} | `{{column.regex}}` (regex) |
{%- elif column.filename %} | filename |
{%- else %} | - |
{%- endif %}
{% endfor %}

