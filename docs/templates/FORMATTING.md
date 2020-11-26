

- [Datasets structure](#datasets-structure)
  - [Dataset tree](#dataset-tree)
  - [Metadata](#metadata)
    - [children notebook](#children-notebook)
    - [recording notebook](#recording-notebook)
- [Annotations](#annotations)
  - [Annotations format](#annotations-format)
  - [Annotations index](#annotations-index)
  - [Annotation importation input format](#annotation-importation-input-format)

# Datasets structure

Datasets must pass the [the validation routine](https://laac-lscp.github.io/ChildRecordsData/#validate-raw-data) with no error and as few warnings as possible before submission.

## Dataset tree

All datasets should have this structure :

```
project
│   
│
└───metadata
│   │   children.csv
│   │   recordings.csv
│   │   annotations.csv
|
└───recordings
│   │   recordings.csv
│   │   recording1.wav
│
└───raw_annotations
│   │   child1.rttm
│   │   child1_3600.TextGrid
│
└───extra
    │   notes.txt
```

The children and recordings notebooks should be formatted according to the standards detailed right below.

## Metadata


### children notebook

Located in the `metadata` folder. Can be named either `children.csv`, `children.xls` or `children.xslx`.

| Name | Description | Required ? | Format |
|------|-------------|------------|--------|
{% for column in children -%}
| **{{column.name}}** | {{column.description}} | {{'**required**' if column.required else 'optional'}}
{%- if column.datetime %} | `{{column.datetime}}` (date/time) |
{%- elif column.regex %} | `{{column.regex}}` (regex) |
{%- elif column.function %} | {{column.function.__name__}} (function) |
{%- elif column.choices %} | {{column.choices|join(", ")}} |
{%- elif column.filename %} | filename |
{%- else %} | - |
{%- endif %}
{% endfor %}

### recording notebook

Located in the `metadata` folder.  Can be named either `recordings.csv`, `recordings.xls` or `recordings.xslx`.

| Name | Description | Required ? | Format |
|------|-------------|------------|--------|
{% for column in recordings -%}
| **{{column.name}}** | {{column.description}} | {{'**required**' if column.required else 'optional'}}
{%- if column.datetime %} | `{{column.datetime}}` (date/time) |
{%- elif column.regex %} | `{{column.regex}}` (regex) |
{%- elif column.function %} | {{column.function.__name__}} (function) |
{%- elif column.choices %} | {{column.choices|join(", ")}} |
{%- elif column.filename %} | filename |
{%- else %} | - |
{%- endif %}
{% endfor %}

# Annotations

## Annotations format

The package provides functions to convert any annotation into the following csv format, with one row per segment :

| Name | Description | Format |
|------|-------------|--------|
{% for column in annotation_segments -%}
| **{{column.name}}** | {{column.description}}
{%- if column.datetime %} | `{{column.datetime}}` (date/time) |
{%- elif column.regex %} | `{{column.regex}}` (regex) |
{%- elif column.function %} | {{column.function.__name__}} (function) |
{%- elif column.choices %} | {{column.choices|join(", ")}} |
{%- elif column.filename %} | filename |
{%- else %} | - |
{%- endif %}
{% endfor %}

## Annotations index

Annotations are indexed in one unique dataframe located at `/metadata/annotations.csv`, with the following format :

| Name | Description | Format |
|------|-------------|--------|
{% for column in annotations -%}
| **{{column.name}}** | {{column.description}}
{%- if column.datetime %} | `{{column.datetime}}` (date/time) |
{%- elif column.regex %} | `{{column.regex}}` (regex) |
{%- elif column.function %} | {{column.function.__name__}} (function) |
{%- elif column.choices %} | {{column.choices|join(", ")}} |
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
{%- elif column.function %} | {{column.function.__name__}} (function) |
{%- elif column.choices %} | {{column.choices|join(", ")}} |
{%- elif column.filename %} | filename |
{%- else %} | - |
{%- endif %}
{% endfor %}

