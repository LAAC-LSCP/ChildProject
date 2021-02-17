

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

ChildRecordsData assumes your data is structured in a specific way before it is imported. This structure is necessary to check, for instance, that there are no unreferenced files, and no referenced files that are actually missing. The data curator therefore needs to organize their data in a specific way (respecting the dataset tree, with all specified metadata files, and all specified columns within the metadata files) before their data can be imported.

To be imported, datasets must pass the [the validation routine](https://laac-lscp.github.io/ChildRecordsData/#validate-raw-data) with no error. We also recommend you pay attention to the warnings, and try to sort as many of those out as possible before submission.

## Dataset tree

All datasets should have this structure before import (so you need to organize your files into this structure):

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
│   └───raw
│   │   │   recording1.wav
│
└───annotations
│   └───vtc
│   │   └───raw
│   │   │   │   child1.rttm
│   └───annotator1
│   │   └───raw
│   │   │   │   child1_3600.TextGrid
│
└───extra
    │   notes.txt
```

The children and recordings notebooks should be formatted according to the standards detailed right below.

## Metadata


### children notebook

The children dataframe needs to be saved at `metadata/children.csv`.

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

The recordings dataframe needs to be saved at `metadata/recordings.csv`.

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

