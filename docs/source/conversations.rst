Conversations summary extraction
--------------------------------

Overview
~~~~~~~~

This package allows to extract descriptive statistics on identified conversations in recordings. The set used for the extraction must contain conversation annotations which is to say have the columns ``segment_onset``, ``segment_offset``, ``speaker_type`` and ``conv_count``.
The :ref:`derive-annotations` pipeline can be used to derivate conversation annotations from diarized annotations; we recommend using this on vtc automated annotations to have automated conversation annotations.
A csv file containing the statistics is produced along with a YML parameter file storing all the options used for the extractions

.. clidoc::

   child-project conversations-summary --help

The conversation extraction will always have the following columns:

.. csv-table::
    :header: "column", "info"
    :widths: 19, 30
    :stub-columns: 1
    
    conversation_onset, start of the conversation (ms) inside the recording
    conversation_offset, end of the conversation (ms) inside the recording
    voc_count, number of vocalizations inside the conversation
    conv_count, identifier of the conversation (unique across the recording)
    interval_last_conv, interval (ms) between the end of previous conversation end start of the current conversation (NA for first)
    recording_filename, recording of the conversation
    
The list of supported functions is shown below:

.. _list-conversation-metrics:

.. custom-table::
    :header: list-conversation-metrics

Standard Conversations
~~~~~~~~~~~~~~~~~~~~~~

The Standard pipeline will extract a list of usual metrics that can be obtained from conversations. Using this pipeline with a set containing conversation annotations
 will output:

.. csv-table::
    :header: "metric", "name", "speaker"
    :widths: 19, 19, 19
    :stub-columns: 1

    "who_initiated", "initiator",
    "who_finished", "finisher",
    "voc_total_dur", "total_duration_of_vocalisations",
    "voc_speaker_count", "CHI_voc_count", 'CHI'
    "voc_speaker_count", "FEM_voc_count", 'FEM'
    "voc_speaker_count", "MAL_voc_count", 'MAL'
    "voc_speaker_count", "OCH_voc_count", 'OCH'
    "voc_speaker_dur", "CHI_voc_dur", 'CHI'
    "voc_speaker_dur", "FEM_voc_dur", 'FEM'
    "voc_speaker_dur", "MAL_voc_dur", 'MAL'
    "voc_speaker_dur", "OCH_voc_dur", 'OCH'

.. clidoc::

   child-project conversations-summary /path/to/dataset output.csv standard --help

Custom Conversations
~~~~~~~~~~~~~~~~~~~~

.. _list-structure:

The Custom conversations pipeline allows you to provide your own list of desired metric to the pipeline to be extracted.
The list must be in a csv file containing the following colums:

- callable (required) : name of the metric to extract, see :ref:`the list <list-conversation-metrics>`
- name (required) : name to use in the resulting metrics. If none is given, a default name will be used. Use this to extract the same metric for different sets and avoid name clashes.
- <argument> (depending on the requirements of the metric you chose) : For each required argument of a metric, add a column of that argument's name.

This is an example of a csv file we use to extract conversation metrics.
We want to extract who initiated the conversation, who finished it, the list of speakers involved and the percentage of speech produced by the target child (CHI) in each conversation and the same for female adult speakers (FEM).
So we write 5 lines, one for each metric, we give the reference to the metric (as they are in the table above), the name that we want in the final output, and for some of them, the required argument(s).

.. csv-table::
    :header: "metric", "name", "speaker"
    :widths: 20, 10, 20

    who_initiated, "initiator",
    who_finished, "finisher",
    participants, "participants",
    voc_dur_contribution, chi_dur_contrib, CHI
    voc_dur_contribution, fem_dur_contrib, FEM

.. clidoc::

    child-project conversations-summary /path/to/dataset output.csv custom --help

Conversations extraction from parameter file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To facilitate the extraction of conversations, one can simply use an exhaustive yml parameter file to launch a new extraction.
This file has the exact same structure as the one produced by the pipeline. So you can use the output parameter file of a previous extraction to rerun the same analysis.

.. clidoc::

    child-project conversations-specification --help
