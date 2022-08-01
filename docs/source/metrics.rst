Metrics extraction
------------------

Overview
~~~~~~~~

This package allows to extract metrics that are commonly used from annotations
produced by the LENA or other pipelines.
A csv file containing the metrics is produced along with a YML parameter file storing all the options used

.. clidoc::

   child-project metrics --help

The Period option aggregates vocalizations for each time-of-the-day-unit based on a period specified by the user.
For instance, if the period is set to ``15Min`` (i.e. 15 minutes), vocalization rates will be reported for each
recording and time-unit (e.g. 09:00 to 09:15, 09:15 to 09:30, etc.).

The output dataframe has :math:`r \times p` rows, where :math:`r` is the amount of recordings (or children if the ``-by`` option is set to ``child_id`` etc.), and :math:`p` is the 
amount of time-bins per day (i.e. :math:`24 \times 4=96` for a 15-minute period).

The output dataframe includes a ``period_start`` and a ``period_end`` columns that contain the onset and offset of each time-unit in HH:MM:SS format.
The ``duration_<set>`` columns contain the total amount of annotated time covering each time-bin and each set, in milliseconds.

If ``--by`` is set to e.g. ``child_id``, then the values for each time-bin will be the average rates across
all the recordings of every child.

The list of supported metrics is shown below:

.. warning::

    Be aware that numerous metrics are rates (every metric ending with 'ph' is) and not absolute counts!
    This can differ with results from other methods of extraction (e.g. LENA metrics).
    Rates are expressed in counts/hour (for events) or in milliseconds/hour (for durations).

.. _list-metrics:
.. custom-table::
    :header: list-metrics

LENA Metrics
~~~~~~~~~~~~

.. clidoc::

   child-project metrics /path/to/dataset output.csv lena --help

ACLEW Metrics
~~~~~~~~~~~~~

.. clidoc::

    child-project metrics /path/to/dataset output.csv aclew --help

Custom metrics
~~~~~~~~~~~~~~

.. _list_structure:
The Custom metrics pipeline allows you to provide your own list of desired metrics to the pipeline to be extracted.
The list must be in a csv file containing the following colums:
 - callable (required) : name of the metric to extract, see :ref:`list-metrics`
 - set (required) : name of the set to extract from, make sure this annotations set is capable (has the required information) to extract this specific metric
 - name (optional) : name to use in the resulting metrics. If none is given, a default name will be used. Use this to extract the same metric for different sets and avoid name clashes.
 - <argument> (depending on the requirements of the metric you chose) : For each required argument of a metric, add a column of that argument's name.

This is an example of a csv file we use to extract metrics.
We want to extract the number of vocalizations per hour of the key child (CHI), male adult (MAL) and female adult (FEM) on 2 different sets to compare their result.
So we write 3 lines per set (vtc and its), each having a different speaker and we also give each metric an explicit name because the default names `voc_chi_ph`, `voc_mal_ph` and `voc_fem_ph` would have clashed between the 2 sets.
Additionaly, we extract linguistic proportion on number of vocalizations and on duration separately from the vcm set. the default names won't clash and no speaker is needed (linguistic proportion is used on CHI) so we leave those columns empty.

.. csv-table::
    :header: "callable", "set", "name", "speaker"
    :widths: 20, 10, 20,20

    voc_speaker_ph,vtc,voc_chi_ph_vtc,CHI
    voc_speaker_ph,vtc,voc_mal_ph_vtc,MAL
    voc_speaker_ph,vtc,voc_fem_ph_vtc,FEM
    voc_speaker_ph,its,voc_chi_ph_its,CHI
    voc_speaker_ph,its,voc_mal_ph_its,MAL
    voc_speaker_ph,its,voc_fem_ph_its,FEM
    lp_n,vcm,,
    lp_dur,vcm,,

.. clidoc::

    child-project metrics /path/to/dataset output.csv custom --help

Metrics from parameter file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To facilitate the extraction of metrics, one can simply use an exhaustive yml parameter file to launch a new extraction.
This file has the exact same structure as the one produced by the pipeline. So you can use an output parameter file to rerun the same analysis.

.. clidoc::

    child-project metrics-specification --help