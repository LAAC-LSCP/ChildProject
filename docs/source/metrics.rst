Metrics extraction
----------------

Overview
~~~~~~~~

This package allows to extract metrics that are commonly used from annotations
produced by the LENA or other pipelines.

.. clidoc::

   child-project metrics --help

The list of supported metrics is shown below:

.. csv-table::
    :header: "Variable", "Description", "pipelines"
    :widths: 15, 50, 5

    voc_fem/mal/och_ph,number of vocalizations by different talker types per hour,"ACLEW,LENA,Period"
    voc_dur_fem/mal/och_ph,total duration of vocalizations by different talker types in seconds per hour,"ACLEW,LENA,Period"
    avg_voc_dur_fem/mal/och,average vocalization length (conceptually akin to MLU) by different talker types,"ACLEW,LENA,Period"
    wc_adu_ph,adult word count (collapsing across males and females),"ACLEW,LENA"
    wc_fem/mal_ph,adult word count by different talker types,"ACLEW,LENA"
    sc_adu_ph,adult syllable count (collapsing across males and females),ACLEW
    sc_fem/mal_ph,adult syllable count by different talker types,ACLEW
    pc_adu_ph,adult phoneme count (collapsing across males and females),ACLEW
    pc_fem/mal_ph,adult phoneme count by different talker types,ACLEW
    freq_n,frequency of child voc out of all vocs based on number of vocalizations,"ACLEW,LENA"
    freq_dur,frequency of child voc out of all vocs based on duration of vocalizations,"ACLEW,LENA"
    cry_voc_chi_ph,number of child vocalizations that are crying,"ACLEW,LENA"
    can_voc_chi_ph,number of child vocs that are canonical,ACLEW
    non_can_vpc_chi_ph,number of child vocs that are non-canonical,ACLEW
    sp_voc_chi_ph,number of child vocs that are speech-like (can+noncan for ACLEW),"ACLEW,LENA"
    cry_voc_dur_chi_ph,total duration of child vocalizations that are crying,"ACLEW,LENA"
    can_voc_dur_chi_ph,total duration of child vocs that are canonical,ACLEW
    non_can_voc_dur_chi_ph,total duration of child vocs that are non-canonical,ACLEW
    sp_voc_dur_chi_ph,total duration of child vocs that are speech-like (can+noncan for ACLEW),"ACLEW,LENA"
    avg_cry_voc_dur_chi,average duration of child vocalizations that are crying,"ACLEW,LENA"
    avg_cran_voc_dur_chi,average duration of child vocs that are canonical,ACLEW
    avg_non_can_voc_dur_chi,average duration of child vocs that are non-canonical,ACLEW
    avg_sp_voc_dur_chi,average duration of child vocs that are speech-like (can+noncan for ACLEW),"ACLEW,LENA"
    lp_n,linguistic proportion = (speech)/(cry+speech) based on number of vocalizations,"ACLEW,LENA"
    cp_n,canonical proportion = canonical /(can+noncan) based on number of vocalizations,ACLEW
    lp_dur,linguistic proportion = (speech)/(cry+speech) based on duration of vocalizations,"ACLEW,LENA"
    cp_dur,canonical proportion = canonical /(can+noncan) based on duration of vocalizations,ACLEW

.. note::

    Average rates are expressed in counts/hour (for events) or in seconds/hour (for durations).

LENA Metrics
~~~~~~~~~~~~

.. clidoc::

   child-project metrics /path/to/dataset output.csv lena --help

ACLEW Metrics
~~~~~~~~~~~~

.. clidoc::

    child-project metrics /path/to/dataset output.csv aclew --help

Period-aggregated metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

The Period Metrics pipeline aggregates vocalizations for each time-of-the-day-unit based on a period specified by the user.
For instance, if the period is set to ``15Min`` (i.e. 15 minutes), vocalization rates will be reported for each
recording and time-unit (e.g. 09:00 to 09:15, 09:15 to 09:30, etc.).

The output dataframe has :math:`r \times p` rows, where :math:`r` is the amount of recordings (or children if the ``-by`` option is set to ``child_id``), and :math:`p` is the 
amount of time-bins per day (i.e. :math:`24 \times 4=96` for a 15-minute period).

The output dataframe includes a ``period`` column that contains the onset of each time-unit in HH:MM:SS format.
The ``duration`` columns contains the total amount of annotations covering each time-bin, in milliseconds.

If ``--by`` is set to e.g. ``child_id``, then the values for each time-bin will be the average rates across
all the recordings of every child.

.. clidoc::

    child-project metrics /path/to/dataset output.csv period --help

..note::

    Average rates are expressed in seconds/hour regardless of the period.
