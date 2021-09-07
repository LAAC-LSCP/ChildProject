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

    voc_fem/mal/och_ph,number of vocalizations by different talker types,"ACLEW,LENA"
    voc_dur_fem/mal/och_ph,total duration of vocalizations by different talker types,"ACLEW,LENA"
    avg_voc_dur_fem/mal/och,average vocalization length (conceptually akin to MLU) by different talker types,"ACLEW,LENA"
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

LENA Metrics
~~~~~~~~~~~~

.. clidoc::

   child-project metrics /path/to/dataset output.csv lena --help

ACLEW Metrics
~~~~~~~~~~~~

.. clidoc::

    child-project metrics /path/to/dataset output.csv aclew --help

Period-aggregated metrics
~~~~~~~~~~~~~~~~~~~~~~~

.. clidoc::

    child-project metrics /path/to/dataset output.csv period --help