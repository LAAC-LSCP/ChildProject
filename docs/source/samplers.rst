.. _samplers:

Samplers
--------

Overview
~~~~~~~~

.. figure:: images/sampler_diagram.png
   :alt: Sampling recordings

   Sampling audio segments to be annotated with ChildProject.

A sampler draws segments from the recordings, according to the algorithm and the parameters defined by the user.
The sampler will produce two files into the `destination` folder :

 - ``segments_YYYYMMDD_HHMMSS.csv``, a CSV dataframe of all sampled segments, with three columns: ``recording_filename``, ``segment_onset`` and ``segment_offset``.
 - ``parameters_YYYYMMDD_HHMMSS.yml``, a Yaml file with all the parameters that were used to generate the samples.

If the folder `destination` does not exist, it is automatically created in the process.

Several samplers are implemented in our package, which are listed below.

The samples can then feed downstream pipelines such as the :ref:`zooniverse` pipeline or the :ref:`eaf-builder`.

.. clidoc::

   child-project sampler --help

All samplers have a few parameters in common:

- ``--recordings``, which sets the white-list of recordings to sample from
- ``--exclude``, which defines the portions of audio to exclude from the samples *after* sampling.

Periodic sampler
~~~~~~~~~~~~~~~~

Draw segments from the recordings periodically.

The ``--period`` argument is between the end of the previous segment until the start of the next. For example
length:60000(1min) period:3540000(59min) will sample the first minute of every hour whereas length:60000(1min)
period:3600000(1h) will sample the 1st min of the 1st hour, the 2nd min o the 2nd hour and so on.

The ``--by`` argument will group recordings to form a single timeline in which the periodicity defines the parts
to annotate, then those parts are extracted from the recordings of the group. this means that recordings following
each other will maintain continuity in sampling period if in the same session and sampling by session_id. It also means
concurrent recordings in the same session will have the same samples kept time/date wise regardless of shifts in start.
The default is to sample by 'recording_filename' which will simply periodicly sample each recording independently.

.. clidoc::

   child-project sampler /path/to/dataset /path/to/destination periodic --help

Vocalization sampler
~~~~~~~~~~~~~~~~~~~~

Draw segments from the recordings, targetting vocalizations from
specific speaker-type(s).

.. clidoc::

   child-project sampler /path/to/dataset /path/to/destination random-vocalizations --help

Energy-based sampler
~~~~~~~~~~~~~~~~~~~~

Draw segments from the recordings, targetting windows with energies
above some threshold.

This algorithm proceeds by segmenting the recordings into windows;
the energy of the signal is computed for each window (users have
the option to apply a band-pass filter to calculate the energy
in some frequency band).

Then, the algorithm samples as many windows as requested by the user
from the windows that have energies above some threshold.
The energy threshold is defined in term of energy quantile. By default,
it is set to 0.8, i.e, only the windows with the 20% highest energies are sampled from.

The sampling is performed unit by unit, where the unit is set through 
the ``--by`` option and can be any either ``recording_filename``
(to sample an equal amount of windows from each recording),
``session_id`` (to equally from each observing day),
or ``child_id`` (to sample equally from each child).


.. clidoc::

   child-project sampler /path/to/dataset /path/to/destination energy-detection --help

High-Volubility sampler
~~~~~~~~~~~~~~~~~~~~~~~

Return the top ``windows_count`` windows (of length ``windows_length``) with the highest volubility from each recording, as calculated from the metric ``metric``.

``metrics`` can be any of three values: words, turns, and vocs.

- The **words** metric sums the amount of words within each window. For LENA annotations, it is equivalent to **awc**.
- The **turns** metric (aka ctc) sums conversational turns within each window. It relies on **lena_conv_turn_type** for LENA annotations. For other annotations, turns are estimated as adult/child speech switches in close temporal proximity.
- The **vocs** metric sums utterances (for LENA annotations) or vocalizations (for other annotations) within each window. If ``metric="vocs"`` and ``speakers=['CHI']``, it is equivalent to the usual cvc metric (child vocalization counts).

.. clidoc::

   child-project sampler /path/to/dataset /path/to/destination high-volubility --help

Conversation sampler
~~~~~~~~~~~~~~~~~~~~

The conversation sampler returns the conversational blocks with the highest amount of turns (between adults and the key child).
The first step is the detection of conversational blocks.
Two consecutive vocalizations are considered part of the same conversational block if they are not separated
by an interval longer than a certain duration, which by default is set to 1000 milliseconds.

Then, the amount of conversational turns (by default, between the key child and female/male adults) is calculated for each conversational block.
The sampler returns, for each unit, the desired amount of conversations with the higher amount of turns.

This sampler, unlike the High-Volubility sampler, returns portions of audio with variable durations.
Fixed duration can still be achieved by clipping or splitting each conversational block.


.. clidoc::

   child-project sampler /path/to/dataset /path/to/destination conversations --help

.. note::

   This sampler ignores LENA's conversational turn types.
