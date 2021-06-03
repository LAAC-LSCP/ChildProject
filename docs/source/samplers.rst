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

 - `segments_YYYYMMDD_HHMMSS.csv`, a CSV dataframe of all sampled segments, with three columns: ``recording_filename``, ``segment_onset`` and ``segment_offset``.
 - `parameters_YYYYMMDD_HHMMSS.yml`, a Yaml file with all the parameters that were used to generate the samples.

If the folder `destination` does not exist, it is automatically created in the process.

Several samplers are implemented in our package, which are listed below.
The samples can then feed downstream pipelines such as the :ref:`zooniverse` pipeline.

.. clidoc::

   child-project sampler --help

Periodic sampler
~~~~~~~~~~~~~~~~

Draw segments from the recordings, periodically

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
(for a uniform sampling across recordings), ``session_id`` (to sample uniformly across
observation days), or ``child_id`` (to sample uniformly across children).


.. clidoc::

   child-project sampler /path/to/dataset /path/to/destination energy-detection --help

High-Volubility sampler
~~~~~~~~~~~~~~~~~~~~~~~

(TODO)

.. clidoc::

   child-project sampler /path/to/dataset /path/to/destination high-volubility --help
