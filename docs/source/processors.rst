Audio processors
----------------

Overview
~~~~~~~~

The package provides several tools for processing the recordings.

.. clidoc::

   child-project process --help

Basic audio conversion
~~~~~~~~~~~~~~~~~~~~~~

Converts all recordings in a dataset to a given encoding. Converted
audios are stored into ``recordings/converted/<profile-name>``.

.. clidoc::

   child-project process /path/to/dataset test basic --help

Example:

::

   child-project process /path/to/dataset 16kHz basic --format=wav --sampling=16000 --codec=pcm_s16le

We typically run the following, to split long sound files every 15
hours, because the software we use for human annotation (ELAN, Praat)
works better with audio that is maximally 15h long:

::

   child-project process /path/to/dataset 16kHz basic --split=15:00:00 --format=wav --sampling=16000 --codec=pcm_s16le

Processing can be restricted to a white-list of recordings only using the ``--recordings`` option:

::

   child-project process /path/to/dataset 16kHz basic --format=wav --sampling=16000 --codec=pcm_s16le --recordings audio1.wav audio2.wav

Values provided to this option should be existing ``recording_filename`` values in ``metadata/recordings.csv``.

The ``--skip-existing`` switch can be used to skip previously processed files.

Multi-core audio conversion with slurm on a cluster
===================================================

If you have access to a cluster with slurm, you can use a command like
the one below to batch-convert your recordings. Please note that you may
need to change some details depending on your cluster (eg cpus per
task). If needed, refer to the `slurm user
guide <https://slurm.schedmd.com/quickstart.html>`__

::

   sbatch --mem=64G --time=5:00:00 --cpus-per-task=4 --ntasks=1 -o namibia.txt child-project process --threads 4 /path/to/dataset 16kHz basic --split=15:00:00 --format=wav --sampling=16000 --codec=pcm_s16le

Vetting
~~~~~~~

The vetting pipeline mutes segments of the recordings provided by the user while preserving the duration of the audio files.
This technique can be used to remove speech that might contain confidential information before releasing the audio.

The input needs to be a CSV dataframe with the following columns: ``recording_filename``, ``segment_onset``, ``segment_onset``.
The timestamps need to be expressed in milliseconds.

.. clidoc::

    child-project process /path/to/dataset test vetting --help

Channel mapping
~~~~~~~~~~~~~~~

The channel mapping pipeline is meant to be used with multi-channel audio recordings,
such as those produced by the BabyLogger.
It allows to filter or to combine channels from the original recordings at your convenience.


.. clidoc::

   child-project process /path/to/dataset test channel-mapping --help

In mathematical terms, assuming the input recordings have :math:`n` channels
with signals :math:`s_{j}(t)`;
If the output recordings should have :math:`m` channels,
the user defines a matrix of weights :math:`w_{ij}` with :math:`m` rows and :math:`n` columns,
such as the signal of each output channel :math:`s'_{i}(t)` is:

.. math::

   s'_{i}(t) = \sum_{j=1}^n w_{ij} s_{j}(t)

The weights matrix is defined through the ``--channels`` parameters.

The weights for each output channel are separated by blanks.
For a given output channel, the weights of each input channels should be separated by commas.

For instance, if one would like to use the following weight matrix (which transforms
4-channel recordings into 2-channel audio):

.. math::

   \begin{pmatrix}
   0 & 0 & 1 & 1 \\ 
   0.5 & 0.5 & 0 & 0
   \end{pmatrix}

Then the correct values for the --channels parameters should be:

.. code-block:: bash

   --channels 0,0,1,1 0.5,0.5,0,0

To make things clear, we provide a couple of examples below.

Muting all channels except for the first
========================================

Let's assume that the original recordings have 4 channels.
The following command will extract the first channel from the recordings:

.. code-block:: bash

   child-project process /path/to/dataset channel1 channel-mapping --channels 1,0,0,0

Invert a stereo signal
======================

Let's assume that the original recordings are stereo signals, i.e. they have two channels.
The command below will flip the two channels:

.. code-block:: bash

   child-project process /path/to/dataset channel1 channel-mapping --channels 0,1 --channels 1,0