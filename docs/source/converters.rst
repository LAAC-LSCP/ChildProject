Audio post-processing
---------------------

Overview
~~~~~~~~

The package provides post-processing tools for the recordings

.. clidoc::

   child-project converters --help

Basic audio conversion
~~~~~~~~~~~~~~~~~~~~~~

Converts all recordings in a dataset to a given encoding. Converted
audios are stored into ``recordings/converted/<profile-name>``.

.. clidoc::

   child-project converters /path/to/dataset test basic --help

Example:

::

   child-project converters /path/to/dataset 16kHz basic --format=wav --sampling=16000 --codec=pcm_s16le

We typically run the following, to split long sound files every 15
hours, because the software we use for human annotation (ELAN, Praat)
works better with audio that is maximally 15h long:

::

   child-project converters /path/to/dataset 16kHz basic --split=15:00:00 --format=wav --sampling=16000 --codec=pcm_s16le

Multi-core audio conversion with slurm on a cluster
===================================================

If you have access to a cluster with slurm, you can use a command like
the one below to batch-convert your recordings. Please note that you may
need to change some details depending on your cluster (eg cpus per
task). If needed, refer to the `slurm user
guide <https://slurm.schedmd.com/quickstart.html>`__

::

   sbatch --mem=64G --time=5:00:00 --cpus-per-task=4 --ntasks=1 -o namibia.txt child-project converters --threads 4 /path/to/dataset 16kHz basic --split=15:00:00 --format=wav --sampling=16000 --codec=pcm_s16le

Vetting
~~~~~~~

.. clidoc::

    child-project converters /path/to/dataset test vetting --help