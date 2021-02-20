Zooniverse
==========

-  `Introduction <#introduction>`__

   -  `Overview <#overview>`__
   -  `Chunk extraction <#chunk-extraction>`__
   -  `Chunk upload <#chunk-upload>`__
   -  `Classifications retrieval <#classifications-retrieval>`__

Introduction
~~~~~~~~~~~~

We are providing here a pipeline to create, upload and analyse long
format recordings using the Zooniverse citizen science platform.

We have an open project aimed at adding vocal maturity labels to
segments LENA labeled as being key child in Zooniverse
(https://www.zooniverse.org/projects/chiarasemenzin/maturity-of-baby-sounds).

If you would like your data labeled with this project, here is what
you’d need to do. 1. Get in touch with us, so we know you are
interested! 2. Have someone trustworthy & with some coding skills
(henceforth, the RA) create a database using the `formatting
instructions and
specifications <http://laac-lscp.github.io/ChildRecordsData/FORMATTING.html>`__.
4. Have the RA create an account on Zooniverse (top right of
zooniverse.org) for them and yourself, & provide us with both handles.
The RA should first update the team section to add you (have ready a
picture and a blurb). The RA can also add your institution’s logo if
you’d like. Both of these are done in the `lab
section <https://www.zooniverse.org/lab/10073>`__ 5. The RA will then
follow the instructions in the present README to create subjects and
push up your data – see below. 6. We also ask the RA to pitch in and
help answer questions in the
`forum <https://www.zooniverse.org/projects/chiarasemenzin/maturity-of-baby-sounds/talk>`__,
at least one comment a day. 7. You can visit the `stats
section <https://www.zooniverse.org/projects/chiarasemenzin/maturity-of-baby-sounds/stats>`__
to look at how many annotations are being done.

You can also use this code and your own knowledge to set up a new
project of your own.

Overview
~~~~~~~~

.. clidoc::

   child-project zooniverse --help


Chunk extraction
~~~~~~~~~~~~~~~~

.. clidoc::

   child-project zooniverse extract-chunks /path/to/dataset --help

If it does not exist, DESTINATION is created. Audio chunks are saved in
wav and mp3 in ``DESTINATION/chunks``. Metadata is stored in a file
named ``DESTINATION/chunks.csv``.

Chunk upload
~~~~~~~~~~~~

.. clidoc::

   child-project zooniverse upload-chunks /path/to/dataset --help


Uploads as many batches of audio chunks as specified to Zooniverse, and
updates ``chunks.csv`` accordingly.

Classifications retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~

.. clidoc::

   child-project zooniverse retrieve-classifications /path/to/dataset --help

Retrieve classifications and save them into
``DESTINATION/classifications.csv``.