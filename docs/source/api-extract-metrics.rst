Metrics python extraction
=========================

Metrics can be extracted both from the command-line interface and from the python API. 
You will find here instructions on how to use the API to customize your metrics 
extraction.

To extract metrics, you can choose to use the pipelines that are defined in the command-line 
or define all the parameters of the extraction yourself.

Use the existing pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~

We first need to initialize the project and import the necessary functions. Here we use
the example project `vandam-data <https://gin.g-node.org/LAAC-LSCP/vandam-data>`__. 
We then initialize our 2 types of metrics :class:`~ChildProject.pipelines.metrics.LenaMetrics` 
and :class:`~ChildProject.pipelines.metrics.AclewMetrics` with the desired parameters.

Here we choose to do a very simple LenaMetrics extraction using all the default values 
and the set named "its". For AclewMetrics, we initialize the class to extract on the set 
"vtc" only between 8am and 5pm on periods of 6 hours, grouped by child_id values and adding 
the values of date_iso, child_dob and child_id to the resulting output.

.. code:: python

    >>> from ChildProject.projects import ChildProject
    >>> from ChildProject.pipelines.metrics import LenaMetrics, AclewMetrics
    >>> project = ChildProject('vandam-data')
    >>> lmetrics = LenaMetrics(project,"its")
    >>> ametrics = AclewMetrics(
    ...     project,
    ...     vtc='vtc',
    ...     from_time='8:00:00',
    ...     to_time='17:00:00',
    ...     rec_cols='date_iso',
    ...     child_cols='child_dob,child_id',
    ...     period='6h',
    ...     by='child_id',
    ... )
    The ALICE set ('alice') was not found in the index.
    The vcm set ('vcm') was not found in the index.

The programm warns us that the alice and vcm sets are not present which is expected given 
that the vandam-data corpus does not have vcm and alice annotations. So the output will not 
contain the metrics extracted from those.

We then launch the extraction for each pipeline. The function populates the ``.metrics`` attribute 
and returns the resulting metrics. Here we save the resulting metrics in csv files with the 
`to_csv <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html>`__ function 
from pandas.

.. code:: python

    >>> lmetrics.extract()
    recording_filename  child_id  duration_its  ...  voc_mal_ph  voc_dur_chi_ph  lp_dur
    0    BN32_010007.mp3         1      50464512  ...   103.86705   178674.867598     NaN

    [1 rows x 20 columns]
    >>> lmetrics.metrics.to_csv('LenaMetrics.csv', index=False)
    >>> ametrics.extract()
    child_id period_start period_end  ... avg_voc_dur_mal avg_voc_dur_och  avg_voc_dur_chi
    0         1     00:00:00   06:00:00  ...             NaN             NaN              NaN
    1         1     06:00:00   12:00:00  ...     1373.208247      935.654378      1159.420822
    2         1     12:00:00   18:00:00  ...     1099.721472      808.012712      1011.550502
    3         1     18:00:00   00:00:00  ...             NaN             NaN              NaN

    [4 rows x 18 columns]
    >>> ametrics.metrics.to_csv('AclewMetrics.csv', index=False)

Define you own metrics
~~~~~~~~~~~~~~~~~~~~~~

You can also create your own metrics by defining your python function calculating the output value.
To do so, define a function taking as arguments:
 - annotations : pandas DataFrame, this is the actual segments of the converted set
 - duration : int, the represents the length that was annotated, use this value to calculate rates per hour for example
 - \*\*kwargs : keyword arguments, this allows the user to give whatever arguments he likes through the list of metrics
and returning, in that order:
 - a default name for the metric to take, it will be used when no specific name was explicitly required by the user
 - the value of the metric, should be a number or np.nan (a distinction is made between 0 and np.nan as np.nan indicates 
that the value can not be calculated).

The function should check the presence of the required columns in the annotations and of the required keyword arguments. 
To make this easier, use the function :func:`ChildProject.pipelines.metricsFunctions.metricFunction` as a decorator 
to perform those checks as well as giving a default name based on the function's name. The decorator should be called 
along with the parameters :
 - args : a set of the names of the required keyword arguments
 - columns : a set of the names of the required columns in the annotations
 - emptyValue : the value to return when no annotations segments are found
 - name : the default name to use the designate this metric. If empty, uses the function name. Be aware that keyword 
arguments found in the name will be replaced by their value (e.g. voc_speaker_ph with ``speaker='CHI'`` will return voc_chi_ph).
The only remaining task of the function is the calculation and return of the value.

Here we define a function that only requires the keyword argument 'speaker' and is calculated only based on the 
'speaker_type' column. When no annotation is found, its value will be 0 and by default it will take the name 
'num_of_voc_speaker' with <speaker> being replaced with the value of the 'speaker' keyword argument.
The returned value is the number of lines belonging to the speaker_type (i.e. its number of vocalizations as an 
absolute value).

.. code:: python

    >>> from ChildProject.projects import ChildProject
    >>> from ChildProject.pipelines.metricsFunctions import metricFunction
    >>> import pandas as pd
    >>> @metricFunction({'speaker'},{'speaker_type'}, 0, 'num_of_voc_speaker')
    ... def voc_speaker(annotations: pd.DataFrame, duration: int, **kwargs):
    ...     return annotations[annotations["speaker_type"]== kwargs["speaker"]].shape[0]
    ...

We defined our custom metric, now we will create our list of wanted metrics. It must be a pandas DataFrame compatible 
with the :ref:`list_structure`. The callable function is used for both names of the default available metrics and 
callables functions that we defined ourselves.
Here we only use the vtc set, we want to extract the number of vocalizations produced by the key child and the mother 
in absolute values (using our newly defined function) but also in values per hour (using the default metric 
<voc_speaker_ph>).

.. code:: python

    >>> input = pd.DataFrame([{
    ...     'set': 'vtc',
    ...     'callable': 'voc_speaker_ph',
    ...     'speaker': 'CHI',
    ... },{
    ...     'set': 'vtc',
    ...     'callable': 'voc_speaker_ph',
    ...     'speaker': 'FEM',
    ... },{
    ...     'set': 'vtc',
    ...     'callable': voc_speaker,
    ...     'speaker': 'CHI',
    ... }{
    ...     'set': 'vtc',
    ...     'callable': voc_speaker,
    ...     'speaker': 'FEM',
    ... }])
    
Last thing left to do is initialize our :class:`ChildProject.pipelines.metrics.Metrics` with the correct 
parameters and launch the extraction

.. code:: python

    >>> from ChildProject.pipelines.metrics import Metrics
    >>> project = ChildProject('vandam-data')
    >>> m = Metrics(
    ...     project,
    ...     metrics_list= input,
    ...     from_time='8:00:00',
    ...     to_time='17:00:00',
    ...     rec_cols='date_iso',
    ...     child_cols='child_dob,child_id',
    ...     period='6h',
    ...     by='child_id',
    ... )
    >>> m.extract()
        child_id period_start period_end  ... voc_fem_ph num_of_voc_chi  num_of_voc_fem
    0         1     00:00:00   06:00:00  ...        NaN            NaN             NaN
    1         1     06:00:00   12:00:00  ...      244.5         1143.0           978.0
    2         1     12:00:00   18:00:00  ...      253.4         1495.0          1267.0
    3         1     18:00:00   00:00:00  ...        NaN            NaN             NaN

    [4 rows x 10 columns]
    >>> m.metrics.to_csv('Metrics.csv', index=False)
