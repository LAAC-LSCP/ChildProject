from abc import ABC, abstractmethod

from datetime import datetime
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from .projects import ChildProject
from .annotations import AnnotationManager

class Plot(ABC):
    def __init__(self, project):
        self.project = project

    @abstractmethod
    def plot(self, **kwargs):
        pass

class AgeDistributionPlot(Plot):

    def __init__(self, project):
        super().__init__(project)

    def plot(self, **kwargs):

        children = self.project.children.copy()
        children['child_dob'] = children['child_dob'].apply(lambda s: datetime.strptime(s, '%Y-%m-%d'))
        recordings = self.project.recordings.copy()

        if 'session_id' not in recordings.columns:
            recordings['session_id'] = recordings['recording_filename']

        recordings = recordings.merge(children, how = 'left', left_on = 'child_id', right_on = 'child_id')
        recordings['date_iso'] = recordings['date_iso'].apply(lambda s: datetime.strptime(s, '%Y-%m-%d'))
        recordings['age'] = recordings.apply(
            lambda row: (row['date_iso'].year - row['child_dob'].year) * 12 + (row['date_iso'].month - row['child_dob'].month),
            axis = 1
        )
        recordings.drop_duplicates(['child_id', 'session_id'], keep = 'first', inplace = True)

        ages = np.arange(recordings['age'].min(), recordings['age'].max()+1)
        children_age_counts = recordings.groupby('child_id')['age'].value_counts(sort = False)
        children_age_counts = children_age_counts.to_frame('count')
        children_age_counts.index = children_age_counts.index.set_names(['child_id', 'age'])

        ids = children['child_id'].unique()
        children_age_counts = children_age_counts.reindex(index = [(child_id, age) for child_id in ids for age in ages], fill_value = 0)

        children_age_counts.sort_values(['child_id', 'age'], inplace = True)
        children_age_counts.reset_index(inplace = True)
        children_age_counts = children_age_counts.pivot(index = 'child_id', columns = 'age', values = 'count')

        levels = np.arange(np.max(children_age_counts.values)+1)
        colors = sns.color_palette('Reds', len(levels))
        cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend="max")

        fig, ax = plt.subplots(
            figsize = (5, 5*children_age_counts.shape[0]/25),
            **kwargs
        )

        im = ax.imshow(children_age_counts.values, cmap = cmap, norm = norm, aspect = 'auto')
        ax.set(
            xticks = np.arange(len(ages))[::3],
            xticklabels = ages[::3],
            yticks = np.arange(len(children_age_counts.index)),
            yticklabels = children_age_counts.index.values
        )
        ax.tick_params(axis = "x", rotation = 90)

        pad = 0.015
        width = 0.03
        height = 0.1
        pos = ax.get_position()
        cax = fig.add_axes([pos.xmax + pad, pos.ymax-height*(pos.ymax-pos.ymin), width, height*(pos.ymax-pos.ymin) ])
        fig.colorbar(im, cax = cax)

        ax.set_title('Recordings per child per age')
        ax.set_xlabel('age in months')

        return fig, ax, children_age_counts

class AnnotationCoveragePlot(Plot):

    def __init__(self, project, sets: list = None, colors: list = None):
        super().__init__(project)

        self.sets = set(sets) if sets is not None else None

        if colors:
            self.colors = colors
        else:
            self.colors = [
                '#eee', # no recording
                '#ff0000', # not annotated
                '#00ff00' # annotated
            ]

    def plot(self, time_resolution = 30000, **kwargs):

        am = AnnotationManager(self.project)
        am.read()

        if self.sets is not None:
            am.annotations = am.annotations[am.annotations['set'].isin(self.sets)]

        sets = list(sorted(am.annotations['set'].unique()))

        recordings = self.project.recordings.sort_values(['child_id', 'recording_filename'])
        recordings = recordings.set_index('recording_filename')
        recs = recordings.index.values
        
        max_duration = self.project.recordings['duration'].max()

        am.annotations['range_onset'] += am.annotations['time_seek']
        am.annotations['range_offset'] += am.annotations['time_seek']

        am.annotations['n_onset'] = am.annotations['range_onset']//time_resolution
        am.annotations['n_offset'] = am.annotations['range_offset']//time_resolution

        rows = len(sets) * len(recs)
        data = np.zeros((rows, int(max_duration/time_resolution)))

        for i, rec in enumerate(recs):
            rec_end = recordings.loc[rec, 'duration']//time_resolution
            data[i*len(sets):(i+1)*len(sets), rec_end:] = -1

        for annotation in am.annotations.to_dict(orient = 'records'):
            set_n = sets.index(annotation['set'])
            rec_n = recordings.index.get_loc(annotation['recording_filename'])

            row = set_n + rec_n*len(sets)
            data[row, annotation['n_onset']:annotation['n_offset']] = 1

        fig, ax = plt.subplots(
            figsize = (10, 5*len(recs)/30),
            **kwargs
        )

        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(self.colors, name = "custom_cmap")

        im = ax.imshow(data, aspect = 'auto', cmap = custom_cmap, interpolation = 'none')
        
        return fig, ax, data