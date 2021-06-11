from abc import ABC, abstractmethod

from datetime import datetime
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from .projects import ChildProject

class Plot(ABC):
    def __init__(self, project):
        self.project = project

    @abstractmethod
    def plot(self):
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