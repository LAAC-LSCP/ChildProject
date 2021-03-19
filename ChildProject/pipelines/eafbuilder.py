import argparse
import os
import pandas as pd
import sys
import os.path
import shutil

try:
    from importlib import resources
except ImportError:
    # TODO: Perhaps add this as a dependency to the resources?
    import importlib_resources as resources

from ChildProject.projects import ChildProject
from ChildProject.pipelines.pipeline import Pipeline

from ChildProject.pipelines.utils import choose_template, create_eaf


class EafBuilderPipeline(Pipeline):
    def __init__(self):
        pass
                
    def run(self, path: str, destination: str, segments: str,
        eaf_type: str, template: str,
        context_onset: float, context_offset: float,
        **kwargs):

        self.project = ChildProject(path)
        self.project.read()


        # TODO: Make sure etf file paths are approprite and robust. 
        etf_path, psfx_path = choose_template(template)

        print("making the "+eaf_type+" eaf file and csv")

        segments = pd.read_csv(segments)

        # TODO: This list of timestamps as tuples might not be ideal/should perhaps be optimized, but I am just replicating the original eaf creation code here.
        timestamps = [(on, off) for on, off in segments.loc[:, ['segment_onset', 'segment_offset']].values]


        for recording in self.project.recordings.to_dict(orient = 'records'):
            recording_filename = os.path.splitext(recording['recording_filename'])[0]

            output_dir = os.path.join(destination, recording_filename)
            with resources.path('ChildProject.etf_templates', etf_path) as e_path:
                create_eaf(e_path,recording_filename+eaf_type+'_'+'its_'+template, output_dir, timestamps,eaf_type,context_onset,context_offset,template)

            with resources.path('ChildProject.etf_templates', psfx_path) as p_path:
                shutil.copy(p_path, os.path.join(output_dir, "{}.pfsx".format(recording_filename+eaf_type+'_'+'its_'+template)))


    @staticmethod
    def setup_parser(parser):
        parser.add_argument("path", help = "project path")
        parser.add_argument("destination", help = "eaf destination")
        parser.add_argument('--segments', help = 'path to the input segments dataframe', required = True)
        # TODO: add other options here such as high-volubility, energy, etc.?
        parser.add_argument('--eaf-type', help = 'eaf-type', choices = ['random', 'periodic'])
        parser.add_argument('--template', help = 'eaf template', choices = ['basic', 'native', 'non-native'], required = True)
        parser.add_argument('--context-onset', help = 'context onset and segment offset difference, 0 for no introductory context', type = float, default = 0)
        parser.add_argument('--context-offset', help = 'context offset and segment offset difference, 0 for no outro context', type = float, default = 0)
