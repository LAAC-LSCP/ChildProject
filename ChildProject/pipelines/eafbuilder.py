import argparse
import os
import pandas as pd
import sys

from ChildProject.projects import ChildProject
from ChildProject.pipelines.pipeline import Pipeline

class EafBuilderPipeline(Pipeline):
    def __init__(self):
        pass
                
    def run(self, path: str, destination: str, segments: str,
        type: str, template: str,
        context_onset: float, context_offset: float,
        **kwargs):

        pass

    @staticmethod
    def setup_parser(parser):
        parser.add_argument("path", help = "project path")
        parser.add_argument("destination", help = "eaf destination")
        parser.add_argument('--segments', help = 'path to the input segments dataframe', required = True)
        parser.add_argument('--type', help = 'eaf-type', choices = ['random', 'periodic'])
        parser.add_argument('--template', help = 'eaf template', choices = ['basic', 'native', 'non-native'])
        parser.add_argument('--context-onset', help = 'context onset and segment offset difference, 0 for no introductory context', type = float, default = 0)
        parser.add_argument('--context-offset', help = 'context offset and segment offset difference, 0 for no outro context', type = float, default = 0)
