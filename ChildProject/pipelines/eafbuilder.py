import argparse
import pandas as pd
import sys
import os
import shutil
import pympi

try:
    from importlib import resources
except ImportError:
    # TODO: Perhaps add this as a dependency to the resources?
    import importlib_resources as resources

from ChildProject.projects import ChildProject
from ChildProject.pipelines.pipeline import Pipeline

def create_eaf(etf_path: str, id: str, output_dir: str,
    timestamps_list: list,
    eaf_type: str, contxt_on: int, contxt_off: int,
    template: str):

    print("ACLEW ID: ", id)

    eaf = pympi.Elan.Eaf(etf_path)
    ling_type = "transcription"
    eaf.add_tier("code_"+eaf_type, ling=ling_type)
    eaf.add_tier("context_"+eaf_type, ling=ling_type)
    eaf.add_tier("code_num_"+eaf_type, ling=ling_type)

    for i, ts in enumerate(timestamps_list):
        print("Creating eaf code segment # ", i+1)
        print("enumerate makes: ", i, ts)
        whole_region_onset = ts[0]
        whole_region_offset = ts[1]

        context_onset = int(whole_region_onset) - contxt_on
        context_offset = int(whole_region_offset) + contxt_off

        if context_onset < 0:
            context_onset = 0.0
        
        codeNumVal = eaf_type + str(i+1)
        eaf.add_annotation("code_"+eaf_type, whole_region_onset, whole_region_offset)
        eaf.add_annotation("code_num_"+eaf_type, whole_region_onset, whole_region_offset, value=codeNumVal)
        eaf.add_annotation("context_"+eaf_type, context_onset, context_offset)

    os.makedirs(output_dir, exist_ok = True)
    eaf.to_file(os.path.join(output_dir, "{}.eaf".format(id)))
    for i in eaf.get_tier_names():
        print(i,":",eaf.get_annotation_data_for_tier(i))

    return eaf

class EafBuilderPipeline(Pipeline):
    def __init__(self):
        pass
                
    def run(self, destination: str, segments: str,
        eaf_type: str, template: str,
        context_onset: int = 0, context_offset: int = 0,
        **kwargs):
        """generate .eaf templates based on intervals to code

        :param path: project path
        :type path: str
        :param destination: eaf destination
        :type destination: str
        :param segments: path to the input segments dataframe
        :type segments: str
        :param eaf_type: eaf-type [random, periodic]
        :type eaf_type: str
        :param template: name of the template to use (basic, native, or non-native)
        :type template: str
        :param context_onset: context onset and segment offset difference in milliseconds, 0 for no introductory context
        :type context_onset: float
        :param context_offset: context offset and segment offset difference in milliseconds, 0 for no outro context
        :type context_offset: float
        """

        # TODO: Make sure etf file paths are approprite and robust. 
        etf_path = "{}.etf".format(template)
        psfx_path = "{}.pfsx".format(template)

        print("making the "+eaf_type+" eaf file and csv")

        segments = pd.read_csv(segments)

        for recording_filename, segs in segments.groupby('recording_filename'):
            recording_filename = os.path.splitext(recording_filename)[0]
            output_filename = recording_filename + '_' + eaf_type + '_' + template
            
            # TODO: This list of timestamps as tuples might not be ideal/should perhaps be optimized, but I am just replicating the original eaf creation code here.
            timestamps = [(on, off) for on, off in segs.loc[:, ['segment_onset', 'segment_offset']].values]

            output_dir = os.path.join(destination, recording_filename)
            with resources.path('ChildProject.templates', etf_path) as e_path:
                create_eaf(
                    e_path,
                    output_filename,
                    output_dir,
                    timestamps,
                    eaf_type,
                    context_onset,
                    context_offset,
                    template
                )

            with resources.path('ChildProject.templates', psfx_path) as p_path:
                shutil.copy(p_path, os.path.join(output_dir, "{}.pfsx".format(output_filename)))


    @staticmethod
    def setup_parser(parser):
        parser.add_argument("--destination", help = "eaf destination")
        parser.add_argument('--segments', help = 'path to the input segments dataframe', required = True)
        # TODO: add other options here such as high-volubility, energy, etc.?
        parser.add_argument('--eaf-type', help = 'eaf-type', choices = ['random', 'periodic'], required = True)
        parser.add_argument('--template', help = 'eaf template', choices = ['basic', 'native', 'non-native'], required = True)
        parser.add_argument('--context-onset', help = 'context onset and segment offset difference in milliseconds, 0 for no introductory context', type = int, default = 0)
        parser.add_argument('--context-offset', help = 'context offset and segment offset difference in milliseconds, 0 for no outro context', type = int, default = 0)
