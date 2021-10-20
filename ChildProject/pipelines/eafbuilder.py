import argparse
import pandas as pd
import sys
import os
import shutil

from ChildProject.projects import ChildProject
from ChildProject.pipelines.pipeline import Pipeline


def create_eaf(
    etf_path: str,
    id: str,
    output_dir: str,
    recording_filename: str,
    timestamps_list: list,
    eaf_type: str,
    contxt_on: int,
    contxt_off: int,
    template: str,
):
    import pympi

    eaf = pympi.Elan.Eaf(etf_path)

    ling_type = "transcription"
    eaf.add_tier("code_" + eaf_type, ling=ling_type)
    eaf.add_tier("context_" + eaf_type, ling=ling_type)
    eaf.add_tier("code_num_" + eaf_type, ling=ling_type)

    for i, ts in enumerate(timestamps_list):
        print("Creating eaf code segment # ", i + 1)
        print("enumerate makes: ", i, ts)
        whole_region_onset = int(ts[0])
        whole_region_offset = int(ts[1])

        context_onset = int(whole_region_onset - contxt_on)
        context_offset = int(whole_region_offset + contxt_off)

        if context_onset < 0:
            context_onset = 0

        codeNumVal = eaf_type + str(i + 1)
        eaf.add_annotation("code_" + eaf_type, whole_region_onset, whole_region_offset)
        eaf.add_annotation(
            "code_num_" + eaf_type,
            whole_region_onset,
            whole_region_offset,
            value=codeNumVal,
        )
        eaf.add_annotation("context_" + eaf_type, context_onset, context_offset)

    destination = os.path.join(output_dir, "{}.eaf".format(id))
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    mime_type = "audio/x-wav"
    mime_types = {"mp3": "audio/mpeg", "mp4": "audio/mp4", "flac": "audio/x-flac"}
    mime_types.update(eaf.MIMES)

    extension = os.path.splitext(recording_filename)[1]
    if extension:
        mime_type = mime_types[extension[1:]]

    eaf.add_linked_file(
        file_path=recording_filename, relpath=recording_filename, mimetype=mime_type
    )

    eaf.to_file(destination)
    for i in eaf.get_tier_names():
        print(i, ":", eaf.get_annotation_data_for_tier(i))

    return eaf


class EafBuilderPipeline(Pipeline):
    def __init__(self):
        pass

    def run(
        self,
        destination: str,
        segments: str,
        eaf_type: str,
        template: str,
        context_onset: int = 0,
        context_offset: int = 0,
        **kwargs
    ):
        """generate .eaf templates based on intervals to code.

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
        :type context_onset: int
        :param context_offset: context offset and segment offset difference in milliseconds, 0 for no outro context
        :type context_offset: int
        """

        try:
            from importlib import resources
        except ImportError:
            # TODO: Perhaps add this as a dependency to the resources?
            import importlib_resources as resources

        etf_path = "{}.etf".format(template)
        pfsx_path = "{}.pfsx".format(template)

        if template in ["basic", "native", "non-native"]:
            with resources.path("ChildProject.templates", etf_path) as etf:
                etf_path = str(etf)

            with resources.path("ChildProject.templates", pfsx_path) as pfsx:
                pfsx_path = str(pfsx)

        if not os.path.exists(etf_path):
            raise Exception("{} cannot be found".format(etf_path))

        if not os.path.exists(pfsx_path):
            raise Exception("{} cannot be found".format(pfsx_path))

        print("making the " + eaf_type + " eaf file and csv")

        segments = pd.read_csv(segments)

        for recording_filename, segs in segments.groupby("recording_filename"):
            recording_prefix = os.path.splitext(recording_filename)[0]
            output_filename = (
                recording_prefix + "_" + eaf_type + "_" + os.path.basename(template)
            )

            # TODO: This list of timestamps as tuples might not be ideal/should perhaps be optimized, but I am just replicating the original eaf creation code here.
            timestamps = [
                (on, off)
                for on, off in segs.loc[:, ["segment_onset", "segment_offset"]].values
            ]

            output_dir = os.path.join(destination, recording_prefix)

            create_eaf(
                etf_path,
                output_filename,
                output_dir,
                recording_filename,
                timestamps,
                eaf_type,
                context_onset,
                context_offset,
                template,
            )

            shutil.copy(
                pfsx_path, os.path.join(output_dir, "{}.pfsx".format(output_filename))
            )

    @staticmethod
    def setup_parser(parser):
        parser.add_argument("--destination", help="eaf destination")
        parser.add_argument(
            "--segments", help="path to the input segments dataframe", required=True
        )
        # TODO: add other options here such as high-volubility, energy, etc.?
        parser.add_argument(
            "--eaf-type",
            help="eaf-type",
            choices=["random", "periodic", "high-volubility", "energy-detection"],
            required=True,
        )
        parser.add_argument(
            "--template",
            help="Which ACLEW templates (basic, native or non-native); otherwise, the path to the etf et pfsx templates, without the extension.",
            required=True,
        )
        parser.add_argument(
            "--context-onset",
            help="context onset and segment offset difference in milliseconds, 0 for no introductory context",
            type=int,
            default=0,
        )
        parser.add_argument(
            "--context-offset",
            help="context offset and segment offset difference in milliseconds, 0 for no outro context",
            type=int,
            default=0,
        )
