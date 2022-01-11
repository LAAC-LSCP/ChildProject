import argparse
import pandas as pd
import sys
import os
import shutil

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.pipeline import Pipeline
from ChildProject.tables import assert_dataframe, assert_columns_presence


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
    speech_segments: pd.DataFrame = None,
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

    if speech_segments is not None:

        from ChildProject.converters import AnnotationConverter

        type_to_id = {
            val: key
            for key, val in reversed(
                list(AnnotationConverter.SPEAKER_ID_TO_TYPE.items())
            )
        }

        for segment in speech_segments.to_dict(orient="records"):
            speaker_id = None

            if (
                "speaker_id" in segment
                and segment["speaker_id"] in AnnotationConverter.SPEAKER_ID_TO_TYPE
            ):
                speaker_id = segment["speaker_id"]
            elif "speaker_type" in segment and segment["speaker_type"] in type_to_id:
                speaker_id = type_to_id[segment["speaker_type"]]

            if speaker_id is None:
                continue

            if speaker_id not in eaf.tiers:
                eaf.add_tier(speaker_id)

            eaf.add_annotation(
                speaker_id,
                int(segment["segment_onset"]),
                int(segment["segment_offset"]),
            )

    destination = os.path.join(output_dir, "{}.eaf".format(id))
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    mime_type = "audio/x-wav"
    mime_types = {"mp3": "audio/mpeg", "mp4": "audio/mp4", "flac": "audio/x-flac"}
    mime_types.update(eaf.MIMES)

    extension = os.path.splitext(recording_filename)[1]
    if extension:
        extension = extension.lower()
        if extension in mime_types:
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
        path: str = None,
        import_speech_from: str = None,
        **kwargs,
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

        assert_dataframe("segments", segments, not_empty=True)
        assert_columns_presence(
            "segments",
            segments,
            {"recording_filename", "segment_onset", "segment_offset"},
        )

        prefill = path and import_speech_from
        if prefill:
            project = ChildProject(path)
            am = AnnotationManager(project)
            am.read()

            prefill_annotations = am.annotations[
                am.annotations["set"] == import_speech_from
            ]

            if len(prefill_annotations) == 0:
                raise ValueError(
                    f"no annotations belonging to '{import_speech_from}' found in '{path}'"
                )

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

            speech_segments = None
            if prefill:
                annotations = pd.concat(
                    [
                        prefill_annotations,
                        segs.assign(
                            recording_filename=recording_filename,
                            set=import_speech_from + "_",
                        ).rename(
                            columns={
                                "segment_onset": "range_onset",
                                "segment_offset": "range_offset",
                            }
                        ),
                    ]
                )
                intersection = am.intersection(annotations)

                if len(intersection) == 0:
                    print(
                        f"""warning: no annotation from '{import_speech_from}'"""
                        f"""matching the selected range for recording '{recording_filename}'"""
                    )
                    continue

                intersection = intersection[intersection["set"] == import_speech_from]

                segments_duration = (
                    segs["segment_offset"] - segs["segment_onset"]
                ).sum()
                annotations_duration = (
                    intersection["range_offset"] - intersection["range_onset"]
                ).sum()

                if segments_duration != annotations_duration:
                    print(
                        f"""warning: annotations do not cover the whole selected range for recording"""
                        f"""'{recording_filename}', {annotations_duration/1000}s covered instead of {segments_duration/1000}s"""
                    )

                speech_segments = am.get_segments(intersection)

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
                speech_segments,
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
        parser.add_argument(
            "--path", help="path to the input dataset", required=False,
        )
        parser.add_argument(
            "--import-speech-from",
            help="set from which active speakers should be imported",
            required=False,
        )

