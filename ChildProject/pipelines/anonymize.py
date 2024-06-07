import glob
import os
import pandas as pd
import json
import re
import shutil

from ..projects import ChildProject
from ..annotations import AnnotationManager
from .pipeline import Pipeline


class AnonymizationPipeline(Pipeline):
    """Anonymize a set of its annotations (`input_set`) and saves it as `output_set`."""

    DEFAULT_REPLACEMENTS = {
        "PrimaryChild": {"DOB": "1000-01-01"},
        "ITS": {
            "fileName": "new_filename_1001",
            "timeCreated": [{"replace_value": "1000-01-01"}, {"only_time": "true"}],
        },
        "TransferTime": {
            "UTCTime": [{"replace_value": "1000-01-01"}, {"only_time": "true"}],
            "LocalTime": [{"replace_value": "1000-01-01"}, {"only_time": "true"}],
        },
        "ChildInfo": {"dob": "1000-01-01"},
        "Child": {"DOB": "1000-01-01", "EnrollDate": "1000-01-01", "id": "A999"},
        "Bar": {
            "startClockTime": [{"replace_value": "1000-01-01"}, {"only_time": "true"}]
        },
        "BarSummary": {
            "leftBoundaryClockTime": [
                {"replace_value": "1000-01-01"},
                {"only_time": "true"},
            ],
            "rightBoundaryClockTime": [
                {"replace_value": "1000-01-01"},
                {"only_time": "true"},
            ],
        },
        "Recording": {
            "startClockTime": [{"replace_value": "1000-01-01"}, {"only_time": "true"}],
            "endClockTime": [{"replace_value": "1000-01-01"}, {"only_time": "true"}],
        },
        "FiveMinuteSection": {
            "startClockTime": [{"replace_value": "1000-01-01"}, {"only_time": "true"}],
            "endClockTime": [{"replace_value": "1000-01-01"}, {"only_time": "true"}],
        },
        "Item": {"timeStamp": [{"replace_value": "1000-01-01"}, {"only_time": "true"}]},
        "ProcessingJob": {
            "logfile": "exec10001010T100010Z_job00000001-10001010_101010_100100.upl.log"
        },
        "ResourceSnapshot": {
            "timelocal": [{"replace_value": "1000-01-01"}, {"only_time": "true"}],
            "timegmt": [{"replace_value": "1000-01-01"}, {"only_time": "true"}],
        },
    }

    def run(
        self,
        path: str,
        input_set: str,
        output_set: str,
        replacements_json_dict: str = "",
        **kwargs
    ):
        """Anonymize a set of its annotations (`input_set`) and saves it as `output_set`."""

        if input_set == output_set:
            raise Exception("input_set and output_set should not be equal")

        project = ChildProject(path)
        project.read()

        replacements = self.DEFAULT_REPLACEMENTS
        if replacements_json_dict:
            replacements = json.load(open(replacements_json_dict, "r"))

        input_set_path = os.path.join(project.path, "annotations", input_set, "raw")
        output_set_path = os.path.join(project.path, "annotations", output_set, "raw")

        if os.path.exists(output_set_path):
            raise Exception("destination {} already exists".format(output_set_path))

        its_files = glob.glob(os.path.join(input_set_path, "**/*.*"), recursive=True)
        for its in its_files:
            inFile = its
            outFile = os.path.join(
                output_set_path, its[len(os.path.join(input_set_path, "")) :]
            )
            os.makedirs(os.path.dirname(outFile), exist_ok=True)

            with open(inFile, "r") as inF:
                with open(outFile, "w") as outF:
                    for line in inF:
                        for node in replacements.keys():
                            if re.search(
                                r"<{}\b".format(node), line
                            ):  # word boundary is important here
                                for name, value in replacements[node].items():
                                    if isinstance(value, list):
                                        if bool(value[1]["only_time"]) is True:
                                            line = re.sub(
                                                r'{}="[0-9\-]*'.format(name),
                                                r'{}="{}'.format(
                                                    name, value[0]["replace_value"]
                                                ),
                                                line,
                                            )
                                            continue

                                    line = re.sub(
                                        r'{}="[a-zA-Z0-9_.:\-]*"'.format(name),
                                        r'{}="{}"'.format(name, value),
                                        line,
                                    )
                        outF.write(line)

    @staticmethod
    def setup_parser(parser):
        parser.add_argument("path", help="project path")
        parser.add_argument("--input-set", help="input annotation set", required=True)
        parser.add_argument("--output-set", help="output annotation set", required=True)
        parser.add_argument(
            "--replacements-json-dict",
            help="path to the replacements configuration (json dict)",
            required=False,
            default="",
        )
