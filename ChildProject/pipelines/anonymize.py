import os
import pandas as pd
import json

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.pipelines.pipeline import Pipeline

class AnonymizationPipeline(Pipeline):
    DEFAULT_REPLACEMENTS = {
        "PrimaryChild": {
            "DOB": "1000-01-01"
        },
        "ITS":{
            "fileName":"new_filename_1001",
            "timeCreated":[{"replace_value":"1000-01-01"},{"only_time":"true"}]
        },
        "TransferTime": {
            "UTCTime": [{"replace_value":"1000-01-01"},{"only_time":"true"}],
            "LocalTime": [{"replace_value":"1000-01-01"},{"only_time":"true"}]
        },
        "ChildInfo": {
            "dob": "1000-01-01"
        },
        "Child": {
            "DOB": "1000-01-01",
            "EnrollDate": "1000-01-01",
            "id": "A999"
        },
        "Bar": {
            "startClockTime": [{"replace_value":"1000-01-01"},{"only_time":"true"}]
        },
        "BarSummary": {
            "leftBoundaryClockTime": [{"replace_value":"1000-01-01"},{"only_time":"true"}],
            "rightBoundaryClockTime": [{"replace_value":"1000-01-01"},{"only_time":"true"}]
        },
        "Recording": {
            "startClockTime": [{"replace_value":"1000-01-01"},{"only_time":"true"}],
            "endClockTime": [{"replace_value":"1000-01-01"},{"only_time":"true"}]
        },
        "FiveMinuteSection": {
            "startClockTime": [{"replace_value":"1000-01-01"},{"only_time":"true"}],
            "endClockTime": [{"replace_value":"1000-01-01"},{"only_time":"true"}]
        },
        "Item": {
            "timeStamp": [{"replace_value":"1000-01-01"},{"only_time":"true"}]
        },
        "ProcessingJob": {
            "logfile": "exec10001010T100010Z_job00000001-10001010_101010_100100.upl.log"
        },
        "ResourceSnapshot":{
            "timelocal": [{"replace_value":"1000-01-01"},{"only_time":"true"}],
            "timegmt": [{"replace_value":"1000-01-01"},{"only_time":"true"}]
        }
    }

    def run(self, path, input_set, output_set, replacements_json_dict = '', **kwargs):
        project = ChildProject(path)
        project.read()

        am = AnnotationManager(project)
        am.read()

        annotations = am.annotations

        replacements = self.DEFAULT_REPLACEMENTS
        if replacements_json_dict:
            replacements = json.load(open(replacements_json_dict, 'r'))


    @staticmethod
    def setup_parser(parser):
        parser.add_argument("path", help = "project path")
        parser.add_argument("--input-set", help = "input annotation set", required = True)
        parser.add_argument("--output-set", help = "output annotation set", required = True)
        parser.add_argument("--replacements-json-dict", help = "path to the replacements configuration (json dict)", required = False, default = '')
