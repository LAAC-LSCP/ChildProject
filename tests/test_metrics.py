from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
from ChildProject.tables import IndexTable
import pandas as pd
import numpy as np
import os
import pytest

@pytest.fixture(scope='function')
def project(request):
    if not os.path.exists("output/metrics"):
        project = ChildProject("examples/valid_raw_data")
        project.import_data("output/metrics")
        project = ChildProject("output/metrics")
        
    yield project

thresholds = [0,0.5,1]

