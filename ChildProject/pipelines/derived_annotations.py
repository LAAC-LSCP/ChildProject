#!usr/bin/env python
# -*- coding: utf8 -*-
#
# created : 2023-11-23
# author : lpeurey
# ------------------------------------------------------------------------------
#   Description:
#       •   pipeline to compute annotations that are derived from other sets
# -----------------------------------------------------------------------------
from pathlib import Path

class AnnotationDeriver(ABC):
    def __init__(self,
                 project,
                 base_set,
                 new_set,
                 overwrite=False,
                 exist_ok=False,
                 ):
        self.project = project
        self.base_set = base_set
        self.new_set = new_set
        self.overwrite = overwrite
        self.exist_ok = exist_ok

        #TODO make sure the project is correct, base set exists, newset does not exist (or empty) if not exist_ok

        self.output_directory = project.path / ChildProject.projects.PROJECT_FOLDERS[1]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        pipelines[cls.SUBCOMMAND] = cls

