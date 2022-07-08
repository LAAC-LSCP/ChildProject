from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst.directives.tables import CSVTable
from docutils import Component

from inspect import getmembers, isfunction, cleandoc

from sphinx.directives.code import CodeBlock

from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager

from ChildProject.pipelines import metricsFunctions

import subprocess

# I think this is a sort of hack (smart hack, but a hack nonetheless !)
# and that it should not be done this way

class CliDoc(CodeBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        command_string = "\n".join(self.content).strip()
        proc = subprocess.Popen(command_string, stdout = subprocess.PIPE, stderr = None, shell = True)
        out = proc.communicate()[0]

        if proc.returncode != 0:
            raise Exception("CliDoc('{}') returned non-zero status code.".format(command_string))

        if not self.arguments:
            self.arguments = ['bash']

        self.content = ["\n$ {}\n{}".format(command_string, out.decode("utf-8"))]

from functools import reduce
import os
import pandas as pd
import textwrap

def wrap(s, width):
    return reduce(lambda x,y: x+"\n| {}".format(y), textwrap.wrap(s, width = width), '')

class IndexTable(CSVTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'header' not in self.options:
            raise KeyError("IndexTable is missing a header attribute.")
        
        array = self.options.pop('header')
        
        table = None
        if array == 'children':
            table = ChildProject.CHILDREN_COLUMNS
        elif array == 'recordings':
            table = ChildProject.RECORDINGS_COLUMNS
        elif array == 'input_annotations':
            table = [c for c in AnnotationManager.INDEX_COLUMNS if not c.generated]
        elif array == 'annotation_segments':
            table = AnnotationManager.SEGMENTS_COLUMNS
        elif array == 'annotations':
            table = [c for c in AnnotationManager.INDEX_COLUMNS if (c.generated or c.required)]
        elif array == 'documentation':
            table = ChildProject.DOCUMENTATION_COLUMNS                

        if not table:
            raise Exception("invalid table '{}'".format(array))

        df = []
        for column in table:
            df_entry = {
                'Name': column.name,
                'Description': wrap(column.description, 50),
                'Required?': '**required**' if column.required else 'optional'
            }

            df_entry['Format'] = ''
            if column.datetime:
                df_entry['Format'] = "``{}``".format(column.datetime)
            elif column.regex:
                df_entry['Format'] = "``{}``".format(column.regex)
            elif column.function:
                df_entry['Format'] = column.function.__name__
            elif column.choices:
                df_entry['Format'] = wrap(", ".join(column.choices), 10)
            elif column.filename:
                df_entry['Format'] = column.filename

            df.append(df_entry)

        self.options['file'] = '{}.csv'.format(array)
        self.options['header-rows'] = 1
        self.options['widths'] = [20, 50, 10, 20]

        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        pd.DataFrame(df).to_csv(os.path.join(dir_path, self.options['file']), index = False)
        
class CustomTable(CSVTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'header' not in self.options:
            raise KeyError("IndexTable is missing a header attribute.")
        
        array = self.options.pop('header')
        
        df = []
        if array == 'list-metrics':
            ignores = {'metricFunction'}
            metrics = getmembers(metricsFunctions, isfunction)
            for name, func in metrics:
                if name in ignores : continue
                doc = func.__doc__.split('Required keyword arguments:',1)
                description = cleandoc(doc[0])
                arguments = cleandoc(doc[1]) if len(doc) > 1 else ""
                df_entry = {
                    'Callable': name,
                    'Description': wrap(description, 50),
                    'Required arguments': wrap(arguments,25),
                }
                df.append(df_entry)
                
        self.options['file'] = '{}.csv'.format(array)
        self.options['header-rows'] = 1
        self.options['widths'] = [20, 50, 25]
        
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        pd.DataFrame(df).to_csv(os.path.join(dir_path, self.options['file']), index = False)
                
def setup(app):
    app.add_directive("clidoc", CliDoc)
    app.add_directive("index-table", IndexTable)
    app.add_directive("custom-table", CustomTable)

    return {
        'version': '0.2',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }