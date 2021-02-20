from docutils import nodes
from docutils.parsers.rst import Directive
from docutils import Component
from sphinx.directives.code import CodeBlock

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

def setup(app):
    app.add_directive("clidoc", CliDoc)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }