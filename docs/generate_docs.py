from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
import jinja2
import time
import subprocess

def cli_doc(command_string):
    proc = subprocess.Popen(command_string, stdout = subprocess.PIPE, stderr = None, shell = True)
    out = proc.communicate()[0]

    if proc.returncode != 0:
        raise Exception("cli_doc('{}') returned non-zero status code.".format(command_string))

    return "```bash\n$ {}\n{}```".format(command_string, out.decode("utf-8"))

# recursively get the sum of durations of each audio in the current directory :
# find . -type f -execdir soxi -D {} \; | awk '{s+=$1} END {printf "%.0f", s}'
projects = [
    {'name': 'Namibia', 'status': 'ready', 'authors': "Gandhi", 'location': 'https://github.com/LAAC-LSCP/namibia-data', 'recordings': 113, 'duration': 5214771/3600},
    {'name': 'Solomon', 'status': 'ready', 'authors': "Sarah", 'location': 'https://github.com/LAAC-LSCP/solomon-data', 'recordings': 388, 'duration': 21435406/3600},
    {'name': 'Tsimane 2017', 'status': 'validation', 'authors': "", 'location': 'https://github.com/LAAC-LSCP/tsimane2017-data', 'recordings': 41, 'duration': 2001601/3600},
    {'name': 'png 2019', 'status': 'ready', 'authors': "", 'location': 'https://github.com/LAAC-LSCP/png2019-data', 'recordings': 51, 'duration': 2737005/3600},
    {'name': 'Vanuatu', 'status': 'raw', 'authors': "", 'location': '/scratch1/projects/ac_lacie01/STRUCTURE/raw/vanuatu', 'recordings': 53, 'duration': 1040709/3600}
]

template = jinja2.Template(open('docs/templates/FORMATTING.md', 'r').read())
open('docs/FORMATTING.md', 'w+').write(
    template.render(
        children = ChildProject.CHILDREN_COLUMNS,
        recordings = ChildProject.RECORDINGS_COLUMNS,
        input_annotations = [c for c in AnnotationManager.INDEX_COLUMNS if not c.generated],
        annotation_segments = AnnotationManager.SEGMENTS_COLUMNS,
        annotations = [c for c in AnnotationManager.INDEX_COLUMNS if (c.generated or c.required)]
    )
)

template = jinja2.Template(open('docs/templates/EXTANT.md', 'r').read())
open('docs/EXTANT.md', 'w+').write(
    template.render(
        projects = projects
    )
)

template = jinja2.Template(open('docs/templates/TOOLS.md', 'r').read())
open('docs/TOOLS.md', 'w+').write(
    template.render(
        cli_doc = cli_doc
    )
)

template = jinja2.Template(open('docs/templates/ZOONIVERSE.md', 'r').read())
open('docs/ZOONIVERSE.md', 'w+').write(
    template.render(
        cli_doc = cli_doc
    )
)