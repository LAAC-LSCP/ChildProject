from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
import jinja2
import time

# recursively get the sum of durations of each audio in the current directory :
# find . -type f -execdir soxi -D {} \; | awk '{s+=$1} END {printf "%.0f", s}'
projects = [
    {'name': 'Namibia', 'status': 'ready', 'authors': "Gandhi", 'location': '/scratch1/data/laac_data/namibia', 'recordings': 113, 'duration': 5214771/3600},
    {'name': 'Solomon', 'status': 'ready', 'authors': "Sarah", 'location': '/scratch1/data/laac_data/solomon', 'recordings': 388, 'duration': 21435406/3600},
    {'name': 'Tsimane 2017', 'status': 'awaiting validation', 'authors': "", 'location': '/scratch1/data/laac_data/tsimane2017', 'recordings': 41, 'duration': 2001601/3600},
    {'name': 'png 2019', 'status': 'awaiting data', 'authors': "", 'location': '/scratch1/data/laac_data/rossel_png_2019', 'recordings': 51, 'duration': 2737005/3600},
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

template = jinja2.Template(open('docs/templates/PROJECTS.md', 'r').read())
open('docs/PROJECTS.md', 'w+').write(
    template.render(
        projects = projects
    )
)