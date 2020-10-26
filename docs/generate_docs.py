from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
import jinja2
import time

projects = [
    {'name': 'Namibia', 'authors': "Gandhi", 'location': '/scratch1/data/laac_data/namibia', 'duration': 5214771/3600},
    {'name': 'Solomon', 'authors': "Sarah", 'location': '/scratch1/data/laac_data/solomon', 'duration': 21435406/3600}
]

formatting = jinja2.Template(open('docs/templates/FORMATTING.md', 'r').read())
open('docs/FORMATTING.md', 'w+').write(
    formatting.render(
        children = ChildProject.CHILDREN_COLUMNS,
        recordings = ChildProject.RECORDINGS_COLUMNS,
        input_annotations = [c for c in AnnotationManager.INDEX_COLUMNS if not c.generated],
        annotation_segments = AnnotationManager.SEGMENTS_COLUMNS,
        annotations = [c for c in AnnotationManager.INDEX_COLUMNS if (c.generated or c.required)]
    )
)

formatting = jinja2.Template(open('docs/templates/PROJECTS.md', 'r').read())
open('docs/PROJECTS.md', 'w+').write(
    formatting.render(
        projects = projects
    )
)