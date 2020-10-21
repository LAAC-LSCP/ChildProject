from ChildProject.projects import ChildProject
from ChildProject.annotations import AnnotationManager
import jinja2

formatting = jinja2.Template(open('docs/FORMATTING_TEMPLATE.md', 'r').read())
open('docs/FORMATTING.md', 'w+').write(
    formatting.render(
        children = ChildProject.CHILDREN_COLUMNS,
        recordings = ChildProject.RECORDINGS_COLUMNS,
        input_annotations = [c for c in AnnotationManager.INDEX_COLUMNS if not c.generated],
        annotation_segments = AnnotationManager.SEGMENTS_COLUMNS
    )
)