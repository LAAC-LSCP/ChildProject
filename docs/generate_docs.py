from ChildProject.ChildProject import ChildProject, IndexColumn
import jinja2

formatting = jinja2.Template(open('docs/FORMATTING_TEMPLATE.md', 'r').read())
open('docs/FORMATTING.md', 'w+').write(
    formatting.render(children = ChildProject.CHILDREN_COLUMNS, recordings = ChildProject.RECORDINGS_COLUMNS)
)