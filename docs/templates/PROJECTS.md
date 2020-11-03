# List of available projects

| Name | Authors | Location | Recordings | Audio length (hours) | Status |
|------|---------|----------|------------|----------------------|--------|
{% for project in projects -%}
| **{{project.name}}** | {{project.authors}} | `{{project.location}}` | {project.recordings} | {{project.duration|round|int}} | {project.status} | 
{% endfor %}
