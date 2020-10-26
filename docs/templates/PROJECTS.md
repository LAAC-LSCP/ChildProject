# List of available projects

| Name | Authors | Location | Audio length (hours) |
|------|-------------|------------|--------|
{% for project in projects -%}
| **{{project.name}}** | {{project.authors}} | `{{project.location}}` | {{project.duration|round|int}} | 
{% endfor %}
