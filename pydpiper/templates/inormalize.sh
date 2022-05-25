inormalize -clobber
  -const {{ conf.const }}
  -{{ conf.method.name }}
  {% if mask %} -mask {{ mask }} {% endif %}
  {{ src }} {{ out }}
