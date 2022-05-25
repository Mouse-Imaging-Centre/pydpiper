mincmath -clobber -2
  {% if const %} -const {{ const }} {% endif %}
  -{{ op }}
  {{ vols|join(" ") }}
  {{ outf }}
