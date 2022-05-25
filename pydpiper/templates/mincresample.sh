mincresample -clobber -2
  {% if interpolation %} -{{ interpolation.name }} {% endif %}
  {% if invert %} -invert {% endif %}
  {{ extra_flags|join(" ") }}
  {# TODO check for transform (and maybe id?) here #}
  -transform {{ xfm.path }}
  -like {{ like.path }}
  {{ img.path }} {{ outf.path }}
