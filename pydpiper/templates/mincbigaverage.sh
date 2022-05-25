mincbigaverage -clobber
  {% if avgnum %} --avgnum {{ avgnum }} {% endif %}
  {# TODO move `tmpdir` outside of `robust` #}
  {% if robust %}
    --robust
    {% if tmpdir %} --tmpdir {{ tmpdir }} {% endif %}
  {% endif %}
  {% if sdfile %} --sdfile {{ sdfile }} {% endif %}
  {{ imgs|join(" ") }}
  {{ avg_path }}
