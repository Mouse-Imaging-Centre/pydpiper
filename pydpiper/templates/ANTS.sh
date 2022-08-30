ANTS 3
  --number-of-affine-iterations 0
  {# TODO should similarity_cmds be generated in the template? #}
  {{ similarity_cmds|join(" ") }}
  -t {{ conf.transformation_model }}
  -r {{ conf.regularization }}
  -i {{ conf.iterations }}
  -o {{ out_xfm }}
  {% if conf.use_mask and fixed.mask %} -x {{ fixed.mask.path }} {% endif %}
