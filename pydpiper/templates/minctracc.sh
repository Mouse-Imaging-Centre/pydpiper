minctracc -clobber -debug
  {% if lin_conf and lin_conf.objective %} -{{ lin_conf.objective.name }} {% endif %}
  {# TODO reorder/consolidate some stuff (e.g. lin_conf and nlin_conf stuff) to remove some conditionals #}
  {% if transform %} -transformation {{ transform.path }}
  {% elif transform_info %} {{ transform_info }}
  {% else %} -identity
  {% endif %}
  {% if lin_conf and lin_conf.transform_type %} -{{ lin_conf.transform_type.name }} {% endif %}
  {% if nlin_conf and nlin_conf.use_simplex %} -use_simplex {% endif %}
  -step {{ conf.step_sizes|join(" ") }}
  {% if lin_conf %}
     -simplex {{ lin_conf.simplex }}
     -tol {{ lin_conf.tolerance }}
     -w_shear {{ lin_conf.w_shear|join(" ") }}
     -w_scales {{ lin_conf.w_scales|join(" ") }}
     -w_rotations {{ lin_conf.w_rotations|join(" ") }}
     -w_translations {{ lin_conf.w_translations|join(" ") }}
  {% endif %}
  {% if nlin_conf %}
     -iterations {{ nlin_conf.iterations }} 
     -similarity_cost_ratio {{ nlin_conf.similarity }}
     -weight {{ nlin_conf.weight }}
     -stiffness {{ nlin_conf.stiffness }}
     -sub_lattice {{ nlin_conf.sub_lattice }}
     -lattice_diameter {{ lattice_diameter|join(" ") }}
     {# TODO should step-size -> lattice diameter setting login be in template or application code? #}
     -nonlinear {% if nlin_conf.objective %} {{ nlin_conf.objective.name }} {% endif %}
  {% endif %}
  {% if conf.use_masks and source_mask %} -source_mask {{ source_mask.path }} {% endif %}
  {% if conf.use_masks and target_mask %} -model_mask {{ target_mask.path }} {% endif %}
  {{ source.path }} {{ target.path }} {{ out_xfm }}
