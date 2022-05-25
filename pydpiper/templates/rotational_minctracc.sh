rotational_minctracc.py
  -t {{ conf.temp_dir }}
  -w {{ w_translation_stepsizes|join(",") }}
  -s {{ resample_stepsize }}
  -g {{ registration_stepsize }}
  -r {{ conf.rotational_range }}
  -i {{ conf.rotational_interval }}
  --simplex {{ simplex }}
  {{ blurred_src }} {{ blurred_dest }}
  {{ out_xfm }} /dev/null
  {% if target_mask %} -m {{ target_mask }} {% endif %}
  {% if source_mask %} --source-mask {{ source_mask }} {% endif %}
