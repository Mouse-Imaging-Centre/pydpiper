nu_estimate -clobber
  -iterations 100 -stop 0.001 -fwhm 0.15 -shrink 4 -lambda 5.0e-02
  -distance {{ distance }}
  {% if mask %} -mask {{ mask.path }} {% endif %}
  {{ src }} {{ out }}
