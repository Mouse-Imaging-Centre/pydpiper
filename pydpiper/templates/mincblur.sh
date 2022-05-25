mincblur -clobber -no_apodize -fwhm {{ fwhm }} {{ img.path }} {{ out_img.path[:-9] }} {% if gradient %} -gradient {% endif %} 
