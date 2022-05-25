mincaverage -clobber {% if normalize %} -normalize {% endif %} -max_buffer_size_in_kb=409620 -sdfile {{ sdfile }} {{ additional_flags }} {{ imgs|join(" ") }} {{ avg }}
