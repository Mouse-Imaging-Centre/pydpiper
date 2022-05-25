
import os

import shlex

from jinja2 import (Environment, ChoiceLoader, FileSystemLoader, PackageLoader,
                    select_autoescape, StrictUndefined)


# the type of this definition could be changed if the type of non-templated commands is changed accordingly
def rendered_template_to_command(s):
    return shlex.split(s)

template_path = os.getenv("PYDPIPER_TEMPLATE_PATH")

if template_path:
    loaders = [FileSystemLoader(d) for d in template_path.split(":")]
    loader = ChoiceLoader([FileSystemLoader(d) for d in template_path.split(":")] +
                          [PackageLoader("pydpiper")])
else:
    loader = PackageLoader("pydpiper")

# a global for now ...
templating_env = Environment(
    loader=loader,
    autoescape=False,
    undefined=StrictUndefined
)

