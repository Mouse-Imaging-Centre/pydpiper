
import os

from configargparse import ArgParser

# cmd-line version of `run`
def foo():
    default_config_file = os.getenv("PYDPIPER_CONFIG_FILE")
    parser  = ArgParser(default_config_files = [default_config_file] if default_config_file else [])
    parser.addExecutorArgumentGroup()
    parser.addApplicationArgumentGroup()
    version = get_distribution("pydpiper").version


"""Actually run the given pipeline (don't deal with help menus, option parsing, etc.).
Everything we need should be inside the stages themselves (e.g., hooks setting memory requirements)
and the options (e.g., queue information)."""
def run(stages, options):
    pass
    # make dirs based on stuff in stages

    
