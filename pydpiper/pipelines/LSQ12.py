import argparse
import os
import sys

from pydpiper.atoms_and_modules.LSQ12 import addLSQ12ArgumentGroup
# TODO: remove the *! :-)
from pydpiper.minc.registration import *
from pydpiper.core.util import directories
from pydpiper.minc.files import MincAtom

def parse_LSQ12_protocol(f): # handle -> LSQ12Conf
    csvReader = csv.reader(f, delimiter=';', skipinitialspace=True)
    d = {}
    for l in csvReader:
        d[l[0]] = l[1:]
    #for k, v in d.iteritems():
    #    if k

def parse_protocol(f, constructor):
    """Parse a protocol file and, if possible, return a protocol object
    of type `constructor` (e.g., LSQ12Conf) - though one which may be only partly
    filled in."""
    csvReader = csv.reader(f, delimiter=';', skipinitialspace=True)
    d = {}
    for l in csvReader:
        d[l[0]]

# Q. should a pipeline (post-cmdline-parsing) take a `files` arg
# (an array of minc files) or just an options arg containing filenames?
def LSQ12_pipeline(options):
    # note that we already have a module called 'lsq12' but that one is intended to
    # be called within a pipeline; this one sets up mapped mincfiles, cmd-line stuff, ...
    imgs = [MincAtom(f) for f in options.files]
    if options.reg_protocol:
        pass # opts = parse_lsq12_protocol
    ### determine LSQ12 settings by overriding defaults with
    ### any settings present in protocol file, if it exists
    conf = None
    stages, _ = lsq12(imgs, conf, curr_dir=options.pipeline_directory or os.getcwd())
    # could add a hook to print a message announcing completion, output files,
    # add more stages here to make a CSV
    return stages

def main(args):
    parser = argparse.ArgumentParser() # TODO setup arguments
    addLSQ12ArgumentGroup(parser)
    options = parser.parse_args(args)
    stages = LSQ12_pipeline(options)
    for d in directories(stages):
        try:
            os.mkdir(d)
        except OSError as e:  # Python2 hack for exists_ok=True
            if not re.match('File exists', e.strerror):
                raise
    #execute(pipeline(stages))


if __name__ == '__main__':
    main(sys.argv)
