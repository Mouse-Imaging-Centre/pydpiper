import csv
import os
import sys

from pydpiper.core.arguments import (AnnotatedParser, execution_parser, lsq6_parser,
                                     lsq12_parser, registration_parser, application_parser, parse, CompoundParser)
from pydpiper.execution.application import execute
from pydpiper.minc.registration import lsq12_pairwise, LSQ12Conf
from pydpiper.minc.files import MincAtom

# TODO why use this strange format instead of, e.g., csv?
def parse_LSQ12_protocol(f) -> LSQ12Conf:
    """Parse a protocol file and, if possible, return a protocol object
    of type `constructor` (e.g., LSQ12Conf) - though one which may be only partly
    filled in."""
    csvReader = csv.reader(f, delimiter=';', skipinitialspace=True)
    d = {}
    for l in csvReader:
        d[l[0]] = l[1:]
    keys = tuple(d.keys())
    raise NotImplementedError


    return default_LSQ12_conf.replace(**d)
    #for k, v in d.items():
    #    if k

# Q. should a pipeline (post-cmdline-parsing) take a `files` arg
# (an array of minc files) or just an options arg containing filenames?
def LSQ12_pipeline(options):
    # note that we already have a module called 'lsq12' but that one is intended to
    # be called within a pipeline; this one sets up mapped mincfiles, cmd-line stuff, ...
    imgs = [MincAtom(f) for f in options.files]
    if options.reg_protocol:
        pass  # opts = parse_lsq12_protocol
    # determine LSQ12 settings by overriding defaults with
    # any settings present in protocol file, if it exists
    conf = None
    # could add a hook to print a message announcing completion, output files,
    # add more stages here to make a CSV
    return lsq12_pairwise(imgs, conf, output_dir=options.pipeline_directory or os.getcwd())

def main(args):
    # this is probably too much heavy machinery since the options aren't tree-shaped; could just use previous style
    p = CompoundParser(
          [execution_parser,
           AnnotatedParser(parser=execution_parser, namespace='execution'),
           AnnotatedParser(parser=application_parser, namespace='application'),
           AnnotatedParser(parser=registration_parser, namespace='registration', cast=RegistrationConf),
           AnnotatedParser(parser=lsq6_parser, namespace='lsq6'),
           AnnotatedParser(parser=lsq12_parser, namespace='lsq12')])

    # TODO could abstract and then parametrize by prefix/ns ??
    options = parse(p, sys.argv[1:])
    stages = LSQ12_pipeline(options).stages
    execute(stages, options)

if __name__ == '__main__':
    main(sys.argv)
