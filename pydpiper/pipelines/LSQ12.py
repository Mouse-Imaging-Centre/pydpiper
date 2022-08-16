import csv
import os
import sys

from pydpiper.core.arguments import (AnnotatedParser, execution_parser, # lsq6_parser,
                                     lsq12_parser, registration_parser, application_parser, parse, CompoundParser)
from pydpiper.execution.application import execute
from pydpiper.minc.registration import (lsq12_pairwise, LSQ12Conf,
                                        default_lsq12_multilevel_minctracc,
                                        parse_minctracc_nonlinear_protocol_file, get_resolution_from_file)
from pydpiper.minc.files import MincAtom
from pydpiper.pipelines.MAGeT import get_imgs


def LSQ12_pipeline(options):

    output_dir    = options.application.output_directory
    pipeline_name = options.application.pipeline_name

    # TODO this is tedious and annoyingly similar to the registration chain and MBM and LSQ6 ...
    processed_dir = os.path.join(output_dir, pipeline_name + "_processed")
    lsq12_dir     = os.path.join(output_dir, pipeline_name + "_lsq12")

    imgs = get_imgs(options.application)

    resolution = (options.registration.resolution  # TODO does using the finest resolution here make sense?
                  or min([get_resolution_from_file(f.path) for f in imgs]))

    # determine LSQ12 settings by overriding defaults with
    # any settings present in protocol file, if it exists
    # could add a hook to print a message announcing completion, output files,
    # add more stages here to make a CSV

    return lsq12_pairwise(imgs, lsq12_conf=options.lsq12, lsq12_dir=lsq12_dir, resolution=resolution)

def main(args):
    # this is probably too much heavy machinery since the options aren't tree-shaped; could just use previous style
    p = CompoundParser(
          [execution_parser,
           application_parser,
           registration_parser,  #cast=RegistrationConf),
           lsq12_parser])

    options = parse(p, args[1:])
    stages = LSQ12_pipeline(options).stages
    execute(stages, options)

if __name__ == '__main__':
    main(sys.argv)
