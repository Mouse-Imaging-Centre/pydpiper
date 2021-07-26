
import os
import sys

from pydpiper.core.arguments import (execution_parser, # lsq6_parser,
                                     lsq12_parser, registration_parser, application_parser, parse, CompoundParser)
from pydpiper.core.files import ImgAtom
from pydpiper.execution.application import execute
from pydpiper.minc.registration import lsq12_pairwise

# TODO why do these smaller pipelines need to be split up by LSQ12/NLIN when we have 'reg strategies'?
# Inability to know whether we're parsing a linear or nonlinear proto??
from pydpiper.pipelines.MAGeT import get_affine_registration_module


def LSQ12_pipeline(options):

    output_dir    = options.application.output_directory
    pipeline_name = options.application.pipeline_name

    # TODO this is tedious and annoyingly similar to the registration chain and MBM and LSQ6 ...
    processed_dir = os.path.join(output_dir, pipeline_name + "_processed")
    lsq12_dir     = os.path.join(output_dir, pipeline_name + "_lsq12")

    reg_algorithms = get_affine_registration_module(options.registration.image_algorithms, options.lsq12.reg_method)

    resolution = (options.registration.resolution  # TODO does using the finest resolution here make sense?
                  or min([reg_algorithms.Algorithms.get_resolution_from_file(f) for f in options.application.files]))

    imgs = [ImgAtom(f, pipeline_sub_dir=processed_dir) for f in options.application.files]

    # TODO check input files as in MBM etc. ...

    # determine LSQ12 settings by overriding defaults with
    # any settings present in protocol file, if it exists
    # could add a hook to print a message announcing completion, output files,
    # add more stages here to make a CSV

    return lsq12_pairwise(imgs, image_algorithms=reg_algorithms, lsq12_conf=options.lsq12,
                          use_robust_averaging=False,  # TODO currently an nlin option - move to options.registration?
                          lsq12_dir=lsq12_dir, resolution=resolution)

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
