
import os
import sys

from pydpiper.core.arguments import (execution_parser, # lsq6_parser,
                                     lsq12_parser, registration_parser, application_parser, parse, CompoundParser)
from pydpiper.core.files import ImgAtom
from pydpiper.execution.application import execute
from pydpiper.minc.ANTS import ANTS, ANTS_ITK
from pydpiper.minc.registration import (lsq12_pairwise, LSQ12Conf,
                                        default_lsq12_multilevel_minctracc,
                                        parse_minctracc_nonlinear_protocol_file,
                                        MincAlgorithms, MultilevelPairwiseRegistration,
                                        MINCTRACC, minctracc, MINCTRACC_LSQ12)
from pydpiper.minc.files import MincAtom
from pydpiper.pipelines.MAGeT import get_affine_registration_module, get_imgs
from pydpiper.itk.tools import Algorithms as ITKAlgorithms
# TODO why do these smaller pipelines need to be split up by LSQ12/NLIN when we have 'reg strategies'?
# Inability to know whether we're parsing a linear or nonlinear proto??


def LSQ12_pipeline(options):

    output_dir    = options.application.output_directory
    pipeline_name = options.application.pipeline_name

    # TODO this is tedious and annoyingly similar to the registration chain and MBM and LSQ6 ...
    processed_dir = os.path.join(output_dir, pipeline_name + "_processed")
    lsq12_dir     = os.path.join(output_dir, pipeline_name + "_lsq12")

<<<<<<< HEAD
    imgs = get_imgs(options.application)

    resolution = (options.registration.resolution  # TODO does using the finest resolution here make sense?
                  or min([get_resolution_from_file(f.path) for f in imgs]))

    try:
      reg_algorithms = { ('minc', 'ANTS') : ANTS,
                         ('minc', 'minctracc') : MINCTRACC_LSQ12,
                         ('itk',  'ANTS') : ANTS_ITK }[(options.registration.image_algorithms,
                                                        options.lsq12.registration_method)]
    except KeyError:
      raise KeyError("unsupported combination of `options.registration.image_algorithms` and `options.lsq12.registration_method` specified")
    #class Reg(MultilevelPairwiseRegistration):
    #    Algorithms = algorithms
=======
    reg_algorithms = get_affine_registration_module(options.registration.image_algorithms, options.lsq12.reg_method)
>>>>>>> [WIP] preview of MBM changes; broken right now

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


def application(): return main(sys.argv)
