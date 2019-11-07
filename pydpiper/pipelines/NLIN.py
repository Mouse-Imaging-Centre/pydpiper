#!/usr/bin/env python3

import os

from configargparse import Namespace
from pydpiper.core.stages import Stages, Result

from pydpiper.minc.analysis import determinants_at_fwhms

from pydpiper.minc.registration import (get_resolution_from_file, nlin_build_model,
                                        get_nonlinear_configuration_from_options, invert_xfmhandler)
from pydpiper.minc.files import MincAtom
from pydpiper.execution.application import mk_application
from pydpiper.core.arguments import (nlin_parser, stats_parser)


# TODO in some sense all the code here (as with LSQ6_pipeline, LSQ12_pipeline) is redundant:
# `NLIN_pipeline` should be expressible as a special case of `mbm_pipeline` by turning off certain parts
# and setting certain parameters appropriately

def NLIN_pipeline(options):

    if options.application.files is None:
        raise ValueError("Please, some files! (or try '--help')")  # TODO make a util procedure for this

    output_dir    = options.application.output_directory
    pipeline_name = options.application.pipeline_name

    # TODO this is tedious and annoyingly similar to the registration chain and MBM and LSQ6 ...
    processed_dir = os.path.join(output_dir, pipeline_name + "_processed")
    nlin_dir      = os.path.join(output_dir, pipeline_name + "_nlin")

    resolution = (options.registration.resolution  # TODO does using the finest resolution here make sense?
                  or min([get_resolution_from_file(f) for f in options.application.files]))

    imgs = [MincAtom(f, pipeline_sub_dir=processed_dir) for f in options.application.files]

    # determine NLIN settings by overriding defaults with
    # any settings present in protocol file, if it exists
    # could add a hook to print a message announcing completion, output files,
    # add more stages here to make a CSV

    initial_target_mask = MincAtom(options.nlin.target_mask) if options.nlin.target_mask else None
    initial_target = MincAtom(options.nlin.target, mask=initial_target_mask)

    full_hierarchy = get_nonlinear_configuration_from_options(nlin_protocol=options.nlin.nlin_protocol,
                                                              flag_nlin_protocol=next(iter(options.nlin.flags_.nlin_protocol)),
                                                              reg_method=options.nlin.reg_method,
                                                              file_resolution=resolution)

    s = Stages()

    nlin_result = s.defer(nlin_build_model(imgs, initial_target=initial_target, conf=full_hierarchy, nlin_dir=nlin_dir))

    # TODO return these?
    inverted_xfms = [s.defer(invert_xfmhandler(xfm)) for xfm in nlin_result.output]

    if options.stats.calc_stats:
        # TODO: put the stats part behind a flag ...

        determinants = [s.defer(determinants_at_fwhms(
                                  xfm=inv_xfm,
                                  inv_xfm=xfm,
                                  blur_fwhms=options.stats.stats_kernels))
                        for xfm, inv_xfm in zip(nlin_result.output, inverted_xfms)]

        return Result(stages=s,
                      output=Namespace(nlin_xfms=nlin_result,
                                       avg_img=nlin_result.avg_img,
                                       determinants=determinants))
    else:
        # there's no consistency in what gets returned, yikes ...
        return Result(stages=s, output=Namespace(nlin_xfms=nlin_result, avg_img=nlin_result.avg_img))


#_nlin_parser = _mk_nlin_parser(ArgParser(add_help=False))
#_nlin_parser.add_argument("--target", dest="target",
#                          type=str,
#                          help="Starting target for non-linear alignment. (Often in 'lsq12 space')."
#                               "[Default = %(default)s]")
nlin_parser.parser.argparser.add_argument("--target", dest="target",
                                          type=str,
                                          help="Starting target for non-linear alignment. (Often in 'lsq12 space')."
                                               "[Default = %(default)s]")
nlin_parser.parser.argparser.add_argument("--target-mask", dest="target_mask",
                                          type=str, default=None,
                                          help="Starting target for non-linear alignment. (Often in 'lsq12 space')."
                                               "[Default = %(default)s]")


nlin_application = mk_application(parsers=[nlin_parser, stats_parser], #, namespace='nlin')],
                                  pipeline=NLIN_pipeline)


if __name__ == "__main__":
    nlin_application()