#!/usr/bin/env python3

import os

from configargparse import Namespace

from pydpiper.core.files import ImgAtom
from pydpiper.core.stages import Stages, Result
from pydpiper.itk.demons import DEMONS
from pydpiper.minc.ANTS import ANTS_MINC, ANTS_ITK

from pydpiper.minc.analysis import determinants_at_fwhms

from pydpiper.minc.registration import (MINCTRACC)
from pydpiper.execution.application import mk_application
from pydpiper.core.arguments import (nlin_parser, stats_parser)


# TODO in some sense all the code here (as with LSQ6_pipeline, LSQ12_pipeline) is redundant:
# `NLIN_pipeline` should be expressible as a special case of `mbm_pipeline` by turning off certain parts
# and setting certain parameters appropriately
from pydpiper.minc.registration_strategies import get_model_building_procedure
from pydpiper.pipelines.MAGeT import (get_imgs)


def NLIN_pipeline(options):

    output_dir    = options.application.output_directory
    pipeline_name = options.application.pipeline_name

    # TODO this is tedious and annoyingly similar to the registration chain and MBM and LSQ6 ...
    #processed_dir = os.path.join(output_dir, pipeline_name + "_processed")
    nlin_dir      = os.path.join(output_dir, pipeline_name + "_nlin")

    imgs = get_imgs(options.application)
    #image_algorithms = get_algorithms(options.registration)

    try:
        reg_algorithms = {
                           ('minc', 'ANTS') : ANTS_MINC,
                           ('minc', 'minctracc'): MINCTRACC,
                           ('itk', 'ANTS') : ANTS_ITK,
                           ('itk', 'demons') : DEMONS,
                         }[(options.registration.image_algorithms,
                                                      options.nlin.reg_method)]
    except KeyError:
        raise KeyError(
            "unsupported combination of `options.registration.image_algorithms` and `options.nlin.reg_method` specified")

    resolution = (options.registration.resolution  # TODO does using the finest resolution here make sense?
                  or min([reg_algorithms.Algorithms.get_resolution_from_file(f) for f in options.application.files]))

    initial_target_mask = ImgAtom(options.nlin.target_mask) if options.nlin.target_mask else None
    initial_target = ImgAtom(options.nlin.target, mask=initial_target_mask)

    #nlin_module = get_nonlinear_component(reg_method=options.nlin.reg_method)

    nlin_build_model_component = get_model_building_procedure(options.nlin.reg_strategy,
                                                              reg_module=reg_algorithms)

    nlin_conf = (nlin_build_model_component.parse_build_model_protocol(
                                              options.nlin.nlin_protocol, resolution=resolution)
                 if options.nlin.nlin_protocol is not None
                 else nlin_build_model_component.get_default_build_model_conf(resolution=resolution))

    s = Stages()

    nlin_result = s.defer(nlin_build_model_component.build_model(imgs=imgs,
                                                  initial_target=initial_target,
                                                  conf=nlin_conf,
                                                  nlin_dir=nlin_dir,
                                                  use_robust_averaging=options.nlin.use_robust_averaging,
                                                  nlin_prefix=""))

    inverted_xfms = [s.defer(nlin_build_model_component.Algorithms.invert_xfmhandler(xfm)) for xfm in nlin_result.output]

    if options.stats.calc_stats:

        # TODO use defer_map
        determinants = [s.defer(nlin_build_model_component.Algorithms.log_determinant(xfm))
                        for xfm in nlin_result.output]

        # TODO this was the old way -- restore blurred/unblurred determinants!!
        #determinants = s.defer(determinants_at_fwhms(
        #                          xfms=inverted_xfms,
        #                          inv_xfms=nlin_result.output,
        #                          blur_fwhms=options.stats.stats_kernels))

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
