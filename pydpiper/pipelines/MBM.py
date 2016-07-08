#!/usr/bin/env python3

import os.path

from configargparse import Namespace
from typing import List

from pydpiper.core.util import NamedTuple

from pydpiper.core.stages       import Result, Stages

#TODO fix up imports, naming, stuff in registration vs. pipelines, ...
from pydpiper.minc.files        import MincAtom
from pydpiper.minc.registration import (lsq6_nuc_inorm, lsq12_nlin_build_model, registration_targets,
                                        LSQ6Conf, LSQ12Conf, get_resolution_from_file, concat_xfmhandlers,
                                        get_nonlinear_configuration_from_options,
                                        invert_xfmhandler, check_MINC_input_files)
from pydpiper.minc.analysis     import determinants_at_fwhms, StatsConf
from pydpiper.core.arguments    import (lsq6_parser, lsq12_parser, nlin_parser, stats_parser, CompoundParser,
                                        AnnotatedParser, NLINConf)
from pydpiper.execution.application    import mk_application


MBMConf = NamedTuple('MBMConf', [('lsq6',  LSQ6Conf),
                                 ('lsq12', LSQ12Conf),
                                 ('nlin',  NLINConf),
                                 ('stats', StatsConf)])


def mbm_pipeline(options : MBMConf):
    imgs = [MincAtom(name, pipeline_sub_dir=os.path.join(options.application.output_directory,
                                                         options.application.pipeline_name + "_processed"))
            for name in options.application.files]

    check_MINC_input_files([img.path for img in imgs])

    return mbm(imgs=imgs, options=options,
               prefix=options.application.pipeline_name,
               output_dir=options.application.output_directory)


def mbm(imgs : List[MincAtom], options : MBMConf, prefix : str, output_dir : str = ""):

    # TODO could also allow pluggable pipeline parts e.g. LSQ6 could be substituted out for the modified LSQ6
    # for the kidney tips, etc...

    # TODO this is tedious and annoyingly similar to the registration chain ...
    lsq6_dir  = os.path.join(output_dir, prefix + "_lsq6")
    lsq12_dir = os.path.join(output_dir, prefix + "_lsq12")
    nlin_dir  = os.path.join(output_dir, prefix + "_nlin")

    s = Stages()

    if len(imgs) == 0:
        raise ValueError("Please, some files!")

    # FIXME: why do we have to call registration_targets *outside* of lsq6_nuc_inorm? is it just because of the extra
    # options required?
    targets = registration_targets(lsq6_conf=options.mbm.lsq6,
                                   app_conf=options.application,
                                   first_input_file=imgs[0].path)

    # TODO this is quite tedious and duplicates stuff in the registration chain ...
    resolution = (options.registration.resolution or
                  get_resolution_from_file(targets.registration_standard.path))
    options.registration = options.registration.replace(resolution=resolution)

    lsq6_result = s.defer(lsq6_nuc_inorm(imgs=imgs,
                                         resolution=resolution,
                                         registration_targets=targets,
                                         lsq6_dir=lsq6_dir,
                                         lsq6_options=options.mbm.lsq6))

    full_hierarchy = get_nonlinear_configuration_from_options(nlin_protocol=options.mbm.nlin.nlin_protocol,
                                                              reg_method=options.mbm.nlin.reg_method,
                                                              file_resolution=resolution)

    lsq12_nlin_result = s.defer(lsq12_nlin_build_model(imgs=[xfm.resampled for xfm in lsq6_result],
                                                       resolution=resolution,
                                                       lsq12_dir=lsq12_dir,
                                                       nlin_dir=nlin_dir,
                                                       nlin_prefix=prefix,
                                                       lsq12_conf=options.mbm.lsq12,
                                                       nlin_conf=full_hierarchy))

    inverted_xfms = [s.defer(invert_xfmhandler(xfm)) for xfm in lsq12_nlin_result.output]

    determinants = s.defer(determinants_at_fwhms(
                             xfms=inverted_xfms,
                             inv_xfms=lsq12_nlin_result.output,
                             blur_fwhms=options.mbm.stats.stats_kernels))

    overall_xfms = [s.defer(concat_xfmhandlers([rigid_xfm, nlin_xfm]))
                    for rigid_xfm, nlin_xfm in zip(lsq6_result, lsq12_nlin_result.output)]

    # the right way (but requires changes to the two-level code ...)
    #output = (pd.DataFrame({ "rigid_xfms"      : lsq6_result,
    #                         "lsq12_nlin_xfms" : lsq12_nlin_result.output,
    #                         "overall_xfms"    : overall_xfms }))
             # .merge(determinants)) # TODO! but need better indexing first ...

                            # TODO transpose these fields?})
                            #avg_img=lsq12_nlin_result.avg_img,  # inconsistent w/ WithAvgImgs[...]-style outputs
                           # "determinants"    : determinants })

    #output.avg_img = lsq12_nlin_result.avg_img
    #output.determinants = determinants   # TODO temporary - remove once incorporated properly into `output` proper
    # TODO add more of lsq12_nlin_result?

    return Result(stages=s, output=Namespace(rigid_xfms=lsq6_result,
                                             lsq12_nlin_xfms=lsq12_nlin_result, #.output,
                                             overall_xfms=overall_xfms,
                                             determinants=determinants,
                                             avg_img=lsq12_nlin_result.avg_img))

mbm_parser = CompoundParser(
               [lsq6_parser,
                lsq12_parser,
                nlin_parser,
                stats_parser])

# TODO cast to MBMConf?
mbm_application = mk_application(parsers=[AnnotatedParser(parser=mbm_parser, namespace='mbm')],
                                 pipeline=mbm_pipeline)

if __name__ == "__main__":
    mbm_application()
