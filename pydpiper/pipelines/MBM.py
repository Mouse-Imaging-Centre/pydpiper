#!/usr/bin/env python3
import csv
import os.path

from configargparse import Namespace
from typing import List

from pydpiper.core.util import NamedTuple

from pydpiper.core.stages       import Result, Stages
#TODO fix up imports, naming, stuff in registration vs. pipelines, ...
from pydpiper.minc.files        import MincAtom
from pydpiper.minc.registration import (lsq6_nuc_inorm, lsq12_nlin_build_model, registration_targets,
                                        mincANTS_default_conf, MultilevelMincANTSConf, LSQ6Conf, LSQ12Conf,
                                        get_resolution_from_file, parse_mincANTS_protocol_file, concat_xfmhandlers)
from pydpiper.minc.analysis     import determinants_at_fwhms, StatsConf
from pydpiper.core.arguments    import (lsq6_parser, lsq12_parser, nlin_parser, stats_parser, CompoundParser,
                                        AnnotatedParser, NLINConf)
from pydpiper.execution.application    import mk_application


MBMConf = NamedTuple('MBMConf', [('lsq6',  LSQ6Conf),
                                 ('lsq12', LSQ12Conf),
                                 ('nlin',  NLINConf),
                                 ('stats', StatsConf)])

# TODO abstract out some common configuration functionality and think about argparsing ...


def mbm_pipeline(options : MBMConf):
    imgs = [MincAtom(name, pipeline_sub_dir=os.path.join(options.application.output_directory,
                                                         options.application.pipeline_name + "_processed"))
            for name in options.application.files]

    return mbm(imgs=imgs, options=options)

def mbm(imgs : List[MincAtom], options : MBMConf):

    # TODO could also allow pluggable pipeline parts e.g. LSQ6 could be substituted out for the modified LSQ6
    # for the kidney tips, etc...
    output_dir    = options.application.output_directory
    pipeline_name = options.application.pipeline_name

    # TODO this is tedious and annoyingly similar to the registration chain ...

    #processed_dir = os.path.join(output_dir, pipeline_name + "_processed")
    lsq12_dir = os.path.join(output_dir, pipeline_name + "_lsq12")
    nlin_dir = os.path.join(output_dir, pipeline_name + "_nlin")

    s = Stages()

    if len(imgs) == 0:
        raise ValueError("Please, some files!")

    # TODO this is quite tedious and duplicates stuff in the registration chain ...
    resolution = (options.registration.resolution or
                  get_resolution_from_file(
                      registration_targets(lsq6_conf=options.mbm.lsq6,
                                           app_conf=options.application).registration_standard.path))
    options.registration = options.registration.replace(resolution=resolution)

    # FIXME: why do we have to call registration_targets *outside* of lsq6_nuc_inorm? is it just because of the extra
    # options required?
    targets = registration_targets(lsq6_conf=options.mbm.lsq6,
                                   app_conf=options.application,
                                   first_input_file=options.application.files[0])

    lsq6_result = s.defer(lsq6_nuc_inorm(imgs=imgs,
                                         resolution=resolution,
                                         registration_targets=targets,
                                         lsq6_options=options.mbm.lsq6))

    if options.mbm.nlin.nlin_protocol:
        with open(options.mbm.nlin.nlin_protocol, 'r') as f:
            reader = csv.reader(f, delimiter=";")  # TODO this should be automatic, not part of MBM, etc.
            full_hierarchy = MultilevelMincANTSConf([c.replace(file_resolution=options.registration.resolution)
                                                     for c in parse_mincANTS_protocol_file(reader).confs])
    else:
        # FIXME just a placeholder ... supply a reasonable set of configurations here ...
        conf1 = mincANTS_default_conf.replace(file_resolution=options.registration.resolution,
                                              iterations="100x100x100x0")
        conf2 = mincANTS_default_conf.replace(file_resolution=options.registration.resolution)
        full_hierarchy = MultilevelMincANTSConf([conf1, conf2])

    lsq12_nlin_result = s.defer(lsq12_nlin_build_model(imgs=[xfm.resampled for xfm in lsq6_result],
                                                       resolution=resolution,
                                                       lsq12_dir=lsq12_dir,
                                                       nlin_dir=nlin_dir,
                                                       lsq12_conf=options.mbm.lsq12,
                                                       nlin_conf=full_hierarchy))

    determinants = [s.defer(determinants_at_fwhms(xfm, blur_fwhms=options.mbm.stats.stats_kernels))
                    for xfm in lsq12_nlin_result.output]

    overall_xfms = [s.defer(concat_xfmhandlers([rigid_xfm, nlin_xfm]))
                    for rigid_xfm, nlin_xfm in zip(lsq6_result, lsq12_nlin_result.output)]

    return Result(stages=s,
                  output=Namespace(rigid_xfms=lsq6_result,
                                   lsq12_nlin_xfms=lsq12_nlin_result,
                                   overall_xfms=overall_xfms,
                                   # TODO transpose these fields?
                                   avg_img=lsq12_nlin_result.avg_img,  # inconsistent w/ WithAvgImgs[...]-style outputs
                                   determinants=determinants))  # TODO better naming

# TODO write a function 'ordinary_parser' ...
mbm_parser = CompoundParser(
               [lsq6_parser,
                lsq12_parser,
                nlin_parser,
                stats_parser])

# TODO could make an MBMConf and cast to it ...
mbm_application = mk_application(parsers=[AnnotatedParser(parser=mbm_parser, namespace='mbm')],
                                 pipeline=mbm_pipeline)


# stuff to do:
# - reduce no. of args to get_initial_targets by taking the whole conf objects instead of small pieces
# - write fn to get resolution (with similar caveats)
# - fix bug re: positional arguments

if __name__ == "__main__":
    mbm_application()
