#!/usr/bin/env python3

import os.path

from pydpiper.core.util import NamedTuple

from pydpiper.core.stages       import Result, Stages
#TODO fix up imports, naming, stuff in registration vs. pipelines, ...
from pydpiper.minc.files        import MincAtom
from pydpiper.minc.registration import (lsq6_nuc_inorm, lsq12_nlin_build_model, registration_targets,
                                        mincANTS_default_conf, MultilevelMincANTSConf, LSQ6Conf, LSQ12Conf,
                                        get_resolution_from_file)
from pydpiper.minc.analysis     import determinants_at_fwhms, StatsConf
from pydpiper.core.arguments    import (lsq6_parser, lsq12_parser, nlin_parser, stats_parser, CompoundParser,
                                        AnnotatedParser, NLINConf)
from pydpiper.execution.application    import mk_application


MBMConf = NamedTuple('MBMConf', [('lsq6',  LSQ6Conf),
                                 ('lsq12', LSQ12Conf),
                                 ('nlin',  NLINConf),
                                 ('stats', StatsConf)])

# TODO abstract out some common configuration functionality and think about argparsing ...


def mbm(options : MBMConf):

    # TODO could also allow pluggable pipeline parts e.g. LSQ6 could be substituted out for the modified LSQ6
    # for the kidney tips, etc...
    output_dir    = options.application.output_directory
    pipeline_name = options.application.pipeline_name

    # TODO this is tedious and annoyingly similar to the registration chain ...
    processed_dir = os.path.join(output_dir, pipeline_name + "_processed")
    lsq12_dir = os.path.join(output_dir, pipeline_name + "_lsq12")
    nlin_dir = os.path.join(output_dir, pipeline_name + "_nlin")

    imgs = [MincAtom(name, pipeline_sub_dir=processed_dir) for name in options.application.files]

    s = Stages()

    if len(options.application.files) == 0:
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

    # FIXME just a placeholder ... TODO: write procedure to convert input nlin_options into appropriate conf
    conf1 = mincANTS_default_conf.replace(file_resolution=options.registration.resolution,
                                          iterations="100x100x100x0")
    conf2 = mincANTS_default_conf.replace(file_resolution=options.registration.resolution)
    full_hierarchy = MultilevelMincANTSConf([conf1, conf2])

    lsq12_nlin_result = s.defer(lsq12_nlin_build_model(imgs=[xfm.resampled for xfm in lsq6_result],
                                                       resolution=resolution,
                                                       lsq12_dir=lsq12_dir,
                                                       nlin_dir=nlin_dir,
                                                       lsq12_conf=options.mbm.lsq12,
                                                       nlin_conf=full_hierarchy))  # FIXME use a real conf

    determinants = [s.defer(determinants_at_fwhms(xfm, blur_fwhms=options.mbm.stats.stats_kernels))
                    for xfm in lsq12_nlin_result.output]

    return Result(stages=s, output=determinants)  # TODO: add more outputs

# TODO write a function 'ordinary_parser' ...
mbm_parser = CompoundParser(
               [lsq6_parser,
                lsq12_parser,
                nlin_parser,
                stats_parser])

# TODO could make an MBMConf and cast to it ...
mbm_application = mk_application(parsers=[AnnotatedParser(parser=mbm_parser, namespace='mbm')],
                                 pipeline=mbm)


# stuff to do:
# - reduce no. of args to get_initial_targets by taking the whole conf objects instead of small pieces
# - write fn to get resolution (with similar caveats)
# - fix bug re: positional arguments

if __name__ == "__main__":
    mbm_application()
