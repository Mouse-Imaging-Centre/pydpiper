#!/usr/bin/env python3

import os.path
import configargparse
import sys

from pydpiper.core.stages       import Result, Stages
#TODO fix up imports, naming, stuff in registration vs. pipelines, ...
from pydpiper.minc.files        import MincAtom
from pydpiper.minc.registration import lsq6_nuc_inorm, lsq12_nlin_build_model
from pydpiper.minc.analysis     import determinants_at_fwhms
from pydpiper.core.arguments    import (lsq6_parser, lsq12_parser, nlin_parser, stats_parser, CompoundParser,
                                        AnnotatedParser, application_parser, parse, registration_parser,
                                        execution_parser)
from pydpiper.execution.application    import execute

# TODO abstract out some common configuration functionality and think about argparsing ...


#class MBMConf(object):
#  pass
  # ['pipeline_name',
  #  'execution_conf',
  #  'lsq6_conf',
  #  'lsq12_conf',
  #  'nlin_conf',
  #  'stats_conf'
  #  ])

# MBM_default_conf = MBMConf(
#   pipeline_name='MBM',
#   execution_conf=None,
#   lsq6_conf=LSQ6_default_conf,
#   lsq12_conf=LSQ12_default_conf,
#   nlin_conf=mincANTS_default_conf,
#   stats_conf={'blur_fwhms' : [None, 0.056]})

def mbm(options):

    # TODO could also allow pluggable pipeline parts e.g. LSQ6 could be substituted out for the modified LSQ6
    # for the kidney tips, etc...
    pipeline_dir = os.path.join('.', options.application.pipeline_name)
    imgs = [MincAtom(name, pipeline_sub_dir=pipeline_dir) for name in options.application.files]

    s = Stages()

    resolution = options.registration.resolution

    lsq6_result = s.defer(lsq6_nuc_inorm(imgs=imgs,
                                         resolution=resolution,
                          # FIXME: why do we have to call get_registration_targets *outside* of lsq6_nuc_inorm?
                                         registration_targets=NotImplemented,
                                         lsq6_options=options.mbm.lsq6))

    lsq12_nlin_result = s.defer(lsq12_nlin_build_model(
        imgs=[xfm.resampled for xfm in lsq6_result], resolution=resolution,
        lsq12_conf=options.mbm.lsq12, nlin_conf=options.mbm.nlin))

    determinants = [s.defer(determinants_at_fwhms(xfm, blur_fwhms=options.stats.stats_kernels))
                    for xfm in lsq12_nlin_result.xfms]

    return Result(stages=s, output=determinants)  # TODO: add more outputs

mbm_parser = CompoundParser(
               [AnnotatedParser(parser=lsq6_parser, namespace='lsq6'),
                AnnotatedParser(parser=lsq12_parser, namespace='lsq12'),
                AnnotatedParser(parser=nlin_parser, namespace='nlin'),
                AnnotatedParser(parser=stats_parser, namespace='stats')])

if __name__ == "__main__":
    p = CompoundParser(
          [AnnotatedParser(parser=mbm_parser, namespace='mbm'),
           AnnotatedParser(parser=application_parser, namespace='application'),
           AnnotatedParser(parser=registration_parser, namespace='registration'),
           AnnotatedParser(parser=execution_parser, namespace='execution')
           ])
    options = parse(p, sys.argv[1:])
    stages, _ = mbm(options)
    execute(stages, options)
