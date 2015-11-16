#!/usr/bin/env python2.7

import os.path

from atom.api import Atom

from pydpiper.core.stages     import Result, Stages
#from ..core.containers import Result
#TODO fix up imports, naming, stuff in registration vs. pipelines, ...
from pydpiper.minc.files            import MincAtom
from pydpiper.pipelines.LSQ6    import LSQ6
from pydpiper.minc.registration import LSQ12_NLIN_build_model
from pydpiper.minc.analysis     import determinants_at_fwhms
import pydpiper.core.arguments  as args
from pydpiper.core.execution    import execute

# TODO abstract out some common configuration functionality and think about argparsing ...

class MBMConf(Atom):
  pass
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

def MBM(options):

    # TODO could also allow pluggable pipeline parts e.g. LSQ6 could be substituted out for the modified LSQ6
    # for the kidney tips, etc...
    pipeline_dir = os.path.join('.', options.pipeline_name)
    #curr_dir = os.path.join(os.getcwd(), conf.pipeline_name, '/processed')
    imgs = [MincAtom(name, output_dir=pipeline_dir) for name in options.files]

    s = Stages()
    
    lsq6_result = s.defer(LSQ6(imgs=imgs, conf=options.lsq6,
                               output_dir=pipeline_dir))

    lsq12_nlin_result = s.defer(LSQ12_NLIN_build_model(
        imgs=[xfm.resampled for xfm in lsq6_result],
        lsq12_conf = options.lsq12_conf, nlin_conf=options.nlin_conf))

    determinants = [s.defer(determinants_at_fwhms(xfm, blur_fwhms=options.stats.stats_kernels))
                    for xfm in lsq12_nlin_result.xfms]

    return Result(stages=s, output=determinants) # TODO: add more outputs

def MBM_parser():
    return args.make_parser([(x, "") for x in
        [args.addExecutorArgumentGroup,
         args.addApplicationArgumentGroup,
         args.addGeneralRegistrationArgumentGroup,
         args.addLSQ6ArgumentGroup,
         args.addLSQ12ArgumentGroup,
         args.addNLINArgumentGroup,
         args.addStatsArgumentGroup]])
    
if __name__ == "__main__":
    options = MBM_parser().parse_args()
    stages, _ = MBM(options)
    execute(stages, options)
