#!/usr/bin/env python2.7

from collections import namedtuple
import os.path

from pydpiper.core.stages     import Stages
#from ..core.containers import Result
#TODO fix up imports, naming, stuff in registration vs. pipelines, ...
from pydpiper.minc.files            import MincAtom
from pydpiper.pipelines.LSQ6    import LSQ6
from pydpiper.minc.registration import LSQ12_NLIN_build_model
from pydpiper.minc.analysis     import determinants_at_fwhms
import pydpiper.core.arguments  as args
from pydpiper.core.containers   import Result
from pydpiper.core.execution    import execute

# TODO abstract out some common configuration functionality and think about argparsing ...

MBMConf = namedtuple('MBMConf',
  ['pipeline_name',
   'execution_conf',
   'lsq6_conf',
   'lsq12_conf',
   'nlin_conf',
   'stats_conf'
   ])

# MBM_default_conf = MBMConf(
#   pipeline_name='MBM',
#   execution_conf=None,
#   lsq6_conf=LSQ6_default_conf,
#   lsq12_conf=LSQ12_default_conf,
#   nlin_conf=mincANTS_default_conf,
#   stats_conf={'blur_fwhms' : [None, 0.056]})

def MBM(conf):

    # TODO could also allow pluggable pipeline parts e.g. LSQ6 could be substituted out for the modified LSQ6
    # for the kidney tips, etc...
    pipeline_dir = os.path.join('.', conf.pipeline_name)
    #curr_dir = os.path.join(os.getcwd(), conf.pipeline_name, '/processed')
    imgs = map(lambda name: MincAtom(name, output_dir=pipeline_dir),
               options.files)

    s = Stages()
    
    lsq6_result = s.defer(LSQ6(imgs=imgs, conf=options.lsq6,
                               output_dir=pipeline_dir))

    lsq12_nlin_result = s.defer(LSQ12_NLIN_build_model(
        imgs=[xfm.resampled for xfm in lsq6_result],
        lsq12_conf = options.lsq12_conf, nlin_conf=options.nlin_conf))

    determinants = \
      [s.defer(determinants_at_fwhms(xfm, blur_fwhms=conf.stats.stats_kernels))
       for xfm in lsq12_nlin_result.xfms]

    #for stage in s:
    #    print(stage.render())

    return Result(stages=s, output=determinants) # TODO: add more outputs

if __name__ == "__main__":
    parser = args.make_parser(
        [(args.addExecutorArgumentGroup, None),
         (args.addApplicationArgumentGroup, None),
         (args.addGeneralRegistrationArgumentGroup, None),
         (args.addMBMArgumentGroup, None),
         (args.addStatsArgumentGroup, None)])
    options = parser.parse_args()
    stages, _ = MBM(options)
    execute(stages, options)
