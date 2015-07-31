#!/usr/bin/env python2.7

from collections import namedtuple
import os.path

from pydpiper.core.stages     import Stages
from pydpiper.core.conversion       import pipeline
#from ..core.containers import Result
#TODO fix up imports, naming, stuff in registration vs. pipelines, ...
from pydpiper.minc.files            import MincAtom
import pydpiper.minc.registration as m
from pydpiper.minc.analysis     import *
from pydpiper.pipelines.LSQ6 import LSQ6_default_conf

# TODO abstract out some common configuration functionality and think about argparsing ...

MBMConf = namedtuple('MBMConf',
  ['pipeline_name',
   'execution_conf',
   'lsq6_conf',
   'lsq12_conf',
   'nlin_conf',
   'stats_conf'
   ])

MBM_default_conf = MBMConf(
  pipeline_name='MBM',
  execution_conf=None,
  lsq6_conf=LSQ6_default_conf,
  lsq12_conf=m.LSQ12_default_conf,
  nlin_conf=m.mincANTS_default_conf,
  stats_conf={'blur_fwhms' : [None, 0.056]})

def MBM(imgs, conf):

    # TODO some of this initial blather could be factored into a pipeline-creating function ...
    # TODO could also allow pluggable pipeline parts e.g. LSQ6 could be substituted out for the modified LSQ6
    # for the kidney tips, etc...
    pipeline_dir = os.path.join('.', conf.pipeline_name)
    #curr_dir = os.path.join(os.getcwd(), conf.pipeline_name, '/processed')
    imgs = map(lambda name: MincAtom(name, curr_dir=pipeline_dir_dir), images)

    lsq6_stages = Stages()
    lsq6_result = lsq6_stages.defer(lsq6(imgs, LSQ6_conf, curr_dir=pipeline_dir))

    #TODO factor out some of this standard model-building stuff into a procedure, similar to FullIterativeLSQ12NLIN
    # in the old code, but maybe also with optional LSQ6 ?  Also used in, e.g., two-level code

    lsq12_stages = Stages()
    lsq12_result = lsq12_stages.defer(m.lsq12(imgs=[x.resampled for x in lsq6_result.xfms], conf=m.MinctraccConf(), curr_dir=pipeline_dir))

    for s in lsq12_stages:
        print(s.render())

    nlin_stages = Stages()
    nlin_result = nlin_stages.defer(m.mincANTS_NLIN(imgs=[x.resampled for x in lsq12_result.xfms],
                                                    avg=lsq12_result.avg_img,
                                                    conf=m.mincANTS_default_conf))

    for s in nlin_stages:
        print(s.render())

    stats_stages = Stages()
    
    overall_xfms = map(lambda f, g: stats_stages.defer(concat([f, g])), lsq12_result.xfms, nlin_result.xfms)

    for xfm in overall_xfms:
        stats_stages.defer(a.determinants_at_fwhms(xfm, blur_fwhms=conf.stats_conf['blur_fwhms']))

    #for s in stats_stages:
    #    print(s.render())

    all_stages = lsq6_stages lsq12_stages | nlin_stages | stats_stages # TODO just use one set of stages ?

    return all_stages
    #TODO accumulate all stages and create a pipeline, directories, etc. -- actually, this is similar for all pipelines, so don't
    #p = pipeline(all_stages)
    #return p

if __name__ == "__main__":
    MBM(['images/img_%d.mnc' % i for i in range(1,4)], conf=MBM_test_conf)
