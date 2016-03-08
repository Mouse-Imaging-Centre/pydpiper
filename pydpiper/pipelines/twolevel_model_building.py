#!/usr/bin/env python3

import csv
import os
import sys

from configargparse import Namespace
from pydpiper.minc.files import MincAtom

from pydpiper.minc.registration import concat_xfmhandlers, mincresample, RegistrationConf
from pydpiper.pipelines.MBM import mbm, mbm_parser, MBMConf
from pydpiper.execution.application import execute
from pydpiper.core.util import NamedTuple, collect
from pydpiper.core.stages import Stages, Result
from pydpiper.core.arguments import (AnnotatedParser, CompoundParser, application_parser, stats_parser,
                                     execution_parser, registration_parser, parse)


TwoLevelConf = NamedTuple("TwoLevelConf", [("first_level_conf", MBMConf),
                                           ("second_level_conf", MBMConf)])


def two_level(options : TwoLevelConf):
    s = Stages()
    if len(options.application.files) != 1:
        raise ValueError("Must supply exactly one input file; got: %s..." % options.application.files)
    with open(options.application.files[0], 'r') as f:
        groups = collect(parse_csv(f))
    first_level_results = (
        [s.defer(mbm(imgs=[MincAtom(f, pipeline_sub_dir=os.path.join(options.application.output_directory,
                                                                     options.application.pipeline_name + "_first_level",
                                                                     gid + "_processed"))
                           for f in files],
                     options=options,
                     output_dir=os.path.join(options.application.output_directory,
                                             options.application.pipeline_name + "_first_level"),
                     prefix=gid))
                     # FIXME set options.output_dir to be current processed dir + + 1st_level + g_id
         for gid, files in groups.items()])
    #print(first_level_results)
    second_level_results = s.defer(mbm(imgs=[r.avg_img for r in first_level_results],
                                       options=options,
                                       prefix=os.path.join(options.application.output_directory,
                                                           options.application.pipeline_name + "_second_level")))
                         # TODO: options.first/second_level ???


    #concatenated_xfms = [s.defer(concat_xfmhandlers([xfm_1, xfm_2]))
    #                     for xfms_1, xfm_2 in zip([r.overall_xfms for r in first_level_results],
    #                                              second_level_results.overall_xfms)
    #                     for xfm_1 in xfms_1]

    # FIXME totally broken ...
    #resampled_determinants = [s.defer(mincresample(img=det, xfm=xfm, like=second_level_results.average))
    #                          for dets, xfm in zip([result.determinants for result in first_level_results],
    #                                               second_level_results.overall_xfms)
    #                          for det in dets]

    return Result(stages=s, output=Namespace())
                                             #concatenated_xfms=concatenated_xfms  ))  #,
                                             #resampled_determinants=resampled_determinants))

def parse_csv(f):
    for row in csv.DictReader(f):
        yield row["group"], row["file"]   # somewhat immoral -- should collect operate on the row object directly?

def main(args):
    p = CompoundParser(
          [execution_parser,
           application_parser,
           registration_parser,
           AnnotatedParser(parser=mbm_parser, namespace="mbm"),   # TODO use this before 1st-/2nd-level args
           #AnnotatedParser(parser=mbm_parser, namespace="first_level", prefix="first-level"),
           #AnnotatedParser(parser=mbm_parser, namespace="second_level", prefix="second-level"),
           #stats_parser
           ])  # TODO add more stats parsers?

    options = parse(p, args[1:])

    execute(two_level(options).stages, options)

if __name__ == "__main__":
    main(sys.argv)