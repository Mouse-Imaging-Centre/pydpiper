#!/usr/bin/env python3
import sys

from pydpiper.execution.application import execute

from pydpiper.core.util import NamedTuple
from pydpiper.core.stages import Stages
from pydpiper.core.arguments import (AnnotatedParser, CompoundParser, application_parser, stats_parser,
                                     execution_parser, registration_parser)


TwoLevelConf = NamedTuple("TwoLevelConf", [("first_level_conf", MBMConf),
                                           ("second_level_conf", MBMConf)])


def addTwoLevelArgumentGroup(parser):
    pass


def two_level(conf : TwoLevelConf):
    s = Stages()
    first_level = s.defer(mbm(conf.first_level))

def main(args):
    p = CompoundParser(
          [AnnotatedParser(parser=execution_parser, namespace='execution'),
           AnnotatedParser(parser=application_parser, namespace='application'),
           AnnotatedParser(parser=registration_parser, namespace='registration', cast=RegistrationConf),
           #AnnotatedParser(parser=mbm_parser, namespace="first-level"),  # TODO actually write this
           # TODO give an example of creating a general mbm parser to be overriden by first and second levels
           AnnotatedParser(parser=mbm_parser, namespace="second-level"),
           AnnotatedParser(parser=stats_parser, namespace='stats')])

    options = parse(p, args[1:])

    execute(two_level(options).stages, options.execution)

if __name__ == "__main__":
    main(sys.argv)