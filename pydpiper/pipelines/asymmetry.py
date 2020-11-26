#!/usr/bin/env python3

# asymmetry pipeline - given input files, run a two-level model with first level groups constituted of each
# input image and its flipped copy

import os
import sys

import pandas as pd
from pydpiper.pipelines.MAGeT import get_imgs

from pydpiper.pipelines.MBM import mk_mbm_parser
from pydpiper.core.stages import Result, Stages
from pydpiper.pipelines.twolevel_model_building import two_level
from pydpiper.core.arguments import (execution_parser, registration_parser, application_parser, parse, CompoundParser,
                                     AnnotatedParser)
from pydpiper.execution.application import execute
from pydpiper.minc.registration import volflip, ensure_distinct_basenames
from pydpiper.minc.files import MincAtom


def asymmetry_pipeline(options):

    output_dir    = options.application.output_directory
    pipeline_name = options.application.pipeline_name
    processed_dir = os.path.join(output_dir, pipeline_name + "_processed")

    s = Stages()

    #imgs_ = [MincAtom(f, pipeline_sub_dir=processed_dir) for f in options.application.files]

    imgs_ = get_imgs(options.application)

    ensure_distinct_basenames([img.path for img in imgs_])

    imgs  = pd.Series(imgs_, index=[img.filename_wo_ext for img in imgs_])
    flipped_imgs = imgs.apply(lambda img: s.defer(volflip(img)))  # TODO add flags to control flip axis ...

    # TODO ugly - MincAtom API should allow this somehow without mutation (also, how to pass into `volflip`, etc.?)
    for f_i in flipped_imgs:
        f_i.output_sub_dir += "_flipped"

    ensure_distinct_basenames(imgs.apply(lambda img: img.path))

    grouped_files_df = pd.DataFrame({'file' : pd.concat([imgs, flipped_imgs])}).assign(group=lambda df: df.index)

    two_level_result = s.defer(two_level(grouped_files_df, options=options))

    return Result(stages=s, output=two_level_result)


def main(args):
    p = CompoundParser(
          [execution_parser,
           application_parser,
           registration_parser,
           AnnotatedParser(parser=mk_mbm_parser(), namespace="mbm")])

    options = parse(p, args[1:])
    stages = asymmetry_pipeline(options).stages
    execute(stages, options)


if __name__ == '__main__':
    main(sys.argv)
