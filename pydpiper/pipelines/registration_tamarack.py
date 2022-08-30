#!/usr/bin/env python3
import copy
import os
import sys

import pandas as pd
from configargparse import ArgParser, Namespace
from pydpiper.minc.analysis import determinants_at_fwhms

from pydpiper.pipelines.registration_chain import get_closest_model_from_pride_of_models

from pydpiper.core.util import NamedTuple, maybe_deref_path
from pydpiper.pipelines.MBM import mk_mbm_parser, mbm, MBMConf
from pydpiper.pipelines.MAGeT import (get_registration_module,
                                      get_rigid_registration_module,
                                      get_affine_registration_module)
from pydpiper.core.stages import Result, Stages
from pydpiper.core.arguments import (execution_parser, registration_parser, application_parser, parse, CompoundParser,
                                     AnnotatedParser, BaseParser)
from pydpiper.execution.application import execute
from pydpiper.minc.registration import (
    ensure_distinct_basenames, lsq12_nlin, get_pride_of_models_mapping, TargetType,
    concat_xfmhandlers, get_linear_configuration_from_options, LinearTransType,
    registration_targets, get_nonlinear_component)
from pydpiper.minc.files import MincAtom


TamarackConf = NamedTuple("TamarackConf", [("first_level_conf", MBMConf),
                                           ("second_level_conf", MBMConf)])

def tamarack_pipeline(options):

    output_dir    = options.application.output_directory
    pipeline_name = options.application.pipeline_name
    #processed_dir = os.path.join(output_dir, pipeline_name + "_processed")
    first_level_dir = os.path.join(output_dir, pipeline_name + "_first_level")

    s = Stages()

    with open(options.application.csv_file, 'r') as f:
        files_df = (pd.read_csv(filepath_or_buffer=f,
                                usecols=['group', 'filename'])
                    .assign(file=lambda df:
                                   df.apply(axis="columns",
                                            func=lambda r:
                                                   MincAtom(r.filename.strip(),
                                                            pipeline_sub_dir=os.path.join(first_level_dir,
                                                                                          "%s_processed" % r.group)))))

    ensure_distinct_basenames(files_df.file.apply(lambda img: img.path))

    #grouped_files_df = pd.DataFrame({'file' : pd.concat([imgs])}).assign(group=lambda df: df.index)

    tamarack_result = s.defer(tamarack(files_df, options=options))

    # first_level_results is currently a nested data structure, not a data frame
    #tamarack_result.first_level_results.applymap(maybe_deref_path).to_csv("first_level_results.csv", index=False)

    tamarack_result.resampled_determinants.applymap(maybe_deref_path).to_csv("resampled_determinants.csv", index=False)
    tamarack_result.overall_determinants.applymap(maybe_deref_path).to_csv("overall_determinants.csv", index=False)

    return Result(stages=s, output=tamarack_result)


def tamarack(imgs : pd.DataFrame, options):
    # columns of the input df: `img` : MincAtom, `timept` : number, ...
    # columns of the pride of models : 'timept' : number, 'model' : MincAtom
    s = Stages()

    # TODO some assertions that the pride_of_models, if provided, is correct, and that this is intended target type

    lsq6_module = get_rigid_registration_module(options.registration.image_algorithms, options.mbm.lsq6.lsq6_method)
    lsq12_module = get_affine_registration_module(options.registration.image_algorithms, options.mbm.lsq12.reg_method)
    reg_module = get_registration_module(options.registration.image_algorithms, options.mbm.nlin.reg_method)
    algorithms = reg_module.Algorithms

    def group_options(options, timepoint):
        options = copy.deepcopy(options)

        if options.mbm.lsq6.target_type == TargetType.pride_of_models:
            options = copy.deepcopy(options)
            targets = get_closest_model_from_pride_of_models(pride_of_models_dict=get_pride_of_models_mapping(
                                                                 pride_csv=options.mbm.lsq6.target_file,
                                                                 output_dir=options.application.output_directory,
                                                                 pipeline_name=options.application.pipeline_name),
                                                             time_point=timepoint)

            options.mbm.lsq6 = options.mbm.lsq6.replace(target_type=TargetType.initial_model,
                                                        target_file=targets.registration_standard.path)

        #    resolution = (options.registration.resolution
        #                  or get_resolution_from_file(targets.registration_standard.path))
        #    options.registration = options.registration.replace(resolution=resolution)

                                                        # FIXME use of registration_standard here is quite wrong ...
                                                        # part of the trouble is that mbm calls registration_targets itself,
                                                        # so we can't send this RegistrationTargets to `mbm` directly ...
                                                        # one option: add yet another optional arg to `mbm` ...
        else:
            #targets = s.defer(registration_targets(lsq6_conf=options.mbm.lsq6,
            #                               app_conf=options.application, reg_conf=options.registration,
            #                               image_algorithms=reg_module,
            #                               first_input_file=imgs.filename.iloc[0]))
            targets = s.defer(registration_targets(lsq6_conf=options.mbm.lsq6,
                                           app_conf=options.application,
                                           reg_conf=options.registration,
                                           image_algorithms=algorithms,
                                           first_input_file=imgs.filename.iloc[0]))

        resolution = (options.registration.resolution or
                      algorithms.get_resolution_from_file(
                          targets.registration_standard.path))

        # This must happen after calling registration_targets otherwise it will resample to options.registration.resolution
        options.registration = options.registration.replace(resolution=resolution)

        return options

    # build all first-level models:
    first_level_results = (
        imgs  # TODO 'group' => 'timept' ?
        .groupby('group', as_index=False)       # the usual annoying pattern to do an aggregate with access
        .aggregate({ 'file' : lambda files: list(files) })  # to the groupby object's keys ... TODO: fix
        .rename(columns={ 'file' : "files" })
        .assign(options=lambda df: df.apply(axis=1, func=lambda row: group_options(options, row.group)))
        .assign(build_model=lambda df:
                              df.apply(axis=1,
                                       func=lambda row: s.defer(
                                           mbm(imgs=row.files,
                                               registration_algorithms=reg_module,
                                               lsq6_module=lsq6_module,
                                               lsq12_module=lsq12_module,
                                               options=row.options,
                                               prefix="%s" % row.group,
                                               output_dir=os.path.join(
                                               options.application.output_directory,
                                               options.application.pipeline_name + "_first_level",
                                               "%s_processed" % row.group)))))
        .sort_values(by='group')

        )

    if all(first_level_results.options.map(lambda opts: opts.registration.resolution)
             == first_level_results.options.iloc[0].registration.resolution):
        options.registration = options.registration.replace(
            resolution=first_level_results.options.iloc[0].registration.resolution)
    else:
        raise ValueError("some first-level models are run at different resolutions, possibly not what you want ...")

    # construction of the overall inter-average transforms will be done iteratively (for efficiency/aesthetics),
    # which doesn't really fit the DataFrame mold ...

    # first register consecutive averages together:
    average_registrations = (
        first_level_results[:-1]
            .assign(next_model=list(first_level_results[1:].build_model))
            # TODO: we should be able to do lsq6 registration here as well!
            .assign(xfm=lambda df: df.apply(axis=1, func=lambda row: s.defer(
                                                      lsq12_nlin(moving=row.build_model.avg_img,
                                                                 fixed=row.next_model.avg_img,
                                                                 lsq12_conf=lsq12_module.parse_multilevel_protocol_file(options.mbm.lsq12.protocol, resolution=options.registration.resolution),
                                                                 #get_linear_configuration_from_options(
                                                                 #    options.mbm.lsq12,
                                                                 #    transform_type=LinearTransType.lsq12,
                                                                 #    file_resolution=options.registration.resolution),
                                                                 lsq12_module=lsq12_module,
                                                                 nlin_module=reg_module,
                                                                 run_lsq12=options.mbm.lsq12.run_lsq12,
                                                                 resolution=options.registration.resolution,
                                                                 nlin_options=options.mbm.nlin.nlin_protocol,
                                                                 resample_moving=True)))))

    # now compose the above transforms to produce transforms from each average to the common average:
    common_time_pt = options.tamarack.common_time_pt
    common_model   = first_level_results[first_level_results.group == common_time_pt].iloc[0].build_model.avg_img
    #common = average_registrations[average_registrations.group == common_time_pt].iloc[0]
    before = average_registrations[average_registrations.group <  common_time_pt]  # asymmetry in before/after since
    after  = average_registrations[average_registrations.group >= common_time_pt]  # we used `next_`, not `previous_`

    # compose 1st and 2nd level transforms and resample into the common average space:
    def suffixes(xs):
        if len(xs) == 0:
            return [[]]
        else:
            ys = suffixes(xs[1:])
            return [[xs[0]] + ys[0]] + ys


    def prefixes(xs):
        if len(xs) == 0:
            return [[]]
        else:
            ys = prefixes(xs[1:])
            return ys + [ys[-1] + [xs[0]]]

    # TODO we can improve the logic here to remove the use of 'invert' by choosing which direction to go in prior based on whether we're before/after
    # the common time point
    xfms_to_common = (
        first_level_results
        .assign(uncomposed_xfms=suffixes(list(before.xfm))[:-1] + [None] + prefixes(list(after.xfm))[1:])
        .assign(xfm_to_common=lambda df: df.apply(axis=1, func=lambda row:
                                ((lambda x: s.defer(algorithms.invert_xfmhandler(x)) if row.group >= common_time_pt else x)
                                   (s.defer(concat_xfmhandlers(row.uncomposed_xfms,
                                                               algorithms=algorithms,
                                                               name=("%s_to_common"
                                                                     if row.group < common_time_pt
                                                                     else "%s_from_common") % row.group))))
                                  if row.uncomposed_xfms is not None else None))
        .drop('uncomposed_xfms', axis=1))  # TODO None => identity??

    # TODO indexing here is not good ...
    first_level_determinants = pd.concat(list(first_level_results.build_model.apply(
                                                lambda x: x.determinants.assign(first_level_avg=x.avg_img))),
                                         ignore_index=True)

    resampled_determinants = (
        pd.merge(left=first_level_determinants,
                 right=xfms_to_common.assign(moving=lambda df: df.xfm_to_common.apply(
                                                              lambda x:
                                                                x.moving if x is not None else None)),
                 left_on="first_level_avg", right_on='moving')
        .assign(resampled_log_full_det=lambda df: df.apply(axis=1, func=lambda row:
                                         s.defer(algorithms.resample(img=row.log_full_det,
                                                                     xfm=row.xfm_to_common.xfm,
                                                                     like=common_model))
                                                 if row.xfm_to_common is not None else row.img),
                #resampled_log_nlin_det=lambda df: df.apply(axis=1, func=lambda row:
                #                         s.defer(algorithms.resample(img=row.log_nlin_det,
                #                                                     xfm=row.xfm_to_common.xfm,
                #                                                     like=common_model))
                #                                 if row.xfm_to_common is not None else row.img))
                )
    )

    inverted_overall_xfms = pd.Series({ xfm : (s.defer(concat_xfmhandlers([xfm, row.xfm_to_common], algorithms=algorithms))
                                                 if row.xfm_to_common is not None else xfm)
                                        for _ix, row in xfms_to_common.iterrows()
                                        for xfm in row.build_model.xfms.lsq12_nlin_xfm })

    overall_xfms = inverted_overall_xfms.apply(lambda x: s.defer(algorithms.invert_xfmhandler(x)))

    #overall_determinants = s.defer(
    #                         determinants_at_fwhms(xfms=overall_xfms, algorithms=algorithms,
    #                                               blur_fwhms=options.mbm.stats.stats_kernels))
    if options.mbm.stats.calc_stats:
        #determinants = s.defer(determinants_at_fwhms(
        #                           xfms=lsq12_nlin_result.output,
        #                           blur_fwhms=options.mbm.stats.stats_kernels,
        #                           algorithms=algorithms))
        # FIXME blurring, nlin det, etc.
        overall_determinants = (pd.DataFrame({ 'overall_xfm' : x } for x in overall_xfms) 
                                .assign(log_full_det = lambda df:
                                          df.overall_xfm.apply(lambda x: s.defer(algorithms.log_determinant(x)))))


    # TODO turn off bootstrap as with two-level code?

    # TODO combine into one data frame
    return Result(stages=s, output=Namespace(first_level_results=first_level_results,
                                             overall_determinants=overall_determinants,
                                             resampled_determinants=resampled_determinants.drop(
                                                 ['options', 'build_model', 'files'],
                                                 axis=1)))

def _mk_tamarack_parser(p):
    #p.add_argument("--csv-file", dest="csv_file", type=str, required=True,
    #               help="CSV file containing at least 'group' and 'file' columns")
    p.add_argument("--common-time-point", dest="common_time_pt", type=float, required=True,
                   help="Time point to resample everything to (must be one of the group IDs).")
    return p


_tamarack_parser = BaseParser(_mk_tamarack_parser(ArgParser(add_help=False)), group_name='tamarack')
tamarack_parser = AnnotatedParser(parser=_tamarack_parser, namespace="tamarack")

def main(args):
    # TODO rewrite using `mk_application`
    p = CompoundParser(
          [execution_parser,
           application_parser,
           registration_parser,
           tamarack_parser,
           AnnotatedParser(parser=mk_mbm_parser(with_common_space=False,
                                                with_maget=True),
                           namespace="mbm")])

    options = parse(p, args[1:])
    stages = tamarack_pipeline(options).stages
    execute(stages, options)


def application(): return main(sys.argv)
