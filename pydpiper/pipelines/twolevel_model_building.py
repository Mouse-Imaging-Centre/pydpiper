#!/usr/bin/env python3

import copy
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from configargparse import Namespace
import pandas as pd
from pydpiper.pipelines.registration_chain import get_closest_model_from_pride_of_models

from pydpiper.minc.analysis import determinants_at_fwhms
from pydpiper.minc.files import MincAtom
from pydpiper.minc.registration import (concat_xfmhandlers, invert_xfmhandler, mincresample_new, ensure_distinct_basenames,
                                        TargetType, get_pride_of_models_mapping, get_resolution_from_file,
                                        registration_targets)
from pydpiper.pipelines.MBM import mbm, MBMConf, mk_mbm_parser
from pydpiper.execution.application import execute
from pydpiper.core.util import NamedTuple, maybe_deref_path
from pydpiper.core.stages import Stages, Result
from pydpiper.core.arguments import (AnnotatedParser, CompoundParser, application_parser,
                                     execution_parser, registration_parser, parse, BaseParser, segmentation_parser)


TwoLevelConf = NamedTuple("TwoLevelConf", [("first_level_conf",  MBMConf),
                                           ("second_level_conf", MBMConf)])





def two_level_pipeline(options : TwoLevelConf):

    def relativize_path(fp):
        #this annoying function takes care of the csv_paths_relative_to_wd flag.
        return os.path.join(os.path.dirname(options.application.csv_file), Path(fp).as_posix()) \
            if not options.application.csv_paths_relative_to_wd \
            else Path(fp).as_posix()

    first_level_dir = options.application.pipeline_name + "_first_level"

    if options.application.files:
        warnings.warn("Got extra arguments: '%s'" % options.application.files)
    with open(options.application.csv_file, 'r') as f:
        try:
            files_df = (pd.read_csv(
                          filepath_or_buffer=f,
                          usecols=['group', 'file'],
                          index_col=False)
                        .assign(file=lambda df:
                                  df.apply(axis="columns",
                                           func=lambda r:
                                             MincAtom(relativize_path(r.file).strip(),
                                                      pipeline_sub_dir=os.path.join(first_level_dir,
                                                                                    "%s_processed" % r.group,
                                                                                    )))))
        except AttributeError:
            warnings.warn("Something went wrong ... does your .csv file have `group` and `file` columns?")
            raise

    # TODO is it actually sufficient that *within-study* filenames are distinct, as follows??
    for name, g in files_df.groupby("group"):   # TODO: collect the outputs
        ensure_distinct_basenames(g.file.map(lambda x: x.path))
    #check_MINC_input_files(files_df.file.map(lambda x: x.path))

    pipeline = two_level(grouped_files_df=files_df, options=options)

    # TODO write these into the appropriate subdirectory ...
    overall = (pipeline.output.overall_determinants
        .drop('inv_xfm', axis=1))
    overall.to_csv("overall_determinants.csv", index=False)

    resampled = (pipeline.output.resampled_determinants
        .drop(['inv_xfm', 'full_det', 'nlin_det'], axis=1))
    resampled.to_csv("resampled_determinants.csv", index=False)

    # rename/drop some columns, bind the dfs and write to "analysis.csv" as it should be.
    # deprecate the two csvs next release.
    # TODO it's a bit silly that we read the csv_file again here but the group names have been lost?
    analysis = pd.read_csv(options.application.csv_file).assign(native_file=lambda df:
                 df.file.apply(relativize_path))

    overall = (overall.drop(["full_det", "nlin_det"], axis=1)
        .rename(columns={"overall_xfm" : "xfm"}))
    resampled = resampled.rename(columns={
        "first_level_log_full_det" : "log_full_det",
        "first_level_log_nlin_det" : "log_nlin_det",
        "first_level_xfm" : "xfm",
        "first_level_log_full_det_resampled" : "resampled_log_full_det",
        "first_level_log_nlin_det_resampled" : "resampled_log_nlin_det"
    })
    (analysis
     .merge(pd.merge(left = resampled.assign(target = lambda df: df.xfm.apply(lambda r: r.target)),
                     right = overall.assign(target  = lambda df: df.xfm.apply(lambda r: r.target)),
                     on = ['target', 'fwhm']),
            on = "native_file")
     .applymap(maybe_deref_path)
     .to_csv("analysis.csv", index=False))

    # TODO it's unfortunate we don't return something like the nice analysis df constructed above (but before  mapping objects to paths)
    return pipeline


def two_level(grouped_files_df, options : TwoLevelConf):
    """
    grouped_files_df - must contain 'group':<any comparable, sortable type> and 'file':MincAtom columns
    """  # TODO weird naming since the grouped_files_df isn't a GroupBy object?  just files_df?
    s = Stages()

    if grouped_files_df.isnull().values.any():
        raise ValueError("NaN values in input dataframe; can't go")

    if options.mbm.lsq6.target_type == TargetType.bootstrap:
        # won't work since the second level part tries to get the resolution of *its* "first input file", which
        # hasn't been created.  We could instead pass in a resolution to the `mbm` function,
        # but instead disable for now:
        raise ValueError("Bootstrap model building currently doesn't work with this pipeline; "
                         "just specify an initial target instead")
    elif options.mbm.lsq6.target_type == TargetType.pride_of_models:
        pride_of_models_mapping = get_pride_of_models_mapping(pride_csv=options.mbm.lsq6.target_file,
                                                              output_dir=options.application.output_directory,
                                                              pipeline_name=options.application.pipeline_name)

    # FIXME this is the same as in the 'tamarack' except for names of arguments/enclosing variables
    def group_options(options, group):
        options = copy.deepcopy(options)

        if options.mbm.lsq6.target_type == TargetType.pride_of_models:

            targets = get_closest_model_from_pride_of_models(pride_of_models_dict=pride_of_models_mapping,
                                                             time_point=group)

            options.mbm.lsq6 = options.mbm.lsq6.replace(target_type=TargetType.initial_model,
                                                        target_file=targets.registration_standard.path)
        else:
            # this will ensure that all groups have the same resolution -- is it necessary?
            targets = s.defer(registration_targets(lsq6_conf=options.mbm.lsq6,
                                                   app_conf=options.application,
                                                   reg_conf=options.registration,
                                                   first_input_file=grouped_files_df.file.iloc[0]))

        resolution = (options.registration.resolution
                        or get_resolution_from_file(targets.registration_standard.path))
        # This must happen after calling registration_targets otherwise it will resample to options.registration.resolution
        options.registration = options.registration.replace(resolution=resolution)
        # no need to check common space settings here since they're turned off at the parser level
        # (a bit strange)
        return options

    first_level_results = (
        grouped_files_df
        .groupby('group', as_index=False)       # the usual annoying pattern to do a aggregate with access
        .aggregate({ 'file' : lambda files: list(files) })  # to the groupby object's keys ... TODO: fix
        .rename(columns={ 'file' : "files" })
        .assign(build_model=lambda df:
                              df.apply(axis=1,
                                       func=lambda row:
                                              s.defer(mbm(imgs=row.files,
                                                          options=group_options(options, row.group),
                                                          prefix="%s" % row.group,
                                                          output_dir=os.path.join(
                                                              options.application.output_directory,
                                                              options.application.pipeline_name + "_first_level",
                                                              "%s_processed" % row.group)))))
        )

    # TODO replace .assign(...apply(...)...) with just an apply, producing a series right away?

    # FIXME right now the same options set is being used for both levels -- use options.first/second_level
    second_level_options = copy.deepcopy(options)
    second_level_options.mbm.lsq6 = second_level_options.mbm.lsq6.replace(run_lsq6=False)
    second_level_options.mbm.segmentation.run_maget = False
    second_level_options.mbm.maget.maget.mask_only = False
    second_level_options.mbm.maget.maget.mask = False

    # FIXME this is probably a hack -- instead add a --second-level-init-model option to specify which timepoint should be used
    # as the initial model in the second level ???  (at this point it doesn't matter due to lack of lsq6 ...)
    if second_level_options.mbm.lsq6.target_type == TargetType.pride_of_models:
        second_level_options.mbm.lsq6 = second_level_options.mbm.lsq6.replace(
            target_type=TargetType.target,  # target doesn't really matter as no lsq6 here, just used for resolution...
            target_file=list(pride_of_models_mapping.values())[0].registration_standard.path)

    # NOTE: running lsq6_nuc_inorm here doesn't work in general (but possibly with rotational minctracc)
    # since the native-space initial model is used, but our images are
    # already in standard space (as we resampled there after the 1st-level lsq6).
    # On the other hand, we might want to run it here (although of course NOT nuc/inorm!) in the future,
    # for instance given a 'pride' of models (one for each group).

    second_level_results = s.defer(mbm(imgs=first_level_results.build_model.map(lambda m: m.avg_img),
                                       options=second_level_options,
                                       prefix=os.path.join(options.application.output_directory,
                                                           options.application.pipeline_name + "_second_level")))

    # FIXME sadly, `mbm` doesn't return a pd.Series of xfms, so we don't have convenient indexing ...
    overall_xfms = [s.defer(concat_xfmhandlers([xfm_1, xfm_2]))
                    for xfms_1, xfm_2 in zip([r.xfms.lsq12_nlin_xfm for r in first_level_results.build_model],
                                             second_level_results.xfms.overall_xfm)
                    for xfm_1 in xfms_1]
    resample  = np.vectorize(mincresample_new, excluded={"extra_flags"})
    defer     = np.vectorize(s.defer)

    # TODO using the avg_img here is a bit clunky -- maybe better to propagate group indices ...
    # only necessary since `mbm` doesn't return DataFrames but namespaces ...

    first_level_determinants = pd.concat(list(first_level_results.build_model.apply(
                                                lambda x: x.determinants.assign(first_level_avg=x.avg_img))),
                                         ignore_index=True)

    # first_level_xfms is only necessary because you otherwise have no access to the input file which is necessary
    # for merging with the input csv. lsq12_nlin_xfm can be used to merge, and rigid_xfm contains the input file.
    # If for some reason we want to output xfms in the future, just don't drop everything.
    first_level_xfms = pd.concat(list(first_level_results.build_model.apply(lambda x: x.xfms.assign(
        first_level_avg=x.avg_img))), ignore_index=True)[["lsq12_nlin_xfm", "rigid_xfm"]].assign(
        native_file=lambda df:df.rigid_xfm.apply(lambda x: x.source.path))
    if options.mbm.segmentation.run_maget:
        maget_df = pd.DataFrame([{"label_file" : x.labels.path, "native_file" : x.orig_path }  #, "_merge" : basename(x.orig_path)}
                                 for x in pd.concat([namespace.maget_result for namespace in first_level_results.build_model])])
        first_level_xfms = pd.merge(left=first_level_xfms,
                                    right=maget_df, on="native_file")

    first_level_determinants = (pd.merge(left=first_level_determinants, right=first_level_xfms,
                                         left_on="inv_xfm", right_on="lsq12_nlin_xfm")
                                .drop(["rigid_xfm", "lsq12_nlin_xfm"], axis=1))

    resampled_determinants = (pd.merge(
        left=first_level_determinants,
        right=pd.DataFrame({'group_xfm' : second_level_results.xfms.overall_xfm})
              .assign(source=lambda df: df.group_xfm.apply(lambda r: r.source)),
        left_on="first_level_avg",
        right_on="source")
        .assign(resampled_log_full_det=lambda df: defer(resample(img=df.log_full_det,
                                                                 xfm=df.group_xfm.apply(lambda x: x.xfm),
                                                                 like=second_level_results.avg_img)),
                resampled_log_nlin_det=lambda df: defer(resample(img=df.log_nlin_det,
                                                                 xfm=df.group_xfm.apply(lambda x: x.xfm),
                                                                 like=second_level_results.avg_img))))
    # TODO only resamples the log determinants, but still a bit ugly ... abstract somehow?
    # TODO shouldn't be called resampled_determinants since this is basically the whole (first_level) thing ...

    inverted_overall_xfms = [s.defer(invert_xfmhandler(xfm)) for xfm in overall_xfms]

    overall_determinants = (s.defer(determinants_at_fwhms(
                                     xfms=inverted_overall_xfms,
                                     inv_xfms=overall_xfms,
                                     blur_fwhms=options.mbm.stats.stats_kernels))
                            .assign(overall_log_full_det=lambda df: df.log_full_det,
                                    overall_log_nlin_det=lambda df: df.log_nlin_det)
                            .drop(['log_full_det', 'log_nlin_det'], axis=1))

    # TODO return some MAGeT stuff from two_level function ??
    # FIXME running MAGeT from within the `two_level` function has the same problem as running it from within `mbm`:
    # it will now run when this pipeline is called from within another one (e.g., n-level), which will be
    # redundant, create filename clashes, etc. -- this should be moved to `two_level_pipeline`.
    # TODO do we need a `pride of atlases` for MAGeT in this pipeline ??
    # TODO at the moment MAGeT is run within the MBM code, but it could be disabled there and run here
    #if options.mbm.segmentation.run_maget:
    #    maget_options = copy.deepcopy(options)
    #    maget_options.maget = options.mbm.maget
    #    fixup_maget_options(maget_options=maget_options.maget,
    #                        lsq12_options=maget_options.mbm.lsq12,
    #                        nlin_options=maget_options.mbm.nlin)
    #    maget_options.maget.maget.mask = maget_options.maget.maget.mask_only = False   # already done above
    #    del maget_options.mbm

        # again using a weird combination of vectorized and loop constructs ...
    #    s.defer(maget([xfm.resampled for _ix, m in first_level_results.iterrows()
    #                   for xfm in m.build_model.xfms.rigid_xfm],
    #                  options=maget_options,
    #                  prefix="%s_MAGeT" % options.application.pipeline_name,
    #                  output_dir=os.path.join(options.application.output_directory,
    #                                          options.application.pipeline_name + "_processed")))

    # TODO resampling to database model ...

    # TODO there should be one table containing all determinants (first level, overall, resampled first level) for each file
    # and another containing some groupwise information (averages and transforms to the common average)
    return Result(stages=s, output=Namespace(first_level_results=first_level_results,
                                             resampled_determinants=resampled_determinants,
                                             overall_determinants=overall_determinants))

# FIXME: better to replace --files by this for all/most pipelines;
# then we can enforce presence of metadata in the CSV file ... (pace MINC2)
#def _mk_twolevel_parser(p):
#    p.add_argument("--csv-file", dest="csv_file", type=str, required=True,
#                   help="CSV file containing at least 'group' and 'file' columns")
#    return p


#_twolevel_parser = BaseParser(_mk_twolevel_parser(ArgParser(add_help=False)), group_name='twolevel')
#twolevel_parser = AnnotatedParser(parser=_twolevel_parser, namespace="twolevel")


def main(args):
    p = CompoundParser(
          [execution_parser,
           application_parser,
           registration_parser,
           #twolevel_parser,
           AnnotatedParser(parser=mk_mbm_parser(with_common_space=False), namespace="mbm"),   # TODO use this before 1st-/2nd-level args
           # TODO to combine the information from all three MBM parsers,
           # could use `ConfigArgParse`r `_source_to_settings` (others?) to check whether an option was defaulted
           # or user-specified, allowing the first/second-level options to override the general mbm settings
           #AnnotatedParser(parser=mbm_parser, namespace="first_level", prefix="first-level"),
           #AnnotatedParser(parser=mbm_parser, namespace="second_level", prefix="second-level"),
           #stats_parser
           #segmentation_parser
           ])  # TODO add more stats parsers?

    options = parse(p, args[1:])

    execute(two_level_pipeline(options).stages, options)

if __name__ == "__main__":
    main(sys.argv)
