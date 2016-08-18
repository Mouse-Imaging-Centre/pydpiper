#!/usr/bin/env python3

import os.path

import pandas as pd
from configargparse import Namespace, ArgParser
from typing import List

from pydpiper.minc.containers import XfmHandler

from pydpiper.core.files import FileAtom

from pydpiper.pipelines.MAGeT import maget, maget_parser, maget_parsers
from pydpiper.core.util import NamedTuple

from pydpiper.core.stages       import Result, Stages

#TODO fix up imports, naming, stuff in registration vs. pipelines, ...
from pydpiper.minc.files        import MincAtom, XfmAtom
from pydpiper.minc.registration import (lsq6_nuc_inorm, lsq12_nlin_build_model, registration_targets,
                                        LSQ6Conf, LSQ12Conf, get_resolution_from_file, concat_xfmhandlers,
                                        get_nonlinear_configuration_from_options,
                                        invert_xfmhandler, check_MINC_input_files, lsq12_nlin, MultilevelMincANTSConf,
                                        LinearTransType, get_linear_configuration_from_options, mincresample_new,
                                        Interpolation)
from pydpiper.minc.analysis     import determinants_at_fwhms, StatsConf
from pydpiper.core.arguments    import (lsq6_parser, lsq12_parser, nlin_parser, stats_parser, CompoundParser,
                                        AnnotatedParser, NLINConf, BaseParser, segmentation_parser)
from pydpiper.execution.application    import mk_application


MBMConf = NamedTuple('MBMConf', [('lsq6',  LSQ6Conf),
                                 ('lsq12', LSQ12Conf),
                                 ('nlin',  NLINConf),
                                 ('stats', StatsConf)])


def mbm_pipeline(options : MBMConf):
    s = Stages()
    imgs = [MincAtom(name, pipeline_sub_dir=os.path.join(options.application.output_directory,
                                                         options.application.pipeline_name + "_processed"))
            for name in options.application.files]

    check_MINC_input_files([img.path for img in imgs])

    prefix = options.application.pipeline_name

    result = s.defer(mbm(imgs=imgs, options=options,
                         prefix=prefix,
                         output_dir=options.application.output_directory))

    # create useful CSVs (note the files listed therein won't yet exist ...)
    for filename, dataframe in (("transforms.csv", result.xfms),
                                ("determinants.csv", result.determinants)):
        with open(filename, 'w') as f:
            def maybe_deref_path(x):
                # ugh ... just a convenience to allow using applymap in a 'generic' way ...
                if isinstance(x, FileAtom) or isinstance(x, XfmAtom):
                    return x.path
                elif isinstance(x, XfmHandler):
                    return x.xfm.path
                else:
                    return x
            f.write(dataframe.applymap(maybe_deref_path).to_csv(index=False))

    # TODO moved here from inside `mbm` for now ... does this make most sense?
    if options.mbm.mbm.run_maget:
        import copy
        maget_options = copy.deepcopy(options)  #Namespace(maget=options)
        #maget_options
        #maget_options.maget = maget_options.mbm
        #maget_options.execution = options.execution
        #maget_options.application = options.application
        maget_options.maget = options.mbm.maget
        del maget_options.mbm

        s.defer(maget([xfm.resampled for _ix, xfm in mbm.xfms.rigid_xfm.iterrows()],
                       options=maget_options,
                       prefix="%s_MAGeT" % prefix,
                       output_dir=os.path.join(options.application.output_directory, prefix + "_processed")))

    return Result(stages=s, output=result)


def mbm(imgs : List[MincAtom], options : MBMConf, prefix : str, output_dir : str = ""):

    # TODO could also allow pluggable pipeline parts e.g. LSQ6 could be substituted out for the modified LSQ6
    # for the kidney tips, etc...

    # TODO this is tedious and annoyingly similar to the registration chain ...
    lsq6_dir  = os.path.join(output_dir, prefix + "_lsq6")
    lsq12_dir = os.path.join(output_dir, prefix + "_lsq12")
    nlin_dir  = os.path.join(output_dir, prefix + "_nlin")

    s = Stages()

    if len(imgs) == 0:
        raise ValueError("Please, some files!")

    # FIXME: why do we have to call registration_targets *outside* of lsq6_nuc_inorm? is it just because of the extra
    # options required?
    targets = registration_targets(lsq6_conf=options.mbm.lsq6,
                                   app_conf=options.application,
                                   first_input_file=imgs[0].path)

    # TODO this is quite tedious and duplicates stuff in the registration chain ...
    resolution = (options.registration.resolution or
                  get_resolution_from_file(targets.registration_standard.path))
    options.registration = options.registration.replace(resolution=resolution)

    lsq6_result = s.defer(lsq6_nuc_inorm(imgs=imgs,
                                         resolution=resolution,
                                         registration_targets=targets,
                                         lsq6_dir=lsq6_dir,
                                         lsq6_options=options.mbm.lsq6))

    full_hierarchy = get_nonlinear_configuration_from_options(nlin_protocol=options.mbm.nlin.nlin_protocol,
                                                              reg_method=options.mbm.nlin.reg_method,
                                                              file_resolution=resolution)

    lsq12_nlin_result = s.defer(lsq12_nlin_build_model(imgs=[xfm.resampled for xfm in lsq6_result],
                                                       resolution=resolution,
                                                       lsq12_dir=lsq12_dir,
                                                       nlin_dir=nlin_dir,
                                                       nlin_prefix=prefix,
                                                       lsq12_conf=options.mbm.lsq12,
                                                       nlin_conf=full_hierarchy))

    inverted_xfms = [s.defer(invert_xfmhandler(xfm)) for xfm in lsq12_nlin_result.output]

    determinants = s.defer(determinants_at_fwhms(
                             xfms=inverted_xfms,
                             inv_xfms=lsq12_nlin_result.output,
                             blur_fwhms=options.mbm.stats.stats_kernels))

    overall_xfms = [s.defer(concat_xfmhandlers([rigid_xfm, nlin_xfm]))
                    for rigid_xfm, nlin_xfm in zip(lsq6_result, lsq12_nlin_result.output)]

    output_xfms = (pd.DataFrame({ "rigid_xfm"      : lsq6_result,
                                  "lsq12_nlin_xfm" : lsq12_nlin_result.output,
                                  "overall_xfm"    : overall_xfms }))
    # we could `merge` the determinants with this table, but preserving information would cause lots of duplication
    # of the transforms (or storing determinants in more columns, but iterating over dynamically known columns
    # seems a bit odd ...)

                            # TODO transpose these fields?})
                            #avg_img=lsq12_nlin_result.avg_img,  # inconsistent w/ WithAvgImgs[...]-style outputs
                           # "determinants"    : determinants })

    #output.avg_img = lsq12_nlin_result.avg_img
    #output.determinants = determinants   # TODO temporary - remove once incorporated properly into `output` proper
    # TODO add more of lsq12_nlin_result?

    # FIXME: this needs to go outside of the `mbm` function to avoid being run from within other pipelines (or
    # those other pipelines need to turn off this option)
    # TODO return some MAGeT stuff from MBM function ??
    # if options.mbm.mbm.run_maget:
    #     import copy
    #     maget_options = copy.deepcopy(options)  #Namespace(maget=options)
    #     #maget_options
    #     #maget_options.maget = maget_options.mbm
    #     #maget_options.execution = options.execution
    #     #maget_options.application = options.application
    #     maget_options.maget = options.mbm.maget
    #     del maget_options.mbm
    #
    #     s.defer(maget([xfm.resampled for xfm in lsq6_result],
    #                   options=maget_options,
    #                   prefix="%s_MAGeT" % prefix,
    #                   output_dir=os.path.join(output_dir, prefix + "_processed")))

    # FIXME: this needs to go outside of the `mbm` function to avoid being run from within other pipelines (or
    # those other pipelines need to turn off this option)
    if options.mbm.common_space.do_common_space_registration:
        if not options.mbm.common_space.common_space_model:
            raise ValueError("No common space template provided!")
        # TODO allow lsq6 registration as well ...
        common_space_model = MincAtom(options.mbm.common_space.common_space_model,
                                      pipeline_sub_dir=os.path.join(options.application.output_directory,
                                                         options.application.pipeline_name + "_processed"))
        # TODO allow different lsq12/nlin config params than the ones used in MBM ...
        # WEIRD ... see comment in lsq12_nlin code ...
        nlin_conf  = full_hierarchy.confs[-1] if isinstance(full_hierarchy, MultilevelMincANTSConf) else full_hierarchy
        # also weird that we need to call get_linear_configuration_from_options here ... ?
        lsq12_conf = get_linear_configuration_from_options(conf=options.mbm.lsq12,
                                                           transform_type=LinearTransType.lsq12,
                                                           file_resolution=resolution)
        xfm_to_common = s.defer(lsq12_nlin(source=lsq12_nlin_result.avg_img, target=common_space_model,
                                           lsq12_conf=lsq12_conf, nlin_conf=nlin_conf,
                                           resample_source=True))

        model_common = s.defer(mincresample_new(img=lsq12_nlin_result.avg_img,
                                                xfm=xfm_to_common.xfm, like=common_space_model,
                                                postfix="_common"))

        overall_xfms_common = [s.defer(concat_xfmhandlers([rigid_xfm, nlin_xfm, xfm_to_common]))
                               for rigid_xfm, nlin_xfm in zip(lsq6_result, lsq12_nlin_result.output)]

        xfms_common = [s.defer(concat_xfmhandlers([nlin_xfm, xfm_to_common]))
                       for nlin_xfm in lsq12_nlin_result.output]

        output_xfms = output_xfms.assign(xfm_common=xfms_common, overall_xfm_common=overall_xfms_common)

        log_nlin_det_common, log_full_det_common = [dets.map(lambda d:
                                                      s.defer(mincresample_new(
                                                        img=d,
                                                        xfm=xfm_to_common.xfm,
                                                        like=common_space_model,
                                                        postfix="_common",
                                                        extra_flags=("-keep_real_range",),
                                                        interpolation=Interpolation.nearest_neighbour)))
                                                    for dets in (determinants.log_nlin_det, determinants.log_full_det)]

        determinants = determinants.assign(log_nlin_det_common=log_nlin_det_common,
                                           log_full_det_common=log_full_det_common)

    output = Namespace(avg_img=lsq12_nlin_result.avg_img, xfms=output_xfms, determinants=determinants)

    if options.mbm.common_space.do_common_space_registration:
        output.model_common = model_common

    return Result(stages=s, output=output)


# TODO move to arguments file?
def _mk_common_space_parser(parser : ArgParser):
    group = parser.add_argument_group("Common space options", "Options for registration/resampling to common (db) space.")
    group.add_argument("--common-space-model", dest="common_space_model",
                       type=str, help="Run MAGeT segmentation on the images.")
    group.add_argument("--no-common-space-registration", dest="do_common_space_registration",
                       default=True, action="store_false", help="Skip registration to common (db) space.")
    return parser

common_space_parser = AnnotatedParser(parser=BaseParser(_mk_common_space_parser(ArgParser(add_help=False)),
                                                        "common_space"),
                                      namespace="common_space")

mbm_parser = CompoundParser(
               [lsq6_parser,
                lsq12_parser,
                nlin_parser,
                stats_parser,
                common_space_parser,
                AnnotatedParser(parser=maget_parsers, namespace="maget", prefix="maget"),
                # TODO note that the maget-specific flags (--mask, --masking-method, etc., also get the "maget-" prefix)
                # which could be changed by putting in the maget-specific parser separately from its lsq12, nlin parsers
                segmentation_parser])

# TODO cast to MBMConf?
mbm_application = mk_application(parsers=[AnnotatedParser(parser=mbm_parser, namespace='mbm')],
                                 pipeline=mbm_pipeline)

if __name__ == "__main__":
    mbm_application()
