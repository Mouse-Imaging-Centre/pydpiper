#!/usr/bin/env python3

import os.path
import warnings
import pandas as pd
from configargparse import Namespace, ArgParser
from typing import List, TypeVar
from pydpiper.minc.containers import XfmHandler

from pydpiper.core.arguments    import (lsq6_parser, lsq12_parser, nlin_parser, stats_parser, CompoundParser,
                                        AnnotatedParser, NLINConf, BaseParser, segmentation_parser)
from pydpiper.core.files import FileAtom
from pydpiper.core.util  import NamedTuple, maybe_deref_path
from pydpiper.core.stages       import Result, Stages
from pydpiper.execution.application    import mk_application

from pydpiper.minc.registration_strategies import get_model_building_procedure
from pydpiper.minc.thickness import cortical_thickness
#TODO fix up imports, naming, stuff in registration vs. pipelines, ...
from pydpiper.minc.files        import MincAtom, XfmAtom
from pydpiper.minc.registration import (lsq6_nuc_inorm, lsq12_nlin_build_model, registration_targets,
                                        LSQ6Conf, LSQ12Conf, get_resolution_from_file, concat_xfmhandlers,
                                        invert_xfmhandler, check_MINC_input_files, lsq12_nlin,
                                        LinearTransType, get_linear_configuration_from_options, mincresample_new,
                                        Interpolation, param2xfm, get_nonlinear_component)
from pydpiper.minc.analysis     import determinants_at_fwhms, StatsConf
from pydpiper.minc.thickness    import thickness_parser

from pydpiper.pipelines.MAGeT import maget, maget_parsers, fixup_maget_options, maget_mask, get_imgs


MBMConf = NamedTuple('MBMConf', [('lsq6',  LSQ6Conf),
                                 ('lsq12', LSQ12Conf),
                                 ('nlin',  NLINConf),
                                 ('stats', StatsConf)])


def mbm_pipeline(options : MBMConf):
    s = Stages()

    imgs = get_imgs(options.application)

    check_MINC_input_files([img.path for img in imgs])

    output_dir = options.application.output_directory

    mbm_result = s.defer(mbm(imgs=imgs, options=options,
                             prefix=options.application.pipeline_name,
                             output_dir=output_dir))

    if options.mbm.common_space.do_common_space_registration:
        s.defer(common_space(mbm_result, options))

    # create useful CSVs (note the files listed therein won't yet exist ...):
    transforms = (mbm_result.xfms.assign(
                            native_file=lambda df: df.rigid_xfm.apply(lambda x: x.source),
                            lsq6_file=lambda df: df.lsq12_nlin_xfm.apply(lambda x: x.source),
                            lsq6_mask_file=lambda df:
                              df.lsq12_nlin_xfm.apply(lambda x: x.source.mask if x.source.mask else ""),
                            nlin_file=lambda df: df.lsq12_nlin_xfm.apply(lambda x: x.resampled),
                            common_space_file=lambda df: df.xfm_to_common.apply(lambda x: x.resampled)
                                                if options.mbm.common_space.do_common_space_registration else None)
     .applymap(maybe_deref_path)
     .drop(["common_space_file"] if not options.mbm.common_space.do_common_space_registration else [], axis=1))

    transforms.to_csv("transforms.csv", index=False)

    transforms = (transforms.assign(label_files = mbm_result.maget_result.apply(lambda x: x.path))) #do this after transforms.to_csv

    determinants = (mbm_result.determinants.drop(["full_det", "nlin_det"], axis=1).applymap(maybe_deref_path))

    determinants.to_csv("determinants.csv", index=False)

    analysis = pd.merge(transforms, determinants, left_on='lsq12_nlin_xfm', right_on='inv_xfm').drop(["inv_xfm"], axis=1)

    if options.application.csv_file==None:
        analysis.to_csv("analysis.csv", index=False)
    else:
        input_csv = pd.read_csv(options.application.csv_file)
        analysis = pd.merge(input_csv, analysis, left_on='file', right_on='native_file').drop(["file"], axis=1)
        analysis.to_csv("analysis.csv", index=False)

    # # TODO moved here from inside `mbm` for now ... does this make most sense?
    # if options.mbm.segmentation.run_maget:
    #     import copy
    #     maget_options = copy.deepcopy(options)  #Namespace(maget=options)
    #     #maget_options
    #     #maget_options.maget = maget_options.mbm
    #     #maget_options.execution = options.execution
    #     #maget_options.application = options.application
    #     maget_options.application.output_directory = os.path.join(options.application.output_directory, "segmentation")
    #     maget_options.maget = options.mbm.maget
    #
    #     fixup_maget_options(maget_options=maget_options.maget,
    #                         nlin_options=maget_options.mbm.nlin,
    #                         lsq12_options=maget_options.mbm.lsq12)
    #     del maget_options.mbm
    #
    #
    #     #def with_new_output_dir(img : MincAtom):
    #         #img = copy.copy(img)
    #         #img.pipeline_sub_dir = img.pipeline_sub_dir + img.output_dir
    #         #img.
    #         #return img.newname_with_suffix(suffix="", subdir="segmentation")
    #
    #     s.defer(maget([xfm.resampled for _ix, xfm in mbm_result.xfms.rigid_xfm.iteritems()],
    #                    options=maget_options,
    #                    prefix="%s_MAGeT" % prefix,
    #                    output_dir=os.path.join(options.application.output_directory, prefix + "_processed")))

    return Result(stages=s, output=mbm_result)


def common_space(mbm_result, options):
    s = Stages()

    # TODO: the interface of this function (basically a destructive 'id' function) is horrific
    # TODO: instead, copy the mbm_result here ??

    if not options.mbm.common_space.common_space_model:
        raise ValueError("No common space template provided!")
    if not options.mbm.common_space.common_space_mask:
        warnings.warn("No common space mask provided ... might be OK if your consensus average mask is OK")
    # TODO allow lsq6 registration as well ...
    common_space_model = MincAtom(options.mbm.common_space.common_space_model,
                                  # TODO fix the subdirectories!
                                  mask=MincAtom(options.mbm.common_space.common_space_mask,
                                                pipeline_sub_dir=os.path.join(
                                                    options.application.output_directory,
                                                    options.application.pipeline_name + "_processed"))
                                  if options.mbm.common_space.common_space_mask else None,
                                  pipeline_sub_dir=os.path.join(options.application.output_directory,
                                                                options.application.pipeline_name + "_processed"))

    # TODO allow different lsq12/nlin config params than the ones used in MBM ...
    # full_hierarchy = get_nonlinear_configuration_from_options(nlin_protocol=options.mbm.nlin.nlin_protocol,
    #                                                          reg_method=options.mbm.nlin.reg_method,
    #                                                          file_resolution=options.registration.resolution)
    # WEIRD ... see comment in lsq12_nlin code ...
    # nlin_conf  = full_hierarchy.confs[-1] if isinstance(full_hierarchy, MultilevelANTSConf) else full_hierarchy
    # also weird that we need to call get_linear_configuration_from_options here ... ?
    #    nlin_build_model_component = model_building_with_initial_target_generation(
    #                                   final_model_building_component=nlin_build_model_component,
    #                                   prelim_model_building_component=prelim_nlin_build_model_component)

    # TODO don't use name 'x_module' for something that's technically not a module ... perhaps unit/component?
    nlin_component = get_nonlinear_component(reg_method=options.mbm.nlin.reg_method)

    lsq12_conf = get_linear_configuration_from_options(conf=options.mbm.lsq12,
                                                       transform_type=LinearTransType.lsq12,
                                                       file_resolution=options.registration.resolution)
    # N.B.: options.registration.resolution has been *updated* correctly by mbm( ). sigh ...
    model_to_common = s.defer(lsq12_nlin(source=mbm_result.avg_img, target=common_space_model,
                                         lsq12_conf=lsq12_conf, nlin_module=nlin_component,
                                         resolution=options.registration.resolution,
                                         nlin_options=options.mbm.nlin.nlin_protocol,  # =nlin_conf,
                                         resample_source=True))

    model_common = s.defer(mincresample_new(img=mbm_result.avg_img,
                                            xfm=model_to_common.xfm, like=common_space_model,
                                            postfix="_common"))

    #overall_xfms_to_common = [s.defer(concat_xfmhandlers([rigid_xfm, nlin_xfm, model_to_common]))
    #                          for rigid_xfm, nlin_xfm in zip(mbm_result.xfms.rigid_xfm,
    #                                                         mbm_result.xfms.lsq12_nlin_xfm)]

    xfms_to_common = [s.defer(concat_xfmhandlers([nlin_xfm, model_to_common]))
                      for nlin_xfm in mbm_result.xfms.lsq12_nlin_xfm]

    mbm_result.xfms = mbm_result.xfms.assign(xfm_to_common=xfms_to_common)
                                            #,overall_xfm_to_common=overall_xfms_to_common)

    if options.mbm.stats.calc_stats:
        log_nlin_det_common, log_full_det_common = (
            [dets.map(lambda d:
                      s.defer(mincresample_new(
                          img=d,
                          xfm=model_to_common.xfm,
                          like=common_space_model,
                          postfix="_common")))
             for dets in (mbm_result.determinants.log_nlin_det, mbm_result.determinants.log_full_det)])

        mbm_result.determinants = mbm_result.determinants.assign(log_nlin_det_common=log_nlin_det_common,
                                                                 log_full_det_common=log_full_det_common)

    mbm_result.model_common = model_common

    return Result(stages=s, output=mbm_result)


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
    # options required?  Also, shouldn't options.registration be a required input (as it contains `input_space`) ...?
    targets = registration_targets(lsq6_conf=options.mbm.lsq6,
                                   app_conf=options.application,
                                   first_input_file=imgs[0].path)

    # TODO this is quite tedious and duplicates stuff in the registration chain ...
    resolution = (options.registration.resolution or
                  get_resolution_from_file(targets.registration_standard.path))
    options.registration = options.registration.replace(resolution=resolution)


    # FIXME: this needs to go outside of the `mbm` function to avoid being run from within other pipelines (or
    # those other pipelines need to turn off this option)
    if options.mbm.segmentation.run_maget or options.mbm.maget.maget.mask:

        # temporary fix...?
        if options.mbm.maget.maget.mask and not options.mbm.segmentation.run_maget:
            # which means that --no-run-maget was specified
            if options.mbm.maget.maget.atlas_lib == None:
                # clearly you do not want to run MAGeT at any point in this pipeline
                err_msg_maget = "\nYou specified not to run MAGeT using the " \
                                "--no-run-maget flag. However, the code also " \
                                "wants to use MAGeT to generate masks for your " \
                                "input files after the 6 parameter alignment (lsq6). " \
                                "Because you did not specify a MAGeT atlas library " \
                                "this can not be done. \nTo run the pipeline without " \
                                "using MAGeT to mask your input files, please also " \
                                "specify: \n--maget-no-mask\n"
                raise ValueError(err_msg_maget)

        import copy
        maget_options = copy.deepcopy(options)  #Namespace(maget=options)
        #maget_options
        #maget_options.maget = maget_options.mbm
        #maget_options.execution = options.execution
        #maget_options.application = options.application
        #maget_options.application.output_directory = os.path.join(options.application.output_directory, "segmentation")
        maget_options.maget = options.mbm.maget

        fixup_maget_options(maget_options=maget_options.maget,
                            nlin_options=maget_options.mbm.nlin,
                            lsq12_options=maget_options.mbm.lsq12)
        del maget_options.mbm

        #def with_new_output_dir(img : MincAtom):
            #img = copy.copy(img)
            #img.pipeline_sub_dir = img.pipeline_sub_dir + img.output_dir
            #img.
            #return img.newname_with_suffix(suffix="", subdir="segmentation")

    # FIXME it probably makes most sense if the lsq6 module itself (even within lsq6_nuc_inorm) handles the run_lsq6
    # setting (via use of the identity transform) since then this doesn't have to be implemented for every pipeline
    if options.mbm.lsq6.run_lsq6:
        lsq6_result = s.defer(lsq6_nuc_inorm(imgs=imgs,
                                             resolution=resolution,
                                             registration_targets=targets,
                                             lsq6_dir=lsq6_dir,
                                             lsq6_options=options.mbm.lsq6))
    else:
        # FIXME the code shouldn't branch here based on run_lsq6 (which should probably
        # be part of the lsq6 options rather than the MBM ones; see comments on #287.
        # TODO don't actually do this resampling if not required (i.e., if the imgs already have the same grids)??
        # however, for now need to add the masks:
        identity_xfm = s.defer(param2xfm(out_xfm=FileAtom(name=os.path.join(lsq6_dir, 'tmp', "id.xfm"),
                                                          pipeline_sub_dir=lsq6_dir,
                                                          output_sub_dir='tmp')))
        lsq6_result  = [XfmHandler(source=img, target=img, xfm=identity_xfm,
                                   resampled=s.defer(mincresample_new(img=img,
                                                                      like=targets.registration_standard,
                                                                      xfm=identity_xfm)))
                        for img in imgs]
    # what about running nuc/inorm without a linear registration step??

    if options.mbm.maget.maget.mask:

        masking_imgs = copy.deepcopy([xfm.resampled for xfm in lsq6_result])
        masked_img = (s.defer(maget_mask(imgs=masking_imgs,
                                         resolution=resolution,
                                         maget_options=maget_options.maget,
                                         pipeline_sub_dir=os.path.join(options.application.output_directory,
                                                                       "%s_atlases" % prefix))))

        masked_img.index = masked_img.apply(lambda x: x.path)

        # replace any masks of the resampled images with the newly created masks:
        for xfm in lsq6_result:
            xfm.resampled = masked_img.loc[xfm.resampled.path]
    else:
        warnings.warn("Not masking your images from atlas masks after LSQ6 alignment ... probably not what you want "
                      "(this can have negative effects on your registration and statistics)")

    #full_hierarchy = get_nonlinear_configuration_from_options(nlin_protocol=options.mbm.nlin.nlin_protocol,
    #                                                          flag_nlin_protocol=next(iter(options.mbm.nlin.flags_.nlin_protocol)),
    #                                                         reg_method=options.mbm.nlin.reg_method,
    #                                                          file_resolution=resolution)

    #I = TypeVar("I")
    #X = TypeVar("X")
    #def wrap_minc(nlin_module: NLIN[I, X]) -> type[NLIN[MincAtom, XfmAtom]]:
    #    class N(NLIN[MincAtom, XfmAtom]): pass

    # TODO now the user has to call get_nonlinear_component followed by parse_<...>; previously various things
    # like lsq12_nlin_pairwise all branched on the reg_method so one didn't have to call get_nonlinear_component;
    # they could still do this if it can be done safety (i.e., not breaking assumptions of various nonlinear units)
    nlin_module = get_nonlinear_component(reg_method=options.mbm.nlin.reg_method)

    nlin_build_model_component = get_model_building_procedure(options.mbm.nlin.reg_strategy,
                                                              # was: model_building.reg_strategy
                                                              reg_module=nlin_module)

    # does this belong here?
    # def model_building_with_initial_target_generation(prelim_model_building_component,
    #                                                   final_model_building_component):
    #     class C(final_model_building_component):
    #         @staticmethod
    #         def build_model(imgs,
    #                         conf     : BuildModelConf,
    #                         nlin_dir,
    #                         nlin_prefix,
    #                         initial_target,
    #                         output_name_wo_ext = None): pass
    #
    #     return C

    #if options.mbm.model_building.prelim_reg_strategy is not None:
    #    prelim_nlin_build_model_component = get_model_building_procedure(options.mbm.model_building.prelim_reg_strategy,
    #                                                                     reg_module=nlin_module)
    #    nlin_build_model_component = model_building_with_initial_target_generation(
    #                                   final_model_building_component=nlin_build_model_component,
    #                                   prelim_model_building_component=prelim_nlin_build_model_component)

    # TODO don't use name 'x_module' for something that's technically not a module ... perhaps unit/component?

    # TODO tedious: why can't parse_build_model_protocol handle the null protocol case? is this something we want?
    nlin_conf = (nlin_build_model_component.parse_build_model_protocol(
                                              options.mbm.nlin.nlin_protocol, resolution=resolution)
                 if options.mbm.nlin.nlin_protocol is not None
                 else nlin_build_model_component.get_default_build_model_conf(resolution=resolution))

    lsq12_nlin_result = s.defer(lsq12_nlin_build_model(nlin_module=nlin_build_model_component,
                                                       imgs=[xfm.resampled for xfm in lsq6_result],
                                                       lsq12_dir=lsq12_dir,
                                                       nlin_dir=nlin_dir,
                                                       nlin_prefix=prefix,
                                                       resolution=resolution,
                                                       lsq12_conf=options.mbm.lsq12,
                                                       nlin_conf=nlin_conf))  #options.mbm.nlin

    inverted_xfms = [s.defer(invert_xfmhandler(xfm)) for xfm in lsq12_nlin_result.output]

    if options.mbm.stats.stats_kernels:
        determinants = s.defer(determinants_at_fwhms(
                                   xfms=inverted_xfms,
                                   inv_xfms=lsq12_nlin_result.output,
                                   blur_fwhms=options.mbm.stats.stats_kernels))
    else:
        determinants = None

    overall_xfms = [s.defer(concat_xfmhandlers([rigid_xfm, lsq12_nlin_xfm]))
                    for rigid_xfm, lsq12_nlin_xfm in zip(lsq6_result, lsq12_nlin_result.output)]

    output_xfms = (pd.DataFrame({ "rigid_xfm"      : lsq6_result,  # maybe don't return this if LSQ6 not run??
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


    # FIXME moved above rest of registration for debugging ... shouldn't use and destructively modify lsq6_result!!!
    if options.mbm.segmentation.run_maget:
        maget_options = copy.deepcopy(maget_options)
        maget_options.maget.maget.mask = maget_options.maget.maget.mask_only = False   # already done above
        # use the original masks here otherwise the masking step will be re-run due to the previous masking run's
        # masks having been applied to the input images:
        maget_result = s.defer(maget([xfm.resampled for xfm in lsq6_result],
                               #[xfm.resampled for _ix, xfm in mbm_result.xfms.rigid_xfm.iteritems()],
                               options=maget_options,
                               prefix="%s_MAGeT" % prefix,
                               output_dir=os.path.join(output_dir, prefix + "_processed")))
        # FIXME add pipeline dir to path and uncomment!
        #maget.to_csv(path_or_buf="segmentations.csv", columns=['img', 'voted_labels'])


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

    # should also move outside `mbm` function ...
    #if options.mbm.thickness.run_thickness:
    #    if not options.mbm.segmentation.run_maget:
    #        warnings.warn("MAGeT files (atlases, protocols) are needed to run thickness calculation.")
    #    # run MAGeT to segment the nlin average:
    #    import copy
    #    maget_options = copy.deepcopy(options)  #Namespace(maget=options)
    #    maget_options.maget = options.mbm.maget
    #    del maget_options.mbm
    #    segmented_avg = s.defer(maget(imgs=[lsq12_nlin_result.avg_img],
    #                                  options=maget_options,
    #                                  output_dir=os.path.join(options.application.output_directory,
    #                                                          prefix + "_processed"),
    #                                  prefix="%s_thickness_MAGeT" % prefix)).ix[0].img
    #    thickness = s.defer(cortical_thickness(xfms=pd.Series(inverted_xfms), atlas=segmented_avg,
    #                                           label_mapping=FileAtom(options.mbm.thickness.label_mapping),
    #                                           atlas_fwhm=0.56, thickness_fwhm=0.56))  # TODO magic fwhms
    #    # TODO write CSV -- should `cortical_thickness` do this/return a table?

    output = Namespace(avg_img=lsq12_nlin_result.avg_img, xfms=output_xfms, determinants=determinants)

    if options.mbm.segmentation.run_maget:
        output.maget_result = maget_result

    return Result(stages=s, output=output)


# TODO move to arguments file?
def _mk_common_space_parser(parser : ArgParser):
    group = parser.add_argument_group("Common space options", "Options for registration/resampling to common (db) space.")
    group.add_argument("--common-space-model", dest="common_space_model",
                       type=str, help="Run MAGeT segmentation on the images.")
    group.add_argument("--common-space-mask", dest="common_space_mask",
                       type=str, help="Mask for common space model")
    group.set_defaults(do_common_space_registration=True)
    group.add_argument("--common-space-registration", dest="do_common_space_registration",
                       action="store_true", help="Do registration to common (db) space. [default]")
    group.add_argument("--no-common-space-registration", dest="do_common_space_registration",
                       default=True, action="store_false", help="Skip registration to common (db) space.")
    return parser


common_space_parser = AnnotatedParser(parser=BaseParser(_mk_common_space_parser(ArgParser(add_help=False)),
                                                        "common_space"),
                                      namespace="common_space")


def _mk_model_building_parser(parser : ArgParser):
    group = parser.add_argument_group("Model building options",
                                      "Options specific to consensus model building")
    #group.add_argument("--registration-strategy", dest="reg_strategy",
    #                   default="build_model", choices=['build_model', 'pairwise', 'tournament'],
    #                   help="Process used for model construction [Default = %(default)s")
    #group.add_argument("--preliminary-registration-strategy", dest="preliminary_reg_strategy",
    #                   default=None, choices=['pairwise', 'tournament'],
    #                   help="Process used to construct a preliminary target for nonlinear model building "
    #                        "(use with '--registration-strategy=build_model' only!)")
    #group.add_argument("--preliminary-registration-protocol", dest="preliminary_reg_protocol",
    #                   default=None, type=str,
    #                   help="Protocol file for the optional preliminary model building")
    group.add_argument("--pairwise-nlin-max-pairs", default=None, type=int, dest="prelim_nlin_max_pairs",
                       help="Maximum number of nonlinear registrations per input file "
                            "for preliminary pairwise nonlinear model construction [default = %(default)s")
    group.add_argument("--pairwise-nlin-max-images", default=25, type=int, dest="prelim_nlin_max_images",
                       help="Maximum number of images to use "
                            "for preliminary pairwise nonlinear model construction [default = %(default)s")
    # TODO prelim_tournament_max_depth  # continue as a build model afterwards??
    return parser


model_building_parser = AnnotatedParser(parser=BaseParser(_mk_model_building_parser(ArgParser(add_help=False)),
                                                          "model_building"),
                                        namespace="model_building")


def mk_mbm_parser(with_common_space : bool = True):
    return CompoundParser(
             [lsq6_parser,
              lsq12_parser,
              nlin_parser,
              model_building_parser,
              stats_parser,
              #thickness_parser,
              AnnotatedParser(parser=maget_parsers, namespace="maget", prefix="maget"),
              # TODO note that the maget-specific flags (--mask, --masking-method, etc., also get the "maget-" prefix)
              # which could be changed by putting in the maget-specific parser separately from its lsq12, nlin parsers
              segmentation_parser] + ([common_space_parser] if with_common_space else []))


# TODO cast to MBMConf?
mbm_application = mk_application(parsers=[AnnotatedParser(parser=mk_mbm_parser(), namespace='mbm')],
                                 pipeline=mbm_pipeline)

if __name__ == "__main__":
    mbm_application()
