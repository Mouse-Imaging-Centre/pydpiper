import os

from typing import List, Optional
from pydpiper.core.stages import Result, Stages
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.files import MincAtom, XfmAtom
from pydpiper.minc.nlin import NLIN
from pydpiper.minc.registration import LSQ6Conf, MultilevelMinctraccConf, MinctraccConf, default_linear_minctracc_conf, \
    LinearTransType, parse_minctracc_linear_protocol_file, default_lsq6_minctracc_conf, \
    default_rotational_minctracc_conf, rotational_minctracc, multilevel_minctracc, xfmconcat, mincresample, \
    Interpolation, lsq12_nlin, RegistrationTargets, nu_correct, default_inormalize_conf, inormalize, mincaverage, \
    mincbigaverage, create_quality_control_images
from pydpiper.minc.analysis import lin_from_nlin


def lsq6(imgs: List[MincAtom],
         target: MincAtom,
         resolution: float,
         conf: LSQ6Conf,
         resample_subdir: str = "resampled",
         resample_images: bool = False,
         post_alignment_xfm: XfmAtom = None,
         propagate_masks: bool = True,
         post_alignment_target: MincAtom = None) -> Result[List[XfmHandler]]:
    """
    post_alignment_xfm -- this is a transformation you want to concatenate with the
                          output of the transform aligning the imgs to the target.
                          The most obvious example is a native_to_standard.xfm transformation
                          from an initial model.
    post_alignment_target -- a MincAtom indicating the space that the post alignment
                             transformation brings you into

    The output name of the transformations generated in this function will be:

    imgs[i].output_sub_dir + "_lsq6.xfm"
    """
    s = Stages()
    xfms_to_target = []  # type: List[XfmHandler]

    if post_alignment_xfm and not post_alignment_target:
        raise ValueError("You've provided a post alignment transformation to lsq6() but not a MincAtom indicating the target for this transformation.")

    if not conf.run_lsq6:
        raise ValueError("You silly person... you've called lsq6(), but also specified --no-run-lsq6. That's not a very sensible combination of things.")

    # FIXME this is a stupid function: it's not very safe (note lack of type of argument) and rather redundant ...
    def conf_from_defaults(defaults) -> MultilevelMinctraccConf:
        conf = MultilevelMinctraccConf(
            [MinctraccConf(step_sizes=[defaults["step_factors"][i] * resolution] * 3,
                           blur_resolution=defaults["blur_factors"][i] * resolution,
                           use_gradient=defaults["gradients"][i],
                           use_masks=True,
                           linear_conf=(default_linear_minctracc_conf(LinearTransType.lsq6)
                                        .replace(w_translations=[defaults["translations"][i]] * 3,
                                                 simplex=defaults["simplex_factors"][i] * resolution)),
                           nonlinear_conf=None)
             for i in range(len(defaults["blur_factors"]))])  # FIXME: don't assume all lengths are equal
        return conf

    ############################################################################
    # alignment - switch on lsq6_method
    ############################################################################
    # FIXME the proliferations of LSQ6Confs vs. MultilevelMinctraccConfs here is very confusing
    if conf.protocol_file is not None:
        mt_conf = parse_minctracc_linear_protocol_file(filename=conf.protocol_file,
                                                       transform_type=LinearTransType.lsq6,
                                                       minctracc_conf=default_lsq6_minctracc_conf)
    if conf.lsq6_method == "lsq6_large_rotations":
        # still not convinced this shouldn't go inside rotational_minctracc somehow,
        # though you may want to override ...

        rotational_configuration = default_rotational_minctracc_conf
        if conf.protocol_file is None:
            defaults = {'blur_factors': [5],
                        'simplex_factors': [20], # this matches the old "--simplex 0.8" for 40micron
                        'step_factors': [10], # this matches the old "-g 0.4" for 40micron
                        'gradients': [False], #NEW
                        'translations': [8*resolution] #to keep it the same as before
                         }
            mt_conf = conf_from_defaults(defaults)
        first_conf, remainder_conf = mt_conf.split_first()

        # now call rotational_minctracc on all input images
        #TODO xfms_to_target_pt1
        xfms_to_target = [s.defer(rotational_minctracc(source=img, target=target,
                                                       mt_conf=first_conf,
                                                       rot_conf=rotational_configuration,
                                                       resolution=resolution,
                                                       output_name_wo_ext=None if post_alignment_xfm else
                                                                          img.output_sub_dir + "_lsq6"))
                          for img in imgs]
        #TODO do xfms_to_target_pt2=s.defer(lsq6_simple)
        #TODO xfms_to_target = xfm_concat on pt1 and pt2

    elif conf.lsq6_method == "lsq6_centre_estimation":
        if conf.protocol_file is None:
            defaults = {'blur_factors': [90, 35, 17, 9, 4],
                        'simplex_factors': [128, 64, 40, 28, 16],
                        'step_factors': [90, 35, 17, 9, 4],
                        'gradients': [False, False, False, True, False],
                        'translations': [0.4, 0.4, 0.4, 0.4, 0.4]
                        }
            mt_conf = conf_from_defaults(defaults)
        xfms_to_target = [s.defer(multilevel_minctracc(source=img, target=target,
                                                       conf=mt_conf,
                                                       transform_info=["-est_center", "-est_translations"]))
                          for img in imgs]
    elif conf.lsq6_method == "lsq6_simple":
        if conf.protocol_file is None:
            defaults = {'blur_factors': [17, 9, 4],
                        'simplex_factors': [40, 28, 16],
                        'step_factors': [17, 9, 4],
                        'gradients': [False, True, False],
                        'translations': [0.4, 0.4, 0.4]
                        }
            mt_conf = conf_from_defaults(defaults)  # FIXME print a warning?!
        xfms_to_target = [s.defer(multilevel_minctracc(source=img,
                                                       target=target,
                                                       conf=mt_conf))
                          for img in imgs]
    else:
        raise ValueError("bad lsq6 method: %s" % conf.lsq6_method)

    transform_types = [conf.linear_conf.transform_type.name for conf in mt_conf.confs]
    if not all(transform_type == "lsq6" for transform_type in transform_types):
        xfms_to_target = [s.defer(lin_from_nlin(xfm, "lsq6")) for xfm in xfms_to_target]

    if post_alignment_xfm:
        composed_xfms = [s.defer(xfmconcat([xfm.xfm, post_alignment_xfm],
                                           name=xfm.xfm.output_sub_dir + "_lsq6"))
                         for xfm in xfms_to_target]
        resampled_imgs = ([s.defer(mincresample(img=native_img,
                                                xfm=overall_xfm,
                                                like=post_alignment_target,
                                                interpolation=Interpolation.sinc,
                                                new_name_wo_ext=native_img.filename_wo_ext + "_lsq6",
                                                subdir=resample_subdir))
                           for native_img, overall_xfm in zip(imgs, composed_xfms)]
                          if resample_images else [None] * len(imgs))
        final_xfmhs = [XfmHandler(xfm=overall_xfm,
                                  target=post_alignment_target,
                                  source=native_img,
                                  # resample the input to the final lsq6 space
                                  # we should go back to basics in terms of the file name that we create here.
                                  # It should be fairly basic. Something along the lines of:
                                  # {orig_file_base}_resampled_lsq6.mnc
                                  # if we are still going to perform either non uniformity correction, or
                                  # intensity normalization, the following file is a temp file:
                                  resampled=resampled_img)
                       for native_img, resampled_img, overall_xfm in zip(imgs, resampled_imgs, composed_xfms)]
    else:
        final_xfmhs = xfms_to_target
        if resample_images:
            for native_img, xfm in zip(imgs, final_xfmhs):  # is xfm.resampled even mutable ?!
                xfm.resampled = s.defer(mincresample(img=native_img,
                                                     xfm=xfm.xfm,
                                                     like=xfm.target,
                                                     interpolation=Interpolation.sinc,
                                                     new_name_wo_ext=native_img.filename_wo_ext + "_lsq6",
                                                     subdir=resample_subdir))

    # we've just performed a 6 parameter alignment between a bunch of input files
    # and a target. The input files could have been the very initial input files to the
    # pipeline, and have no masks associated with them. In that case, and if the target does
    # have a mask, we should add masks to the resampled files now.
    if propagate_masks and resample_images:
        mask_to_add = post_alignment_target.mask if post_alignment_target else target.mask
        for xfm in final_xfmhs:
            if not xfm.resampled.mask:
                xfm.resampled.mask = mask_to_add

    # could have stuff for creating an average and/or QC images here, but seemingly we use this either
    # as part of lsq6_nuc_inorm or in more unusual situations such as the registration chain where
    # we first aggregate the results of multiple lsq6 calls and create a common montage, so haven't bothered ...

    # TODO return average, etc.?
    return Result(stages=s, output=final_xfmhs)


def lsq6_lsq12_nlin(source: MincAtom,
                    target: MincAtom,
                    lsq6_conf: LSQ6Conf,
                    lsq12_conf: MinctraccConf,
                    nlin_module: NLIN,
                    resolution: float,
                    nlin_options,  # nlin_module.Conf,  ??  want an nlin_module.Conf here ...
                    #nlin_conf: Union[MultilevelMinctraccConf, MultilevelANTSConf, ANTSConf],  # sigh ... ...
                    resampled_post_fix_string: str = None):
    """
    Full source to target registration. First calls the lsq6() module to align the
    source to the target rigidly, then calls lsq12_nlin for the linear and nonlinear
    alignment (*not* model building)
    :return:
    """
    s = Stages()


    # first run the rigid alignment
    # if an resampled_post_fix_string is given, don't use the resampling
    # from the lsq6() module, as it will create a standard name

    source_lsq6_to_target_xfm = s.defer(lsq6(imgs=[source],
                                             target=target,
                                             resolution=resolution,
                                             conf=lsq6_conf,
                                             resample_images=True if not resampled_post_fix_string else False))

    # resample the source file manually if an resampled_post_fix_string was provided:
    if resampled_post_fix_string:
        source_resampled = s.defer(mincresample(img=source,
                                                xfm=source_lsq6_to_target_xfm[0].xfm,
                                                like=target,
                                                interpolation=Interpolation.sinc,
                                                new_name_wo_ext=source.filename_wo_ext + "_lsq6_to_" + resampled_post_fix_string,
                                                subdir="resampled"))
    else:
        source_resampled = source_lsq6_to_target_xfm[0].resampled


    # then run the linear and nonlinear alignment:
    rigid_source_lsq12_and_nlin_xfm = s.defer(lsq12_nlin(source=source_resampled,
                                                         target=target,
                                                         lsq12_conf=lsq12_conf,
                                                         nlin_module=nlin_module,
                                                         resolution=resolution,
                                                         nlin_options=nlin_options,
                                                         resample_source=True if not resampled_post_fix_string else False))

    # resample the source file manually if an resampled_post_fix_string was provided:
    if resampled_post_fix_string:
        source_resampled_final = s.defer(mincresample(img=source_resampled,
                                                      xfm=rigid_source_lsq12_and_nlin_xfm.xfm,
                                                      like=target,
                                                      interpolation=Interpolation.sinc,
                                                      new_name_wo_ext=source.filename_wo_ext + "_lsq6_lsq12_and_nlin_to_" + resampled_post_fix_string,
                                                      subdir="resampled"))
        rigid_source_lsq12_and_nlin_xfm.resampled = source_resampled_final


    return Result(stages=s, output=rigid_source_lsq12_and_nlin_xfm)


def lsq6_nuc_inorm(imgs: List[MincAtom],
                   registration_targets: RegistrationTargets,
                   resolution: float,
                   lsq6_options: LSQ6Conf,
                   lsq6_dir: str,
                   create_qc_images: bool = True,
                   create_average: bool = True,
                   subject_matter: Optional[str] = None):
    s = Stages()

    # run the actual 6 parameter registration
    init_target = registration_targets.registration_native or registration_targets.registration_standard

    source_imgs_to_lsq6_target_xfms = s.defer(lsq6(imgs=imgs, target=init_target,
                                                   resolution=resolution,
                                                   conf=lsq6_options,
                                                   resample_images=not (lsq6_options.nuc or lsq6_options.inormalize),
                                                   post_alignment_xfm=registration_targets.xfm_to_standard,
                                                   post_alignment_target=registration_targets.registration_standard))

    xfms_to_final_target_space = [xfm_handler.xfm for xfm_handler in source_imgs_to_lsq6_target_xfms]

    # resample the mask from the initial model to native space
    # we can use it for either the non uniformity correction or
    # for intensity normalization later on
    masks_in_native_space = None  # type: List[MincAtom]
    if registration_targets.registration_standard.mask and \
        (lsq6_options.nuc or lsq6_options.inormalize):
        # we should apply the non uniformity correction in
        # native space. Given that there is a mask, we should
        # resample it to that space using the inverse of the
        # lsq6 transformation we have so far
        masks_in_native_space = [s.defer(mincresample(img=registration_targets.registration_standard.mask,
                                                      xfm=xfm_to_lsq6,
                                                      like=native_img,
                                                      interpolation=Interpolation.nearest_neighbour,
                                                      invert=True))
                                   if not native_img.mask else native_img.mask
                                 for native_img, xfm_to_lsq6 in zip(imgs, xfms_to_final_target_space)]

    # NUC
    nuc_imgs_in_native_space = None  # type: List[MincAtom]
    if lsq6_options.nuc:
        # if masks are around, they will be passed along to nu_correct,
        # if not we create a list with the same length as the number
        # of images with None values
        # what we get back here is a list of MincAtoms with NUC files
        # we will always apply a final resampling to these files,
        # so these will always be temp files
        nuc_imgs_in_native_space = [s.defer(nu_correct(src=native_img,
                                                       resolution=resolution,
                                                       mask=native_img_mask,
                                                       subject_matter=subject_matter,
                                                       subdir="tmp"))
                                    for native_img, native_img_mask
                                    in zip(imgs,
                                           masks_in_native_space if masks_in_native_space
                                                                 else [None] * len(imgs))]

    inorm_imgs_in_native_space = None  # type: List[MincAtom]
    if lsq6_options.inormalize:
        # TODO: this is still static
        inorm_conf = default_inormalize_conf
        input_imgs_for_inorm = nuc_imgs_in_native_space if nuc_imgs_in_native_space else imgs
        # same as with the NUC files, these intensity normalized files will be resampled
        # using the lsq6 transform no matter what, so these ones are temp files
        inorm_imgs_in_native_space = (
            [s.defer(inormalize(src=nuc_img,
                                conf=inorm_conf,
                                mask=native_img_mask,
                                subdir="tmp"))
             for nuc_img, native_img_mask in zip(input_imgs_for_inorm,
                                                 masks_in_native_space or [None] * len(input_imgs_for_inorm))])

    # the only thing left to check is whether we have to resample the NUC/inorm images to LSQ6 space:
    if lsq6_options.inormalize:
        # the final resampled files should be the normalized files resampled with the
        # lsq6 transformation
        final_resampled_lsq6_files = [s.defer(mincresample(
                                                img=inorm_img,
                                                xfm=xfm_to_lsq6,
                                                like=registration_targets.registration_standard,
                                                interpolation=Interpolation.sinc,
                                                new_name_wo_ext=inorm_img.filename_wo_ext + "_lsq6",
                                                subdir="resampled"))
                                      for inorm_img, xfm_to_lsq6
                                      in zip(inorm_imgs_in_native_space,
                                             xfms_to_final_target_space)]
    elif lsq6_options.nuc:
        # the final resampled files should be the non uniformity corrected files
        # resampled with the lsq6 transformation
        nuc_filenames_wo_ext_lsq6 = [nuc_img.filename_wo_ext + "_lsq6" for nuc_img in
                                     nuc_imgs_in_native_space]
        final_resampled_lsq6_files = [s.defer(mincresample(img=nuc_img,
                                                           xfm=xfm_to_lsq6,
                                                           like=registration_targets.registration_standard,
                                                           interpolation=Interpolation.sinc,
                                                           new_name_wo_ext=nuc_filename_wo_ext,
                                                           subdir="resampled"))
                                      for nuc_img, xfm_to_lsq6, nuc_filename_wo_ext
                                      in zip(nuc_imgs_in_native_space,
                                             xfms_to_final_target_space,
                                             nuc_filenames_wo_ext_lsq6)]
    else:
        # in this case neither non uniformity correction nor intensity normalization was applied,
        # so we must have passed `resample_source=True` to the actual lsq6 calls above, and thus:
        final_resampled_lsq6_files = [xfm.resampled for xfm in source_imgs_to_lsq6_target_xfms]

    # we've just performed a 6 parameter alignment between a bunch of input files
    # and a target. The input files could have been the very initial input files to the
    # pipeline, and have no masks associated with them. In that case, and if the target does
    # have a mask, we should add masks to the resampled files now.
    # TODO potentially done twice if neither nuc nor inorm corrections applied since `lsq6` also does this
    mask_to_add = registration_targets.registration_standard.mask
    for resampled_input in final_resampled_lsq6_files:
        if not resampled_input.mask:
            resampled_input.mask = mask_to_add

    if create_average:
        if lsq6_options.copy_header_info:
            s.defer(mincaverage(imgs=final_resampled_lsq6_files,
                                output_dir=lsq6_dir,
                                copy_header_from_first_input=True))
        else:
            s.defer(mincbigaverage(imgs=final_resampled_lsq6_files,
                                   output_dir=lsq6_dir))
        #s.defer(mincaverage(imgs=final_resampled_lsq6_files,
        #                    output_dir=lsq6_dir,
        #                    copy_header_from_first_input=lsq6_options.copy_header_info))

    if create_qc_images:
        s.defer(create_quality_control_images(imgs=final_resampled_lsq6_files,
                                              #montage_dir=lsq6_dir,  # FIXME
                                              montage_output=os.path.join(lsq6_dir, "LSQ6_montage")))

    # note that in the return, the registration target is given as "registration_standard".
    # the actual registration might have been between the input file and a potential
    # "registration_native", but since we concatenated that transform with the
    # native_to_standard.xfm, the "registration_standard" file is the correct target
    # with respect to the transformation that's returned
    #
    # TODO: potentially add more to this return. Perhaps we want to pass along
    #       non uniformity corrected / intensity normalized files in native space?
    return Result(stages=s, output=[XfmHandler(source=src_img,
                                               target=registration_targets.registration_standard,
                                               xfm=lsq6_xfm,
                                               resampled=final_resampled)
                                    for src_img, lsq6_xfm, final_resampled in
                                    zip(imgs, xfms_to_final_target_space, final_resampled_lsq6_files)])