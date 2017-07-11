#!/usr/bin/env python3

import copy
import glob
import warnings
from collections import defaultdict
from configargparse import ArgParser
import os
from typing import List
import pandas as pd
import numpy  as np

from pydpiper.core.arguments        import (lsq12_parser, nlin_parser, stats_parser,
                                            CompoundParser, AnnotatedParser, BaseParser)
from pydpiper.core.stages import Stages, Result
from pydpiper.execution.application import mk_application
from pydpiper.minc.analysis import voxel_vote
from pydpiper.minc.files            import MincAtom, XfmAtom
from pydpiper.minc.registration     import (check_MINC_input_files, lsq12_nlin, custom_formatwarning,
                                            get_nonlinear_configuration_from_options,
                                            get_linear_configuration_from_options, LinearTransType,
                                            mincresample_new, mincmath, Interpolation, xfmconcat, xfminvert,
                                            get_nonlinear_component)

warnings.formatwarning = custom_formatwarning


# TODO move to a more sensible location !!
def get_imgs(options):
    #options : application_options
    if options.csv_file and options.files:
        raise ValueError("both --csv-file and --files specified ...")

    if options.csv_file:
        try:
            csv = pd.read_csv(options.csv_file)
        except:
            warnings.warn("couldn't read csv ... did you supply `file` column?")
            raise

        # FIXME check `file` column is present ...

        csv_base = os.path.dirname(options.csv_file)

        if hasattr(csv, 'mask_file'):
            masks = [MincAtom(os.path.join(csv_base, mask),
                              pipeline_sub_dir=os.path.join(options.output_directory,
                                                            options.pipeline_name + "_processed"))
                     if isinstance(mask, str) else None  # better way to handle missing (nan) values?
                     for mask in csv.mask_file]
        else:
            masks = [None] * len(csv.file)

        imgs = [MincAtom(os.path.join(csv_base, name), mask=mask,
                         pipeline_sub_dir=os.path.join(options.output_directory,
                                                       options.pipeline_name + "_processed"))
                # TODO does anything break if we make imgs a pd.Series?
                for name, mask in zip(csv.file, masks)]
    elif options.files:
        imgs = [MincAtom(name, pipeline_sub_dir=os.path.join(options.output_directory,
                                                             options.pipeline_name + "_processed"))
                for name in options.files]
    else:
        raise ValueError("No images supplied")
    return imgs


def find_by(f, xs, on_empty=None):  # TODO move to util
    for x in xs:
        if f(x):
            return x
    if on_empty is not None:
        return on_empty()
    else:
        raise ValueError("Nothing in iterable satisfies the predicate")


def get_atlases(maget_options, pipeline_sub_dir : str):

    atlas_dir, atlas_csv = maget_options.atlas_lib, maget_options.atlas_csv
    max_templates = maget_options.max_templates

    if atlas_dir is not None and atlas_csv is not None:
        raise ValueError("Atlases specified via both directory and csv; what should I do?")

    if atlas_dir is not None:
        atlases = atlases_from_dir(atlas_lib=atlas_dir,
                                   pipeline_sub_dir=pipeline_sub_dir)
    elif atlas_csv is not None:
        atlases = atlases_from_csv(atlas_csv, pipeline_sub_dir=pipeline_sub_dir)
    else:
        raise ValueError("Need either an atlas_dir or atlas_csv ...")

    if len(atlases) == 0:
        raise ValueError("Zero atlases found in specified location '%s' ..." % (atlas_dir or atlas_csv))

    # TODO check the atlases exist (known in atlases_from_dir case!)/are readable files?!
    # TODO issue a warning if not all atlases used or if more atlases requested than available?
    # TODO also, doesn't slicing with a higher number (i.e., if max_templates > n) go to the end of the list anyway?
    # TODO arbitrary; could choose atlases better ...
    # TODO probably the [:max_templates] shouldn't be done here but in the application?
    return atlases[:max_templates]


def atlases_from_csv(atlas_csv : str, pipeline_sub_dir : str) -> pd.Series:
    d = os.path.dirname(atlas_csv)
    df = (pd.read_csv(atlas_csv, usecols=["file", "mask_file", "label_file"])
          .apply(axis=1, func=lambda row:
                   MincAtom(name=os.path.join(d, row.file), pipeline_sub_dir=pipeline_sub_dir,
                            mask=MincAtom(os.path.join(d, row.mask_file), pipeline_sub_dir=pipeline_sub_dir),
                            labels=MincAtom(os.path.join(d, row.label_file), pipeline_sub_dir=pipeline_sub_dir))))

    #if np.isnan(df).any().any():
    #    raise ValueError("missing values in atlas CSV, currently not supported")

    return df


def atlases_from_dir(atlas_lib : str, pipeline_sub_dir : str):
    """Read in atlases from directory specified and return them as processed MincAtoms (with masks and labels).
    Assumes atlas/label/mask groups have one of the following naming schemes:
    (1) name_average.mnc, name_labels.mnc, name_mask.mnc
    (2) name.mnc, name_labels.mnc, name_mask.mnc
    Note that all three files are required even if masking is not used."""

    return process_atlas_files(glob.iglob(os.path.join(atlas_lib, "*.mnc")), pipeline_sub_dir=pipeline_sub_dir)


def process_atlas_files(filenames : List[str], pipeline_sub_dir) -> List[MincAtom]:

    suffixes = ["_average.mnc", "_mask.mnc", "_labels.mnc"] + [".mnc"]  # N.B.: order matters ...

    d = defaultdict(dict)  # TODO: rename `d`
    for filename in filenames:
        suffix = find_by(filename.endswith, suffixes)
        base = filename[:-len(suffix)]   # :-l
        d[base][suffix] = filename

    grouped_atlas_files = {}
    for group, files in d.items():
        img_file   = files.get("_average.mnc") or files.get(".mnc")  # FIXME: is this OK when just _mask, _labels exist?
        mask_file  = files.get("_mask.mnc")
        label_file = files.get("_labels.mnc")
        if len(files) > 3 or not (img_file and mask_file and label_file):
            raise ValueError("atlas filename conventions are wrong for atlas '%s'" % group)
        else:
            # TODO: screw around with filenames/directories as usual
            grouped_atlas_files[group] = MincAtom(name=img_file,
                                                  mask=MincAtom(name=mask_file, pipeline_sub_dir=pipeline_sub_dir),
                                                  labels=MincAtom(name=label_file, pipeline_sub_dir=pipeline_sub_dir),
                                                  pipeline_sub_dir=pipeline_sub_dir)

    return pd.Series(list(grouped_atlas_files.values()))


def maget_mask(imgs : List[MincAtom], maget_options, resolution : float,
               pipeline_sub_dir : str, atlases=None):

    s = Stages()

    resample  = np.vectorize(mincresample_new, excluded={"extra_flags"})
    defer     = np.vectorize(s.defer)

    original_imgs = imgs
    imgs = copy.deepcopy(imgs)
    original_imgs = pd.Series(original_imgs, index=[img.path for img in original_imgs])
    for img in imgs:
        img.output_sub_dir = os.path.join(img.output_sub_dir, "masking")

    # TODO dereference maget_options -> maget_options.maget outside maget_mask call?
    if atlases is None:
        atlases = get_atlases(maget_options.maget, pipeline_sub_dir=pipeline_sub_dir)

    lsq12_conf = get_linear_configuration_from_options(maget_options.lsq12,
                                                       LinearTransType.lsq12,
                                                       resolution)

    #nlin_module = get_nonlinear_component(reg_method=options.mbm.nlin.reg_method)

    #masking_nlin_hierarchy = get_nonlinear_configuration_from_options(maget_options.maget.masking_nlin_protocol,
    #                                                                  next(iter(maget_options.maget.flags_.masking_nlin_protocol)),
    #                                                                  maget_options.maget.mask_method,
    #                                                                  resolution)

    masking_nlin_component = get_nonlinear_component(reg_method=maget_options.maget.mask_method)

    masking_nlin_conf = (masking_nlin_component.parse_protocol_file(
                           maget_options.maget.masking_nlin_protocol, resolution=resolution)
                         if maget_options.maget.masking_nlin_protocol is not None
                         else masking_nlin_component.get_default_conf(resolution=resolution))

    # TODO lift outside then delete
    #masking_imgs = copy.deepcopy(imgs)
    #for img in masking_imgs:
    #    img.pipeline_sub_dir = os.path.join(img.pipeline_sub_dir, "masking")

    masking_alignments = pd.DataFrame({ 'img'   : img,
                                        'atlas' : atlas,
                                        'xfm'   : s.defer(
                                          lsq12_nlin(source=img, target=atlas,
                                                     lsq12_conf=lsq12_conf,
                                                     nlin_options=maget_options.maget.masking_nlin_protocol,
                                                     #masking_nlin_conf,
                                                     resolution=resolution,
                                                     nlin_module=masking_nlin_component,
                                                     resample_source=False))}
                                      for img in imgs for atlas in atlases)

    # propagate a mask to each image using the above `alignments` as follows:
    # - for each image, voxel_vote on the masks propagated to that image to get a suitable mask
    # - run mincmath -clobber -mult <img> <voted_mask> to apply the mask to the files
    masked_img = (
        masking_alignments
        .assign(resampled_mask=lambda df: defer(resample(img=df.atlas.apply(lambda x: x.mask),
                                                         xfm=df.xfm.apply(lambda x: x.xfm),
                                                         like=df.img,
                                                         invert=True,
                                                         interpolation=Interpolation.nearest_neighbour,
                                                         postfix="-input-mask",
                                                         subdir="tmp",
                                                         # TODO annoying hack; fix mincresample(_mask) ...:
                                                         #new_name_wo_ext=df.apply(lambda row:
                                                         #    "%s_to_%s-input-mask" % (row.atlas.filename_wo_ext,
                                                         #                             row.img.filename_wo_ext),
                                                         #    axis=1),
                                                         extra_flags=("-keep_real_range",))))
        .groupby('img', as_index=False)
        .aggregate({'resampled_mask' : lambda masks: list(masks)})
        .rename(columns={"resampled_mask" : "resampled_masks"})
        .assign(voted_mask=lambda df: df.apply(axis=1,
                                               func=lambda row:
                                                 s.defer(mincmath(op="max", vols=sorted(row.resampled_masks),
                                                                  new_name="%s_max_mask" % row.img.filename_wo_ext,
                                                                  subdir="tmp"))))
        .apply(axis=1, func=lambda row: row.img._replace(mask=row.voted_mask)))

    # resample the atlas images back to the input images:
    # (note: this doesn't modify `masking_alignments`, but only stages additional outputs)
    masking_alignments.assign(resampled_img=lambda df:
      defer(resample(img=df.atlas,
                     xfm=df.xfm.apply(lambda x: x.xfm),
                     subdir="tmp",
                     # TODO delete this stupid hack:
                     #new_name_wo_ext=df.apply(lambda row:
                     #  "%s_to_%s-resampled" % (row.atlas.filename_wo_ext,
                     #                          row.img.filename_wo_ext),
                     #                          axis=1),
                     like=df.img, invert=True)))

    for img in masked_img:
        img.output_sub_dir = original_imgs.loc[img.path].output_sub_dir

    return Result(stages=s, output=masked_img)


# TODO make a non-destructive version of this that creates a new options object ... it should take an overall options
# object, copy it, and put the maget options at top level.
def fixup_maget_options(lsq12_options, nlin_options, maget_options):

    # This function is used by programs that want to incorporate MAGeT
    # into a larger pipeline (for instance MBM or the twolevel code).
    # The lsq12 alignment is often performed twice. Once in order to
    # create masks (for MAGeT) and then for the actual pipeline.
    # The naming convention we use (naming the different stages of
    # the lsq12 alignment _lsq12_0.xfm, _lsq12_1.xfm etc.) conflicts
    # when the lsq12 protocols are different for the different stages.
    # So for now we won't allow it, and indicate that here
    if maget_options.lsq12.protocol and lsq12_options.protocol and \
        (maget_options.lsq12.protocol != lsq12_options.protocol):
        flag_maget = next(iter(maget_options.lsq12.flags_.protocol))
        flag_lsq12 = next(iter(lsq12_options.flags_.protocol))
        error_msg = "\n\nYou have specified a different lsq12 protocol " \
                    "for MAGeT as compared to the lsq12 protocol for the " \
                    "main registration: \n\n" + flag_maget + " " + \
                    maget_options.lsq12.protocol + "\n" + flag_lsq12 + " " + \
                    lsq12_options.protocol + "\n\nOne lsq12 protocol can be used " \
                    "for both parts of the pipeline. Please decide which of the two " \
                    "you want to use, and specify only that one using the " + flag_lsq12 \
                    + " flag.\n\n"
        raise ValueError(error_msg)

    # if both the maget_options.lsq12.protocol and the lsq12_options
    if maget_options.lsq12.protocol is None:
        maget_options.lsq12.protocol = lsq12_options.protocol

    if maget_options.nlin.nlin_protocol is None:
        if maget_options.nlin.reg_method == nlin_options.reg_method:
            maget_options.nlin.nlin_protocol = nlin_options.nlin_protocol
        else:
            raise ValueError("I'd use the nlin protocol for MAGeT as well but different programs are specified")

    if maget_options.maget.masking_nlin_protocol is None:
        if maget_options.maget.mask_method == maget_options.nlin.reg_method:
            maget_options.maget.masking_nlin_protocol = maget_options.nlin.nlin_protocol
        else:
            flag_masking_nlin_protocol = next(iter(maget_options.maget.flags_.masking_nlin_protocol))
            flag_nlin_part_nlin_protocol = next(iter(maget_options.nlin.flags_.nlin_protocol))
            flag_masking_method = next(iter(maget_options.maget.flags_.mask_method))
            flag_nlin_method = next(iter(maget_options.nlin.flags_.reg_method))
            error_string = "\n\nThe MAGeT non-linear masking protocol (" + \
                           flag_masking_nlin_protocol + ") " \
                           "was not specified. Tried to use the non-linear protocol for the main part of " \
                           "MAGeT (" + flag_nlin_part_nlin_protocol + \
                           "), but the registration methods for masking (" + \
                           flag_masking_method + \
                           ") and the main MAGeT procedure (" + \
                           flag_nlin_method + \
                           ") are different. (" + maget_options.maget.mask_method + " vs " + \
                           maget_options.nlin.reg_method + ")\n\nDefaults are:\n" + \
                           flag_masking_method + " minctracc\n" + \
                           flag_nlin_method + " minctracc\n" + \
                           flag_masking_nlin_protocol + " default_nlin_MAGeT_minctracc_prot.csv\n" + \
                           flag_nlin_part_nlin_protocol + " default_nlin_MAGeT_minctracc_prot.csv\n\nWhich can be " \
                           "found in either of these two locations:\n" \
                           "/hpf/largeprojects/MICe/tools/protocols/nonlinear/\n" \
                           "/axiom2/projects/software/non-linear-protocol/\n\n"
            raise ValueError(format(error_string))


# TODO support LSQ6 registrations??
def maget(imgs : List[MincAtom], options, prefix, output_dir, build_model_xfms=None):
    # FIXME prefix, output_dir aren't used !!

    s = Stages()

    maget_options = options.maget.maget

    resolution = options.registration.resolution  # TODO or get_resolution_from_file(...) -- only if file always exists!

    pipeline_sub_dir = os.path.join(options.application.output_directory,
                                    options.application.pipeline_name + "_atlases")

    atlases = get_atlases(maget_options, pipeline_sub_dir)

    lsq12_conf = get_linear_configuration_from_options(options.maget.lsq12,
                                                       transform_type=LinearTransType.lsq12,
                                                       file_resolution=resolution)

    nlin_component = get_nonlinear_component(options.maget.nlin.reg_method)



    #nlin_hierarchy = get_nonlinear_configuration_from_options(options.maget.nlin.nlin_protocol,
    #                                                          next(iter(options.maget.nlin.flags_.nlin_protocol)),
    #                                                          reg_method=options.maget.nlin.reg_method,
    #                                                          file_resolution=resolution)

    if maget_options.mask or maget_options.mask_only:

        # this used to return alignments but doesn't currently do so
        masked_img = s.defer(maget_mask(imgs=imgs,
                                        maget_options=options.maget, atlases=atlases,
                                        pipeline_sub_dir=pipeline_sub_dir + "_masking", # FIXME repeats all alignments!!!
                                        resolution=resolution))

        # now propagate only the masked form of the images and atlases:
        imgs    = masked_img
        #atlases = masked_atlases  # TODO is this needed?

    if maget_options.mask_only:
        # register each input to each atlas, creating a mask
        return Result(stages=s, output=masked_img)   # TODO rename `alignments` to `registrations`??
    else:
        if maget_options.mask:
            del masked_img
        # this `del` is just to verify that we don't accidentally use this later, since these potentially
        # coarser alignments shouldn't be re-used (but if the protocols for masking and alignment are the same,
        # hash-consing will take care of things), just the masked images they create; can be removed later
        # if a sensible use is found

        # images with labels from atlases
        # N.B.: Even though we've already registered each image to each initial atlas, this happens again here,
        #       but using `nlin_hierarchy` instead of `masking_nlin_hierarchy` as options.
        #       This is not 'work-efficient' in the sense that this computation happens twice (although
        #       hopefully at greater precision the second time!), but the idea is to run a coarse initial
        #       registration to get a mask and then do a better registration with that mask (though I'm not
        #       sure exactly when this is faster than doing a single registration).
        #       This _can_ allow the overall computation to finish more rapidly
        #       (depending on the relative speed of the two alignment methods/parameters,
        #       number of atlases and other templates used, number of cores available, etc.).
        atlas_labelled_imgs = (
            pd.DataFrame({ 'img'        : img,
                           'label_file' : s.defer(  # can't use `label` in a pd.DataFrame index!
                              mincresample_new(img=atlas.labels,
                                               xfm=s.defer(lsq12_nlin(source=img,
                                                                      target=atlas,
                                                                      nlin_module=nlin_component,
                                                                      lsq12_conf=lsq12_conf,
                                                                      nlin_options=options.maget.nlin.nlin_protocol,
                                                                      resolution=resolution,
                                                                      #nlin_conf=nlin_hierarchy,
                                                                      resample_source=False)).xfm,
                                               like=img,
                                               invert=True,
                                               interpolation=Interpolation.nearest_neighbour,
                                               extra_flags=('-keep_real_range',)))}
                         for img in imgs for atlas in atlases)
        )

        if maget_options.pairwise:

            def choose_new_templates(ts, n):
                # currently silly, but we might implement a smarter method ...
                # FIXME what if there aren't enough other imgs around?!  This silently goes weird ...
                return pd.Series(ts[:n+1])  # n+1 instead of n: choose one more since we won't use image as its own template ...

            # FIXME: the --max-templates flag is ambiguously named ... should be --max-new-templates
            # (and just use all atlases)
            templates = pd.DataFrame({ 'template' : choose_new_templates(ts=imgs,
                                                                         n=maget_options.max_templates - len(atlases))})
            # note these images are the masked ones if masking was done ...

            # the templates together with their (multiple) labels from the atlases (this merge just acts as a filter)
            labelled_templates = pd.merge(left=atlas_labelled_imgs, right=templates,
                                          left_on="img", right_on="template").drop('img', axis=1)

            # images with new labels from the templates
            imgs_and_templates = pd.merge(#left=atlas_labelled_imgs,
                                          left=pd.DataFrame({ "img" : imgs }).assign(fake=1),
                                          right=labelled_templates.assign(fake=1),
                                          on='fake')
                                          #left_on='img', right_on='template')  # TODO do select here instead of below?

            #if build_model_xfms is not None:
            #    # use path instead of full mincatom as key in case we're reading these in from a CSV:
            #    xfm_dict = { x.source.path : x.xfm for x in build_model_xfms }

            template_labelled_imgs = (
                imgs_and_templates
                .rename(columns={ 'label_file' : 'template_label_file' })
                # don't register template to itself, since otherwise atlases would vote on that template twice
                .select(lambda ix: imgs_and_templates.img[ix].path
                                     != imgs_and_templates.template[ix].path)  # FIXME hardcoded df name !!
                .assign(label_file=lambda df: df.apply(axis=1, func=lambda row:
                          s.defer(
                            mincresample_new(
                              img=row.template_label_file,
                              xfm=s.defer(
                                    lsq12_nlin(source=row.img,
                                               target=row.template,
                                               lsq12_conf=lsq12_conf,
                                               resolution=resolution,
                                               nlin_module=nlin_component,
                                               nlin_options=options.maget.nlin.nlin_protocol,
                                               #nlin_conf=nlin_hierarchy,
                                               resample_source=False)).xfm
                                  if build_model_xfms is None
                                  # use transforms from model building if we have them:
                                  else s.defer(
                                         xfmconcat([build_model_xfms[row.img.path],
                                                   s.defer(xfminvert(build_model_xfms[row.template.path],
                                                                     subdir="tmp"))])),
                              like=row.img,
                              invert=True,
                              interpolation=Interpolation.nearest_neighbour,
                              extra_flags=('-keep_real_range',)))))
            ) if len(imgs) > 1 else pd.DataFrame({ 'img' : [], 'label_file' : [] })
              # ... as no distinct templates to align if only one image supplied (#320)

            imgs_with_all_labels = pd.concat([atlas_labelled_imgs[['img', 'label_file']],
                                              template_labelled_imgs[['img', 'label_file']]],
                                             ignore_index=True)
        else:
            imgs_with_all_labels = atlas_labelled_imgs

        segmented_imgs = (
                imgs_with_all_labels
                .groupby('img')
                .aggregate({'label_file' : lambda resampled_label_files: list(resampled_label_files)})
                .rename(columns={ 'label_file' : 'label_files' })
                .reset_index()
                .assign(voted_labels=lambda df: df.apply(axis=1, func=lambda row:
                          s.defer(voxel_vote(label_files=row.label_files,
                                             output_dir=os.path.join(row.img.pipeline_sub_dir, row.img.output_sub_dir)))))
                .apply(axis=1, func=lambda row: row.img._replace(labels=row.voted_labels))
        )

        return Result(stages=s, output=segmented_imgs)


def maget_pipeline(options):

    imgs = get_imgs(options.application)
    check_MINC_input_files([img.path for img in imgs])
    # TODO fixup masking protocols ...

    if options.application.csv_file is not None:
        df = pd.read_csv(options.application.csv_file)
        # FIXME only works on MBM output, not, e.g., twolevel!
        if 'lsq12_nlin_xfm' in df.columns:
            csv_dir = os.path.dirname(options.application.csv_file)
            build_model_xfms = { row.file :
                                     XfmAtom(name=os.path.join(csv_dir, row.lsq12_nlin_xfm),
                                             pipeline_sub_dir=os.path.join(options.application.output_directory,
                                                                           options.application.pipeline_name
                                                                             + "_build_model_xfms"))
                                 for _, row in df.iterrows() }
        else:
            build_model_xfms = None
    else:
        build_model_xfms = None

    result = maget(imgs=imgs, options=options,
                   prefix=options.application.pipeline_name,
                   output_dir=options.application.output_directory,
                   build_model_xfms=build_model_xfms)

    (pd.DataFrame({ 'img_file'   : result.output.apply(lambda row: row.path),
                    'label_file' : result.output.apply(lambda row: row.labels.path),
                    #'mask_file'  : result.output.apply(lambda row: row.mask.path if row.mask else "NA")
                  })
        .to_csv(os.path.join(options.application.output_directory, "segmentations.csv"), index=False))

    return result


def _mk_maget_parser(parser : ArgParser):
    group = parser.add_argument_group("MAGeT options", "Options for running MAGeT.")
    group.add_argument("--atlas-library", dest="atlas_lib",  # can't make required=True since may not be using MAGeT :|
                       type=str,                             # TODO: check existence of this dir?
                       help="Directory of existing atlas/label pairs (alternative to --atlas-csv)")
    group.add_argument("--atlas-csv", dest="atlas_csv", type=str,
                       help="CSV containing (at least) `file`, `mask_file`, `label_file` columns "
                            "(alternative to --atlas-library)")
    group.add_argument("--pairwise", dest="pairwise",
                       action="store_true",
                       help="""If specified, register inputs to each other pairwise. [Default]""")
    group.add_argument("--no-pairwise", dest="pairwise",
                       action="store_false",
                       help="""If specified, only register inputs to atlases in library.""")
    parser.set_defaults(pairwise=True)
    group.add_argument("--mask", dest="mask",
                       action="store_true", default=True,
                       help="Create a mask for all images prior to handling labels. [Default = %(default)s]")
    group.add_argument("--no-mask", dest="mask",
                       action="store_false", default=True,
                       help="Opposite of --mask.")
    group.add_argument("--mask-only", dest="mask_only",
                       action="store_true", default=False,
                       help="Create a mask for all images only, do not run full algorithm. [Default = %(default)s]")
    group.add_argument("--max-templates", dest="max_templates",
                       default=25, type=int,
                       help="Maximum number of templates to generate. [Default = %(default)s]")
    group.add_argument("--masking-method", dest="mask_method",
                       default="minctracc", type=str,
                       help="Specify whether to use minctracc or ANTS for masking. [Default = %(default)s].")
    group.add_argument("--masking-nlin-protocol", dest="masking_nlin_protocol",
                       # TODO basically copied from nlin parser
                       type=str, default=None,
                       help="Can optionally specify a registration protocol that is different from nlin protocol. "
                            "Parameters must be specified as in either or the following examples: \n"
                            "applications_testing/test_data/minctracc_example_nlin_protocol.csv \n"
                            "applications_testing/test_data/mincANTS_example_nlin_protocol.csv \n"
                            "[Default = %(default)s]")
    return parser
    # maybe wire the masking-nlin-protocol to the nlin-protocol?

maget_parser = AnnotatedParser(parser=BaseParser(_mk_maget_parser(ArgParser(add_help=False)),
                                                 "maget"),
                               namespace="maget")

maget_parsers = CompoundParser([lsq12_parser, nlin_parser, maget_parser])

maget_application = mk_application(parsers=[AnnotatedParser(parser=maget_parsers,
                                                            namespace="maget")],
                                   pipeline=maget_pipeline)

if __name__ == "__main__":
    maget_application()
