#!/usr/bin/env python3
import copy
import glob
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
from pydpiper.minc.files            import MincAtom
from pydpiper.minc.registration     import (check_MINC_input_files, lsq12_nlin,
                                            get_nonlinear_configuration_from_options,
                                            get_linear_configuration_from_options, LinearTransType,
                                            mincresample_new, mincmath, Interpolation)


def find_by(f, xs, on_empty=None):  # TODO move to util
    for x in xs:
        if f(x):
            return x
    if on_empty is not None:
        return on_empty()
    else:
        raise ValueError("Nothing in iterable satisfies the predicate")


def read_atlas_dir(atlas_lib : str, pipeline_sub_dir : str) -> List[MincAtom]:
    """Read in atlases from directory specified return them as processed MincAtoms (with masks and labels).
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
        base = filename.rstrip(suffix)
        d[base][suffix] = filename
    # TODO in some error situations the last letter of the basename seems to be rstripped for some reason ...
    # (e.g., supply the 56um-init-model dir in place of an atlas dir)

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


def maget_mask(imgs : List[MincAtom], maget_options, resolution : float, pipeline_sub_dir : str, atlases=None):

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
        atlases =  atlases_from_dir(atlas_lib=maget_options.maget.atlas_lib,
                                    max_templates=maget_options.maget.max_templates,
                                    pipeline_sub_dir=pipeline_sub_dir)

    lsq12_conf = get_linear_configuration_from_options(maget_options.lsq12,
                                                       LinearTransType.lsq12,
                                                       resolution)

    masking_nlin_hierarchy = get_nonlinear_configuration_from_options(maget_options.maget.masking_nlin_protocol,
                                                                      maget_options.maget.mask_method,
                                                                      resolution)

    # TODO lift outside then delete
    #masking_imgs = copy.deepcopy(imgs)
    #for img in masking_imgs:
    #    img.pipeline_sub_dir = os.path.join(img.pipeline_sub_dir, "masking")

    masking_alignments = pd.DataFrame({ 'img'   : img,
                                        'atlas' : atlas,
                                        'xfm'   : s.defer(lsq12_nlin(source=img, target=atlas,
                                                                     lsq12_conf=lsq12_conf,
                                                                     nlin_conf=masking_nlin_hierarchy,
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
        .groupby('img', sort=False, as_index=False)
        # sort=False: just for speed (might also need to implement more comparison methods on `MincAtom`s)
        .aggregate({'resampled_mask' : lambda masks: list(masks)})
        .rename(columns={"resampled_mask" : "resampled_masks"})
        .assign(voted_mask=lambda df: df.apply(axis=1,
                                               func=lambda row:
                                                 s.defer(voxel_vote(label_files=row.resampled_masks,
                                                                    name="%s_voted_mask" % row.img.filename_wo_ext,
                                                                    # FIXME create and pass a mincatom cloned from row.img instead?
                                                                    output_dir=os.path.join(row.img.pipeline_sub_dir,
                                                                                            row.img.output_sub_dir,
                                                                                            "tmp")))))
        .apply(axis=1, func=lambda row: row.img._replace(mask=row.voted_mask)))
        #.assign(img=lambda df: df.apply(axis=1,
        #                                func=lambda row:
        #                                       row.img._replace(mask=row.voted_mask))))

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

    # replace the table of alignments with a new one with masked images
    #masking_alignments = (pd.merge(left=masking_alignments.assign(unmasked_img=lambda df: df.img),
    #                               right=masked_img,
    #                               on=["img"], how="right", sort=False)
    #                      .assign(img=lambda df: df.masked_img))

    # put the output_sub_dirs of the images (but not masks?  TODO) back to their original locations ...
    #masked_img.apply(lambda x: x._replace(output_sub_dir=original_imgs.ix[x.path].output_sub_dir))

    for img in masked_img:
        img.output_sub_dir = original_imgs.ix[img.path].output_sub_dir

    return Result(stages=s, output=masked_img)


# TODO make a non-destructive version of this that creates a new options object ... it should take an overall options
# object, copy it, and put the maget options at top level.
def fixup_maget_options(lsq12_options, nlin_options, maget_options):

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
            raise ValueError("I'd use the MAGeT nlin protocol for masking as well but different programs are specified")


def atlases_from_dir(atlas_lib : str, max_templates : int, pipeline_sub_dir : str):

    atlas_library = read_atlas_dir(atlas_lib=atlas_lib, pipeline_sub_dir=pipeline_sub_dir)

    if len(atlas_library) == 0:
        raise ValueError("No atlases found in specified directory '%s' ..." % atlas_lib)

    num_atlases_needed = min(max_templates, len(atlas_library))

    # TODO issue a warning if not all atlases used or if more atlases requested than available?
    # TODO also, doesn't slicing with a higher number (i.e., if max_templates > n) go to the end of the list anyway?
    # TODO arbitrary; could choose atlases better ...
    return atlas_library[:num_atlases_needed]


# TODO support LSQ6 registrations??
def maget(imgs : List[MincAtom], options, prefix, output_dir):     # FIXME prefix, output_dir aren't used !!

    s = Stages()

    maget_options = options.maget.maget

    resolution = options.registration.resolution  # TODO or get_resolution_from_file(...) -- only if file always exists!

    pipeline_sub_dir = os.path.join(options.application.output_directory,
                                    options.application.pipeline_name + "_atlases")

    if maget_options.atlas_lib is None:
        raise ValueError("Need some atlases ...")

    # TODO should alternately accept a CSV file ...
    atlases = atlases_from_dir(atlas_lib=maget_options.atlas_lib,
                               max_templates=maget_options.max_templates,
                               pipeline_sub_dir=pipeline_sub_dir)

    lsq12_conf = get_linear_configuration_from_options(options.maget.lsq12,
                                                       transform_type=LinearTransType.lsq12,
                                                       file_resolution=resolution)

    nlin_hierarchy = get_nonlinear_configuration_from_options(options.maget.nlin.nlin_protocol,
                                                              reg_method=options.maget.nlin.reg_method,
                                                              file_resolution=resolution)

    resample  = np.vectorize(mincresample_new, excluded={"extra_flags"})
    defer     = np.vectorize(s.defer)

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

        if maget_options.pairwise:

            def choose_new_templates(ts, n):
                # currently silly, but we might implement a smarter method ...
                # FIXME what if there aren't enough other imgs around?!  This silently goes weird ...
                return ts[:n+1]  # n+1 instead of n: choose one more since we won't use image as its own template ...

            new_templates = choose_new_templates(ts=imgs, n=maget_options.max_templates)
            # note these images are the masked ones if masking was done ...

            # TODO write a function to do these alignments and the image->atlas one above
            # align the new templates chosen from the images to the initial atlases:
            new_template_to_atlas_alignments = (
                pd.DataFrame({ 'img'   : template,
                               'atlas' : atlas,
                               'xfm'   : s.defer(lsq12_nlin(source=template, target=atlas,
                                                            lsq12_conf=lsq12_conf,
                                                            nlin_conf=nlin_hierarchy,
                                                            resample_source=False))}
                             for template in new_templates for atlas in atlases))
                             # ... and these atlases are multiplied by their masks (but is this necessary?)

            # label the new templates from resampling the atlas labels onto them:
            # TODO now vote on the labels to be used for the new templates ...
            # TODO extract into procedure?
            new_templates_labelled = (
                new_template_to_atlas_alignments
                .assign(resampled_labels=lambda df: defer(
                                               resample(img=df.atlas.apply(lambda x: x.labels),
                                                                      xfm=df.xfm.apply(lambda x: x.xfm),
                                                                      interpolation=Interpolation.nearest_neighbour,
                                                                      extra_flags=("-keep_real_range",),
                                                                      like=df.img, invert=True)))
                .groupby('img', sort=False, as_index=False)
                .aggregate({'resampled_labels' : lambda labels: list(labels)})
                .assign(voted_labels=lambda df: df.apply(axis=1,
                                                         func=lambda row:
                                                           s.defer(voxel_vote(label_files=row.resampled_labels,
                                                                              name="%s_template_labels" %
                                                                                   row.img.filename_wo_ext,
                                                                              output_dir=os.path.join(
                                                                                  row.img.pipeline_sub_dir,
                                                                                  row.img.output_sub_dir,
                                                                                  "labels"))))))

            # TODO write a procedure for this assign-groupby-aggregate-rename...
            # FIXME should be in above algebraic manipulation but MincAtoms don't support flexible immutable updating
            for row in pd.merge(left=new_template_to_atlas_alignments, right=new_templates_labelled,
                                on=["img"], how="right", sort=False).itertuples():
                row.img.labels = s.defer(mincresample_new(img=row.voted_labels, xfm=row.xfm.xfm, like=row.img,
                                                          invert=True, interpolation=Interpolation.nearest_neighbour,
                                                          #postfix="-input-labels",
                                                          # this makes names really long ...:
                                                          # TODO this doesn't work for running MAGeT on the nlin avg:
                                                          #new_name_wo_ext="%s_on_%s" %
                                                          #                (row.voted_labels.filename_wo_ext,
                                                          #                 row.img.filename_wo_ext),
                                                          #postfix="_labels_via_%s" % row.xfm.xfm.filename_wo_ext,
                                                          new_name_wo_ext="%s_via_%s" % (row.voted_labels.filename_wo_ext,
                                                                                         row.xfm.xfm.filename_wo_ext),
                                                          extra_flags=("-keep_real_range",)))

            # now that the new templates have been labelled, combine with the atlases:
            # FIXME use the masked atlases created earlier ??
            all_templates = pd.concat([new_templates_labelled.img, atlases], ignore_index=True)

            # now take union of the resampled labels from the new templates with labels from the original atlases:
            #all_alignments = pd.concat([image_to_template_alignments,
            #                            alignments.rename(columns={ "atlas" : "template" })],
            #                           ignore_index=True, join="inner")

        else:
            all_templates = atlases

        # now register each input to each selected template
        # N.B.: Even though we've already registered each image to each initial atlas, this happens again here,
        #       but using `nlin_hierarchy` instead of `masking_nlin_hierarchy` as options.
        #       This is not 'work-efficient' in the sense that this computation happens twice (although
        #       hopefully at greater precision the second time!), but the idea is to run a coarse initial
        #       registration to get a mask and then do a better registration with that mask (though I'm not
        #       sure exactly when this is faster than doing a single registration).
        #       This _can_ allow the overall computation to finish more rapidly
        #       (depending on the relative speed of the two alignment methods/parameters,
        #       number of atlases and other templates used, number of cores available, etc.).
        image_to_template_alignments = (
            pd.DataFrame({ "img"      : img,
                           "template" : template_img,
                           "xfm"      : xfm }
                         for img in imgs      # TODO use the masked imgs here?
                         for template_img in
                             all_templates
                             # FIXME delete this one alignment
                             #labelled_templates[labelled_templates.img != img]
                             # since equality is equality of filepaths (a bit dangerous)
                             # TODO is there a more direct/faster way just to delete the template?
                         for xfm in [s.defer(lsq12_nlin(source=img, target=template_img,
                                                        lsq12_conf=lsq12_conf,
                                                        nlin_conf=nlin_hierarchy))]
                         )
        )

        # now do a voxel_vote on all resampled template labels, just as earlier with the masks
        voted = (image_to_template_alignments
                 .assign(resampled_labels=lambda df:
                                            defer(resample(img=df.template.apply(lambda x: x.labels),
                                                           # FIXME bug: at this point templates from template_alignments
                                                           # don't have associated labels (i.e., `None`s) -- fatal
                                                           xfm=df.xfm.apply(lambda x: x.xfm),
                                                           interpolation=Interpolation.nearest_neighbour,
                                                           extra_flags=("-keep_real_range",),
                                                           like=df.img, invert=True)))
                 .groupby('img', sort=False)
                 # TODO the pattern groupby-aggregate(lambda x: list(x))-reset_index-assign is basically a hack
                 # to do a groupby-assign with access to the group name;
                 # see http://stackoverflow.com/a/30224447/849272 for a better solution
                 # (note this pattern occurs several times in MAGeT and two-level code)
                 .aggregate({'resampled_labels' : lambda labels: list(labels)})
                 .reset_index()
                 .assign(voted_labels=lambda df: defer(np.vectorize(voxel_vote)(label_files=df.resampled_labels,
                                                                                output_dir=df.img.apply(
                                                                                    lambda x: os.path.join(
                                                                                        x.pipeline_sub_dir,
                                                                                        x.output_sub_dir))))))

        # TODO doing mincresample -invert separately for the img->atlas xfm for mask, labels is silly
        # (when Pydpiper's `mincresample` does both automatically)?

        # blargh, another destructive update ...
        for row in voted.itertuples():
            row.img.labels = row.voted_labels

        # returning voted_labels as a column is slightly redundant, but possibly useful ...
        return Result(stages=s, output=voted)  # voted.drop("voted_labels", axis=1))


def maget_pipeline(options):

    check_MINC_input_files(options.application.files)

    imgs = pd.Series({ name : MincAtom(name,
                                       pipeline_sub_dir=os.path.join(options.application.output_directory,
                                                                     options.application.pipeline_name + "_processed"))
                       for name in options.application.files })

    # TODO fixup masking protocols ...

    return maget(imgs=imgs, options=options,
                 prefix=options.application.pipeline_name,
                 output_dir=options.application.output_directory)


def _mk_maget_parser(parser : ArgParser):
    group = parser.add_argument_group("MAGeT options", "Options for running MAGeT.")
    group.add_argument("--atlas-library", dest="atlas_lib",  # can't make required=True since may not be using MAGeT :|
                       type=str,                             # TODO: check existence of this dir?
                       help="Directory of existing atlas/label pairs")
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
                       help="Specify whether to use minctracc or mincANTS for masking. [Default = %(default)s].")
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
