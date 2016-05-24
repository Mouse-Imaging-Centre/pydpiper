#!/usr/bin/env python3

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


def read_atlas_dir(atlas_lib : str) -> List[MincAtom]:
    """Read in atlases from directory specified return them as processed MincAtoms (with masks and labels).
    Assumes atlas/label/mask groups have one of the following naming schemes:
    (1) name_average.mnc, name_labels.mnc, name_mask.mnc
    (2) name.mnc, name_labels.mnc, name_mask.mnc
    Note that all three files are required even if masking is not used."""

    return process_atlas_files(glob.iglob(os.path.join(atlas_lib, "*.mnc")))


def process_atlas_files(filenames : List[str]) -> List[MincAtom]:

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
                                                  mask=MincAtom(name=mask_file),
                                                  labels=MincAtom(name=label_file))

    return pd.Series(list(grouped_atlas_files.values()))

# TODO support LSQ6 registrations??
def maget(imgs : List[MincAtom], options, prefix, output_dir):

    s = Stages()

    maget_options = options.maget.maget

    #atlas_dir = os.path.join(output_dir, "input_atlases") ???

    # TODO should alternately accept a CSV file ...
    atlas_library = read_atlas_dir(atlas_lib=maget_options.atlas_lib)

    if len(atlas_library) == 0:
        raise ValueError("No atlases found in specified directory '%s' ..." % options.maget.maget.atlas_lib)

    lsq12_conf = get_linear_configuration_from_options(options.maget.lsq12,
                                                       LinearTransType.lsq12,
                                                       options.registration.resolution)

    masking_nlin_hierarchy = get_nonlinear_configuration_from_options(options.maget.maget.nlin_protocol,
                                                                      options.maget.maget.mask_method,
                                                                      options.registration.resolution)

    nlin_hierarchy = get_nonlinear_configuration_from_options(options.maget.nlin.nlin_protocol,
                                                              options.maget.nlin.reg_method,
                                                              options.registration.resolution)

    num_atlases_needed = min(maget_options.max_templates, len(atlas_library))
    # TODO arbitrary; could choose atlases better ...
    atlases = atlas_library[:num_atlases_needed]
    # TODO issue a warning if not all atlases used?
    # TODO also, doesn't slicing with a higher number (i.e., if max_templates > n) go to the end of the list anyway?

    resample  = np.vectorize(mincresample_new, excluded={"extra_flags"})
    defer     = np.vectorize(s.defer)

    # plan the basic registrations between all image-atlas pairs; store the result paths in a table
    masking_alignments = pd.DataFrame({ 'img'   : img,
                                        'atlas' : atlas,
                                        'xfm'   : s.defer(lsq12_nlin(source=img, target=atlas,
                                                                     lsq12_conf=lsq12_conf,
                                                                     nlin_conf=masking_nlin_hierarchy,
                                                                     resample_source=False))}
                                      for img in imgs for atlas in atlases)

    if maget_options.mask or maget_options.mask_only:
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
            # sort=False: for speed, and since we haven't implemented comparisons on `MincAtom`s
            #.apply(lambda row: s.defer(voxel_vote(label_files=rowmask)))
            .aggregate({'resampled_mask' : lambda masks: list(masks)})
                                         #  lambda masks: s.defer(voxel_vote(label_files=masks,
                                         #                                   output_dir="TODO", name="voted_mask")) })  #,
                        #'img' : lambda imgs: imgs[0]})
            .rename(columns={"resampled_mask" : "resampled_masks"})
            # NB aggregate can keep the other columns around if a reduction operation is supplied for them ...?
            #.reset_index())
            #.apply(lambda df: s.defer(voxel_vote(label_files=df.resampled_masks,
            #                                     name="voted_mask",
            #                                     output_dir=df.img.pipeline_sub_dir)),
            #       axis=1)
            .assign(voted_mask=lambda df: df.apply(axis=1,
                                                   func=lambda row:
                                                     s.defer(voxel_vote(label_files=row.resampled_masks,
                                                                        name="voted_mask",
                                                                        output_dir="TODO"))))
        #    # TODO drop resampled_masks col
        #    #.rename(columns={"resampled_mask" : "voted_mask"}))

            .assign(masked_img=lambda df:
                                  df.apply(axis=1,
                                           func=lambda row:
                                                  s.defer(mincmath(op="mult",
                                                                   # img must precede mask here
                                                                   # for output image range to be correct:
                                                                   vols=[row.img, row.voted_mask],
                                                                   new_name="%s_masked" % row.img.filename_wo_ext,
                                                                   subdir="resampled")))))  #['img']

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

        # apply atlases' masks:
        #for atlas in atlases:
        #    # again, img must precede mask here for correct output image range:
        #    s.defer(mincmath(op='mult', vols=[atlas, atlas.mask], subdir="resampled",
        #                     new_name="%s_masked" % atlas.filename_wo_ext))
        masked_atlases = atlases.apply(lambda atlas:
                           s.defer(mincmath(op='mult', vols=[atlas, atlas.mask], subdir="resampled",
                                            new_name="%s_masked" % atlas.filename_wo_ext)))

        # replace the table of alignments with a new one with masked images
        masking_alignments = (pd.merge(left=masking_alignments.assign(unmasked_img=lambda df: df.img),
                                       right=masked_img,
                                       on=["img"], how="right", sort=False)
                                .assign(img=lambda df: df.masked_img))

        # now propagate only the masked form of the images and atlases:
        imgs    = masking_alignments.img
        atlases = masked_atlases  # TODO is this needed?

    if maget_options.mask_only:
        # register each input to each atlas, creating a mask
        return Result(stages=s, output=masking_alignments)   # TODO rename `alignments` to `registrations`??
    else:
        del masking_alignments
        # this `del` is just to verify that we don't accidentally use this later, since my intent is that these
        # coarser alignments shouldn't be re-used, just the masked images they create; can be removed later
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
            # TODO extract into procedure
            new_templates_labelled = (
                new_template_to_atlas_alignments
                .assign(resampled_labels=lambda df: defer(
                                               resample(img=df.atlas.apply(lambda x: x.labels),
                                                                      xfm=df.xfm.apply(lambda x: x.xfm),
                                                                      interpolation=Interpolation.nearest_neighbour,
                                                                      extra_flags=("-keep_real_range",),
                                                                      like=df.img, invert=True)))
                .groupby('img', sort=False)
                .aggregate({'resampled_labels' : lambda labels:
                               s.defer(voxel_vote(label_files=labels,
                                                  output_dir="TODO"))})
                .rename(columns={"resampled_labels" : "voted_labels"})
                .reset_index())

            #
            # TODO write a procedure for this ...
            # FIXME should be in above algebraic manipulation but MincAtoms don't support flexible immutable updating
            for row in pd.merge(left=new_template_to_atlas_alignments, right=new_templates_labelled,
                                on=["img"], how="right", sort=False).itertuples():
                row.img.labels = s.defer(mincresample_new(img=row.voted_labels, xfm=row.xfm.xfm, like=row.img,
                                                          invert=True, interpolation=Interpolation.nearest_neighbour,
                                                          postfix="-input-labels",
                                                          #new_name_wo_ext="%s_to_%s-input-labels" %
                                                          #                (row.img.filename_wo_ext,
                                                          #                 row.voted_labels.filename_wo_ext),
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
        #       number of atlases and other templates used, number of cores available, etc.)
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

            ### propagate labels from templates to images
            ###all_alignments = alignments.rename(columns={ "atlas" : "template" })

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
                 # old stuff that didn't seem to give access to img inside `voxel_voxel` call:
                 #.aggregate({'resampled_labels' : lambda labels: s.defer(voxel_vote(label_files=labels,
                 #                                                                  output_dir="TODO"))})
                 #.rename(columns={"resampled_labels" : "voted_labels"})
                 #.reset_index())
                 .aggregate({'resampled_labels' : lambda labels: list(labels)})
                 .reset_index()
                 .assign(voted_labels=lambda df: defer(np.vectorize(voxel_vote)(label_files=df.resampled_labels,
                                                                                output_dir=df.img.apply(
                                                                                    lambda x: x.pipeline_sub_dir)))))

        # TODO doing mincresample -invert separately for the img->atlas xfm for mask, labels is silly
        # (when Pydpiper's `mincresample` does both automatically)?

        # blargh, another destructive update ...
        for row in voted.itertuples():
            row.img.labels = row.voted_labels

        return Result(stages=s, output=voted.drop("voted_labels", axis=1))


def maget_pipeline(options):

    check_MINC_input_files(options.application.files)

    imgs = pd.Series({ name : MincAtom(name,
                                       pipeline_sub_dir=os.path.join(options.application.output_directory,
                                                                     options.application.pipeline_name + "_processed"))
                       for name in options.application.files })

    return maget(imgs=imgs, options=options,
                 prefix=options.application.pipeline_name,
                 output_dir=options.application.output_directory)


def _mk_maget_parser(parser : ArgParser):
    group = parser.add_argument_group("MAGeT options", "Options for running MAGeT.")
    group.add_argument("--atlas-library", dest="atlas_lib",
                       type=str, default="atlas_label_pairs",
                       help="Directory of existing atlas/label pairs")
    group.add_argument("--pairwise", dest="pairwise",
                       action="store_true",
                       help="""If specified, register inputs to each other pairwise. [Default]""")
    group.add_argument("--no-pairwise", dest="pairwise",
                       action="store_false",
                       help="""If specified, only register inputs to atlases in library.""")
    parser.set_defaults(pairwise=True)
    group.add_argument("--mask", dest="mask",
                       action="store_true", default=False,
                       help="Create a mask for all images prior to handling labels. [Default = %(default)s]")
    group.add_argument("--mask-only", dest="mask_only",
                       action="store_true", default=False,
                       help="Create a mask for all images only, do not run full algorithm. [Default = %(default)s]")
    group.add_argument("--max-templates", dest="max_templates",
                       default=25, type=int,
                       help="Maximum number of templates to generate. [Default = %(default)s]")
    group.add_argument("--masking-method", dest="mask_method",
                       default="minctracc", type=str,
                       help="Specify whether to use minctracc or mincANTS for masking. [Default = %(default)s].")
    group.add_argument("--masking-nlin-protocol", dest="nlin_protocol",  # TODO basically copy/pasted from nlin parser
                       type=str, default=None,
                       help="Can optionally specify a registration protocol that is different from defaults. "
                            "Parameters must be specified as in either or the following examples: \n"
                            "applications_testing/test_data/minctracc_example_nlin_protocol.csv \n"
                            "applications_testing/test_data/mincANTS_example_nlin_protocol.csv \n"
                            "[Default = %(default)s]")
    return parser
    # maybe wire the masking-nlin-protocol to the nlin-protocol?

maget_parser = AnnotatedParser(parser=BaseParser(_mk_maget_parser(ArgParser(add_help=False)),
                                                 "maget"),
                               namespace="maget")

maget_application = mk_application(parsers=[AnnotatedParser(parser=CompoundParser([#lsq6_parser,
                                                                                   lsq12_parser,
                                                                                   nlin_parser, stats_parser,
                                                                                   maget_parser]),
                                                            namespace="maget")],
                                   pipeline=maget_pipeline)

if __name__ == "__main__":
    maget_application()
