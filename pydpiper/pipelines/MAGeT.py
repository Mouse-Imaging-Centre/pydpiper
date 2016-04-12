#!/usr/bin/env python3

import glob
from collections import defaultdict
from configargparse import ArgParser
import os
from typing import List
from pandas import DataFrame
import numpy as np

from pydpiper.core.arguments        import (lsq6_parser, lsq12_parser, nlin_parser, stats_parser,
                                            CompoundParser, AnnotatedParser, BaseParser)
from pydpiper.core.stages import Stages
from pydpiper.execution.application import mk_application
from pydpiper.minc.analysis import voxel_vote, mincmath
from pydpiper.minc.files            import MincAtom
from pydpiper.minc.registration     import (check_MINC_input_files, lsq12_nlin,
                                            get_nonlinear_configuration_from_options,
                                            get_linear_configuration_from_options, LinearTransType, mincresample)


def find_by(f, xs, on_empty=None):  # TODO move to util
    for x in xs:
        if f(x):
            return x
    if on_empty is not None:
        return on_empty()
    else:
        raise ValueError("Nothing in iterable satisfies the predicate")


def index_by(f, xs, on_empty=None):  # TODO move to util
    for ix, x in enumerate(xs):
        if f(x):
            return ix
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
        img_file   = files.get("_average.mnc") or files.get(".mnc")
        mask_file  = files.get("_mask.mnc")
        label_file = files.get("_labels.mnc")
        if len(files) > 3 or not (img_file and mask_file and label_file):
            raise ValueError("atlas filename conventions are wrong for atlas '%s'" % group)
        else:
            # TODO: screw around with filenames/directories as usual
            grouped_atlas_files[group] = MincAtom(name=img_file,
                                                  mask=MincAtom(name=mask_file),
                                                  labels=MincAtom(name=label_file))

    return list(grouped_atlas_files.values())

def maget(imgs : List[MincAtom], options, prefix, output_dir):

    s = Stages()

    maget_options = options.maget.maget

    atlas_dir = os.path.join(output_dir, "input_atlases")

    # TODO should alternately accept a CSV file ...
    atlas_library = read_atlas_dir(atlas_lib=maget_options.atlas_lib)

    if len(atlas_library) == 0:
        raise ValueError("No atlases found in specified directory '%s' ..." % options.maget.maget.atlas_lib)

    lsq12_conf = get_linear_configuration_from_options(options.maget.lsq12,
                                                       LinearTransType.lsq12,
                                                       options.registration.resolution)

    nlin_hierarchy = get_nonlinear_configuration_from_options(options.maget.nlin.nlin_protocol,
                                                              options.maget.nlin.reg_method,
                                                              options.registration.resolution)

    num_atlases_needed = min(maget_options.max_templates, len(atlas_library))
    atlases = atlas_library[:num_atlases_needed]   # TODO arbitrary; could choose better ... ?!

    # plan the basic registrations between all image-atlas pairs; store the result paths in a table
    alignments = DataFrame({ 'img'   : img,
                             'atlas' : atlas,
                             'xfm'   : s.defer(lsq12_nlin(source=img, target=atlas,
                                                          lsq12_conf=lsq12_conf,
                                                          nlin_conf=nlin_hierarchy))}
                           for img in imgs for atlas in atlases)

    resample  = np.vectorize(mincresample)
    defer     = np.vectorize(s.defer)
    # resample the atlas mask to the image:  # TODO only if options.mask ... ?
    # TODO is this better when transposed??

    # FIXME delete
    #def propagate_masks(img):
    #    relevant_xfms = alignments.groupby('img')
    #    raise NotImplementedError

    # FIXME get rid of this options.maget.maget stuff ...
    if maget_options.mask or maget_options.mask_only:
        voted = (alignments.assign(resampled_mask=lambda df: defer(resample(img=df.atlas.apply(lambda x: x.mask),
                                                                            xfm=df.xfm.apply(lambda x: x.xfm),
                                                                            like=df.img, invert=True)))
                 .groupby('img')  #, as_index=False)
                 .aggregate({'resampled_mask': lambda masks: s.defer(voxel_vote(masks))})  # TODO args to voxel_vote?
                 ) #.assign(masked_img=lambda df: defer(np.vectorize(mincmath)(op='mult', vols=[df.img, df.voted_mask]))))

        # propagate a mask to each image using the above `alignments` as follows:
        #imgs = [propagate_masks(img) for img in imgs]
        # FIXME mask the atlases/each atlas already has a mask?
        # - for each image, voxel_vote on the masks propagated to that image to get a suitable mask
        # - run mincmath -clobber -mult <img> <voted_mask> to apply the mask to the files
        raise NotImplementedError

    if maget_options.mask_only:
        # register each input to each atlas, creating a mask
        return ()   # FIXME actually return something
    else:
        if maget_options.pairwise:
            raise NotImplementedError
        else:
            for img in imgs:
                img.labels = s.defer(voxel_vote(label_files=NotImplemented))  # TODO mask=False (why?)
                # assume we have a voxel_vote that doesn't crash on a single input label file ...
        return imgs  # ??

def maget_pipeline(options):

    check_MINC_input_files(options.application.files)

    imgs = [MincAtom(name, pipeline_sub_dir=os.path.join(options.application.output_directory,
                                                         options.application.pipeline_name + "_processed"))
            for name in options.application.files]

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
    return parser


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
