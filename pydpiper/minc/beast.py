import os

from configargparse import ArgParser
import pandas as pd
from typing import Optional, List, Tuple
from pyminc.volumes.factory import volumeFromFile

from pydpiper.core.arguments import AnnotatedParser, BaseParser
from pydpiper.core.stages import CmdStage, Stages, Result
from pydpiper.core.util import NamedTuple, AutoEnum, flatten
from pydpiper.minc.files import MincAtom, XfmAtom
from pydpiper.minc.registration import (MultilevelMinctraccConf, multilevel_minctracc, safe_minc_modify_header,
                                        xfmaverage, mincresample_new)


class BeastFilter(AutoEnum):
    median = nlm_filter = ()

# MincBeastConf = NamedTuple("MincBeastConf",
#                            [("probability", Optional[bool]),
#                             ("flip", Optional[bool]),
#                             ("load_moments", Optional[bool]),
#                             ("fill", Optional[bool]),
#                             ("filter", Optional[BeastFilter]),
#                             ("abspath", Optional[bool]),
#                             ("verbose", Optional[bool]),
#                             ("patch_size", Optional[int]),
#                             ("search_area", Optional[int]),
#                             ("alpha", Optional[float]),
#                             ("beta", Optional[float]),
#                             ("threshold", Optional[float]),
#                             ("selection_num", Optional[int]),
#                             ("positive", Optional[str]),
#                             ("output_selection", Optional[str]),
#                             ("count", Optional[str]),
#                             ("configuration", Optional[str]),
#                             ("mask", Optional[str]),
#                             ("same_resolution", Optional[bool]),
#                             ("no_mask", Optional[bool]),
#                             ("no_positive", Optional[bool])])

def mincbeast(img : MincAtom,
              library_dir : str,
              #conf : MincBeastConf,
              extra_flags : Tuple[str],
              segmentation_name=None,
              subdir=None):
    """Run mincbeast on the input file after an LSQ12 alignment, resampling, etc.  Note we assume the mincbeast
    library files have already been generated prior to the start of the pipeline and so these aren't tracked in the
    input/output files of the Pydpiper `CmdStage`."""
    segmentation = img.newname_with_suffix(segmentation_name if segmentation_name else "segmentation", subdir=subdir)

    s = CmdStage(inputs=(img,), outputs=(segmentation,),
                 cmd=["mincbeast", "-clobber"] + extra_flags + [library_dir, img.path, segmentation.path])

    # hack ... need to look at config file, # of voxels, ...
    s.when_runnable_hooks.append(lambda s: s.setMemory(20))

    return Result(stages=Stages([s]), output=segmentation)


BeastNormalizeConf = NamedTuple("BeastNormalizeConf",
                                [("noreg", bool),
                                 ("non3",  bool),
                                 ("inorm", Optional[int])])


# these will omit the optional flags (i.e., use beast_normalize's defaults); our default is really to noreg though
# since registration seems more likely to work at the native resolution using the user-supplied protocols
# rather than at 1mm using bestlinreg.pl's defaults
default_beast_normalize_conf = BeastNormalizeConf(noreg=False, non3=False, inorm=None)


def beast_normalize(img        : MincAtom,
                    model_path : str,  # could be MincAtom
                    outfile    : Optional[MincAtom] = None,  # just include in XfmHandler?
                    #outxfm     : Optional[XfmAtom],
                    conf       : BeastNormalizeConf = default_beast_normalize_conf,
                    subdir     : str = None,
                    init_xfm   : Optional[XfmAtom] = None):

    outfile = outfile or img.newname_with_suffix("normalized", subdir=subdir)

    outxfm  = img.newname_with_suffix("_")

    s = CmdStage(cmd=["beast_normalize"]
                     + ["-modeldir", os.path.dirname(model_path), "-modelname", os.path.basename(model_path)]
                     + (["-noreg"] if conf.noreg else [])
                     + (["-non3"] if conf.non3 else [])
                     + (["-intensity_norm", str(conf.inorm)] if conf.inorm is not None else [])
                     + [img.path, outfile.path, outxfm.path],
                 inputs=(img,),  # could include model:MincAtom if we wanted to create beast models in pydpiper
                 outputs=(outfile, outxfm))
    return Result(stages=Stages([s]), output=outxfm)


def beast_segment(imgs : List[MincAtom],
                  library_dir : str,
                  linear_conf : MultilevelMinctraccConf,
                  #beast_conf  : MincBeastConf,
                  model_path  : str,  # N.B.: currently the model should already have "1mm" resolution, sorry ...
                  #resolution : float,
                  pipeline_sub_dir : str,
                  beast_flags : Tuple[str] = (),
                  beast_normalize_conf : BeastNormalizeConf = default_beast_normalize_conf):
    """
    The library_dir should be configured for mincbeast with the addition of `library.stx.native`
    and `library.masks.native` files containing paths to native-resolution files related to the library*mm files
    as follows: using `minc_modify_header` to alter the header of a native-resolution image should produce
    the corresponding 1mm file (and similarly for the segmentations).
    """

    if library_dir is None:
        raise ValueError("mincbeast library dir not specified")
    if model_path is None:
        raise ValueError("beast_normalize model path not specified")

    s = Stages()

    atlas_paths = pd.read_csv(os.path.join(library_dir, "library.stx.native"),
                              names=["native_img"], header=None).native_img
    # in general the provided segmentations may not be whole-brains masks.  However, if not, we're only interested
    # in segmenting the given structure(s) anyway, so can use the segmentation as a mask, assuming integer values > 0
    # are treated as 1 by the registration algorithm (of course, the resulting image will be poorly registered
    # outside the given ROIs, so should be used carefully unless a whole-brain segmentation was specified):
    segmentation_paths = pd.read_csv(os.path.join(library_dir, "library.masks.native"),
                                     names=["native_segmentation"], header=None).native_segmentation

    atlases = pd.Series([MincAtom(name=os.path.join(library_dir, img),
                                  # alternate: use the overall union mask??
                                  mask=MincAtom(name=os.path.join(library_dir, segmentation),
                                                pipeline_sub_dir=pipeline_sub_dir),
                                  pipeline_sub_dir=pipeline_sub_dir)
                         for img, segmentation in zip(atlas_paths, segmentation_paths)])

    alignments = (pd.DataFrame({ 'img'   : img,
                                 'atlas' : atlas,
                                 'xfm'   : s.defer(multilevel_minctracc(source=img, target=atlas,
                                                                        conf=linear_conf))}
                               for img in imgs for atlas in atlases)
                  .groupby('img')
                  .aggregate({'xfm' : lambda xfms: s.defer(xfmaverage([xfm.xfm for xfm in xfms]))})
                  .reset_index()
                  .assign(resampled=lambda df: df.apply(axis=1, func=lambda row:
                            s.defer(mincresample_new(img=row.img, xfm=row.xfm, like=atlases[0]))))
                  .assign(modified=lambda df: df.apply(axis=1, func=lambda row:
                            s.defer(safe_minc_modify_header(infile=row.resampled,
                                                            subdir="tmp",
                                                            flags=["-dinsert", "xspace:step=1",
                                                                   "-dinsert", "yspace:step=1",
                                                                   "-dinsert", "zspace:step=1"]))))
                  .assign(normalized=lambda df: df.apply(axis=1, func=lambda row:
                            s.defer(beast_normalize(img=row.modified,
                                                    model_path=model_path,
                                                    conf=beast_normalize_conf.replace(noreg=True),
                                                    subdir="tmp"))))
                  .assign(segmented=lambda df: df.apply(axis=1, func=lambda row:
                            s.defer(mincbeast(img=row.normalized,
                                              library_dir=library_dir,
                                              extra_flags=beast_flags,
                                              subdir="tmp"))))
                  # just restoring the original header is fine since we didn't transform in the fake-resolution space:
                  # TODO what if the dimnames are weird/vectorial ?! hopefully things have already failed noisily ...
                  .assign(segmented=lambda df: df.apply(axis=1, func=lambda row: s.defer(
                            safe_minc_modify_header(
                                infile=row.segmented,
                                flags=flatten(*(["-dinsert", "%s:step=%s" % (dimname, step)]
                                                for dimname, step in
                                                  zip(volumeFromFile(row.img.path).dimnames,
                                                      volumeFromFile(row.img.path).separations))))))))

    return Result(stages=s, output=alignments.segmented)


def _mk_beast_parser(parser : ArgParser):
    group = parser.add_argument_group("BEaST options", "Options for running mincbeast.")
    group.add_argument("--library-dir", dest="library_dir",
                       type=str, help="Path to BEaST library dir, configured for mincbeast but with additional "
                                      "library.stx.native and library.masks.native files containing the template "
                                      "images and segmentations/masks at native resolution and such that "
                                      "'minc_modify_header -dinsert xspace:step=1 "
                                      "-dinsert yspace:step=1 -dinsert zspace:step=1' produces the 1mm images.")
    group.add_argument("--model-path", dest="model_path",
                       type=str, help="Path to '1mm' model (in mincbeast-compatible model dir).")
    group.add_argument("--beast-flags", dest="beast_flags", default="", help="Extra flags to pass to mincbeast")
    return parser


beast_parser = AnnotatedParser(parser=BaseParser(_mk_beast_parser(ArgParser(add_help=False)),
                                                 "beast"),
                               namespace="beast")

