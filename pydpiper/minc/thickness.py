import warnings
from argparse import Namespace
from typing import Optional, List

import pandas as pd
from configargparse import ArgParser

from pydpiper.core.arguments import AnnotatedParser, BaseParser
from pydpiper.core.util import AutoEnum
from pydpiper.minc.registration import lsq12_nlin, concat_xfmhandlers, mincblur, mincresample, mincresample_new, \
    mincreshape, minc_label_ops, LabelOp, optional
from pydpiper.core.files import FileAtom
from pydpiper.core.stages import Stages, CmdStage, Result
#from pydpiper.core.util   import NamedTuple
#from pydpiper.minc.containers import XfmHandler
#from pydpiper.minc.registration import concat_xfmhandlers, invert_xfmhandler, mincmath
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.files import MincAtom, XfmAtom


class Smoothing(AutoEnum):
    sinc = laplace = none = ()


def decimate(in_obj : FileAtom,
             reduction : float,
             smoothing_method     : Optional[Smoothing] = None,
             smoothing_iterations : Optional[int] = None):

    decimated = in_obj.newname_with_suffix("_decimated_%s" % reduction
                                           + ("_smooth" if smoothing_method not in (Smoothing.none, None) else ""))

    stage = CmdStage(inputs=(in_obj,), outputs=(decimated,),
                     cmd=["decimate.py"]
                         + (["--smoothing-method", smoothing_method.name]
                            if smoothing_method not in ("none", None) else [])
                         + (["--smoothing-iterations", str(smoothing_iterations)]
                            if smoothing_iterations is not None else [])
                         + [str(reduction), in_obj.path, decimated.path])

    return Result(stages=Stages([stage]), output=decimated)


def diffuse(obj_file      : FileAtom,
            input_signal  : FileAtom,
            #output_signal : FileAtom,
            kernel        : Optional[float] = None,
            iterations    : Optional[int]   = None,
            parametric    : Optional[int]   = None):
    output_signal = input_signal.newname_with_suffix("_thickness")
    stage = CmdStage(inputs=(obj_file, input_signal), outputs=(output_signal,),
                     cmd=["diffuse"]  # TODO make an abstraction for this; see Nix stdlib's `optional`
                         + ["-kernel", str(kernel)] if kernel is not None else []
                         + ["-iterations", iterations] if iterations is not None else []
                         + ["-parametric", parametric] if parametric is not None else []
                         + [obj_file, input_signal, output_signal])
    return Result(stages=Stages([stage]), output=output_signal)


# def laplacian_thickness(surface        : FileAtom,
#                         args           : List[str],
#                         grey_surface   : Optional[FileAtom] = None,
#                         white_surface  : Optional[FileAtom] = None,
#                         grid           : Optional[MincAtom] = None,
#                         sample         : Optional[MincAtom] = None,
#                         max_iterations : Optional[int] = None):
#     if not xor(grid is not None, grey_surface is not None and white_surface is not None):
#         raise ValueError("need a grid or two surfaces")
#     thickness = NotImplemented
#     stage = CmdStage(inputs=(grey_surface, white_surface)
#                             + ((sample,) if sample else ())
#                             + ((surface,) if surface else ()),
#                      outputs=(thickness,),
#                      cmd=["laplacian_thickness"]
#                          + (["-max_iterations", str(max_iterations)] if max_iterations is not None else [])
#                          + (["-like", sample.path] if sample is not None else [])
#                          + (["-from_grid", grid] if grid else [grey_surface.path, white_surface.path])
#                          + [surface.path]
#                          + [thickness.path])
#     return Result(stages=Stages([stage]), output=thickness)


class Side(AutoEnum):
    left = right = ()


def make_laplace_grid(input_labels   : FileAtom,
                      label_mapping  : FileAtom,
                      binary_closing : bool = None,
                      side           : Optional[Side] = None):
    out_grid = input_labels.newname_with_suffix("_laplace_grid" + ("_%s" % side.name if side else ""))

    s = CmdStage(inputs=(input_labels, label_mapping), outputs=(out_grid,),
                 cmd=["make_laplace_grid", "--clobber"]
                     + optional(binary_closing, "--binary_closing")
                     + optional(side, "--%s" % side.name)
                     + [input_labels.path, label_mapping.path, out_grid.path])

    return Result(stages=Stages([s]), output=out_grid)


# TODO move to util
def xor(x, y):
    """N.B.: not lazy like `or` ..."""
    return bool(x) != bool(y)


def marching_cubes(in_volume : MincAtom,
                   min_threshold : float = None,
                   max_threshold : float = None,
                   threshold     : float = None):

    if not xor(threshold is None, min_threshold is None and max_threshold is None):
        raise ValueError("specify either threshold or min and max thresholds")

    out_volume = FileAtom(in_volume.newname_with(NotImplemented))  # forget MINCy fields  # FIXME this coercion doesn't work

    stage = CmdStage(inputs=(in_volume,), outputs=(out_volume,),
                     cmd=["marching_cubes", in_volume.path, out_volume.path]
                         + ([str(threshold)] if threshold is not None else [str(min_threshold), str(max_threshold)]))

    return Result(stages=Stages([stage]), output=out_volume)


def minclaplace(input_grid        : MincAtom,
                extra_args        : List[str] = [],
                solution_vertices : Optional[FileAtom] = None,
                create_surface    : bool = True,) -> Result[FileAtom]:
    # TODO the ambiguity of the return type is slightly annoying ...
    # best to create separate minclaplace_at_vertices for the case when `--solve-at-vertices` is used?
    solved = input_grid.newname_with_suffix("_solved", ext=".txt" if solution_vertices else ".mnc")
    if create_surface:
        out_surface = input_grid.newname_with_suffix("_surface", ext=".obj")
    stage = CmdStage(inputs=(input_grid,), outputs=(solved,) + ((out_surface,) if create_surface else ()),
                     cmd=["minclaplace"]
                         + (["--solve-at-vertices=%s" % solution_vertices.path]
                            if solution_vertices is not None else [])
                         + (["--create-surface=%s" % out_surface.path] if create_surface else [])
                         + extra_args
                         + [input_grid.path, solved.path])

    return Result(stages=Stages([stage]),
                  output=Namespace(solved=solved, surface=out_surface)
                         if create_surface else Namespace(solved=solved))


def surface_mask2(input   : MincAtom,
                  surface : FileAtom,
                  args    : List[str] = []) -> Result[MincAtom]:
    mask_vol = surface.newname_with_suffix("_mask", ext=".mnc")
    stage = CmdStage(inputs=(input, surface),
                     outputs=(mask_vol,),
                     cmd=["surface_mask2", "-clobber"] + args + [input.path, surface.path, mask_vol.path])
    return Result(stages=Stages([stage]), output=mask_vol)


def reconstitute_laplacian_grid(cortex  : MincAtom,
                                grid    : MincAtom,
                                midline : MincAtom) -> Result[MincAtom]:
    output_grid = grid.newname_with_suffix("_reconstituted")
    stage = CmdStage(inputs=(cortex, grid, midline), outputs=(output_grid,),
                     cmd=["reconstitute_laplacian_grid", midline.path, cortex.path, grid.path, output_grid.path])
    return Result(stages=Stages([stage]), output=output_grid)


def transform_objects(input_obj : FileAtom,
                      xfm       : XfmAtom) -> Result[FileAtom]:  # XfmAtom -> XfmHandler??
    output_obj = input_obj.newname_with_suffix("_resampled_via_%s" % xfm.filename_wo_ext)
    stage = CmdStage(inputs=(input_obj, xfm), outputs=(output_obj,),
                     cmd=["transform_objects", input_obj.path, xfm.path, output_obj.path])
    return Result(stages=Stages([stage]), output=output_obj)


def cortical_thickness(xfms   : pd.Series,  # nlin avg -> subject XfmHandler (iirc)...
                       atlas  : MincAtom,   # nlin avg
                       label_mapping : FileAtom,
                       atlas_fwhm : float,
                       thickness_fwhm : float):

    try:
        import vtk
    except:
        warnings.warn("couldn't `import vtk`, without which `decimate.py` is unable to run ...")
        raise

    s = Stages()

    # generate thickness maps for the average:
    left_grid, right_grid = [s.defer(make_laplace_grid(input_labels=atlas.labels,
                                                       label_mapping=label_mapping,
                                                       binary_closing=True, side=side))
                             for side in (Side.left, Side.right)]

    atlas_left_thickness, atlas_right_thickness = (
        [s.defer(decimate(
            s.defer(minclaplace(input_grid=grid,
                                extra_args=["--create-surface-range", "0", "10"])).surface,  # enclose entire cortex
            reduction=0.8,  # FIXME: magic number ... implement a way to specify number rather than fraction instead?
            smoothing_method=Smoothing.laplace))
         for grid in (left_grid, right_grid)])

    # as per comment in MICe_thickness, blur atlas instead of transformed object files ... ?
    # (maybe this workaround is now obsolete)
    blurred_atlas = s.defer(mincblur(img=atlas, fwhm=atlas_fwhm)).img

    # TODO rename this dataframe
    resampled = (pd.DataFrame(
      {
        'xfm' : xfms,
        # resample the atlas files to each subject:
        'blurred_atlas_grid_resampled'  :
            xfms.apply(lambda xfm: s.defer(mincresample_new(img=blurred_atlas, xfm=xfm.xfm, like=xfm.target))),
        'atlas_left_resampled'    :
            xfms.apply(lambda xfm: s.defer(transform_objects(input_obj=atlas_left_thickness, xfm=xfm.xfm))),
        'atlas_right_resampled'   :
            xfms.apply(lambda xfm: s.defer(transform_objects(input_obj=atlas_right_thickness, xfm=xfm.xfm))),
      })
        .assign(left_grid=lambda df: df.xfm.map(lambda xfm: s.defer(
                    make_laplace_grid(input_labels=xfm.target,
                                      label_mapping=label_mapping,
                                      binary_closing=True,
                                      side=Side.left))),
                right_grid=lambda df: df.xfm.map(lambda xfm: s.defer(
                    make_laplace_grid(input_labels=xfm.target,
                                      label_mapping=label_mapping,
                                      binary_closing=True,
                                      side=Side.right))))
        .assign(left_thickness=lambda df: df.apply(axis=1, func=lambda row:
                  s.defer(minclaplace(input_grid=row.left_grid,
                                      solution_vertices=row.atlas_left_resampled))),
                right_thickness=lambda df: df.apply(axis=1, func=lambda row:
                  s.defer(minclaplace(input_grid=row.right_grid,
                                      solution_vertices=row.atlas_right_resampled))))
        .assign(smooth_left_fwhm=lambda df: df.apply(axis=1, func=lambda row:
                  s.defer(diffuse(obj_file=row.atlas_left_resampled,
                                  input_signal=row.left_thickness.solved,
                                  kernel=thickness_fwhm,
                                  iterations=1000))),
                smooth_right_fwhm=lambda df: df.apply(axis=1, func=lambda row:
                  s.defer(diffuse(obj_file=row.atlas_right_resampled,
                                  input_signal=row.right_thickness.solved,
                                  kernel=thickness_fwhm,
                                  iterations=1000)))))
    return Result(stages=s, output=resampled)


def _mk_thickness_parser(parser : ArgParser):
    group = parser.add_argument_group("Thickness", "Thickness calculation options.")
    group.add_argument("--run-thickness", action='store_true', dest="run_thickness",
                       help="Run thickness computation.")
    group.add_argument("--no-run-thickness", action='store_false', dest="run_thickness",
                       help="Don't run thickness computation.")
    parser.set_defaults(run_thickness=True)
    group.add_argument("--label-mapping", type=str, dest="label_mapping",
                       help="path to CSV file mapping; see minclaplace/wiki/LaplaceGrid")
    group.add_argument("--atlas-fwhm", dest="atlas_fwhm", type=float, # default ?!
                       help="Blurring kernel (mm) for atlas")
    group.add_argument("--thickness-fwhm", dest="thickness_fwhm", type=float, # default??
                       help="Blurring kernel (mm) for cortical surfaces")
    return parser


thickness_parser = AnnotatedParser(parser=BaseParser(_mk_thickness_parser(ArgParser(add_help=False)),
                                                     "thickness"),
                                   namespace="thickness")