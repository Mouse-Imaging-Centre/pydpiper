from argparse import Namespace
from typing import List, Optional, Tuple
import os
import pandas as pd

from pydpiper.core.stages import Stages, CmdStage, Result
from pydpiper.core.util   import NamedTuple
from pydpiper.minc.files  import MincAtom, xfmToMinc
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.registration import concat_xfmhandlers, invert_xfmhandler, mincmath



#TODO find the nicest API (currently determinants_at_fwhms, but still weird)
#and write documentation indicating it

def lin_from_nlin(xfm : XfmHandler) -> Result[XfmHandler]:
    # TODO add dir argument
    out_xfm = xfm.xfm.newname_with_suffix("_linear_part", subdir="tmp")
    stage = CmdStage(inputs=(xfm.source, xfm.xfm), outputs=(out_xfm,),
                     cmd = (['lin_from_nlin', '-clobber', '-lsq12']
                            + (['-mask', xfm.source.mask.path] if xfm.source.mask else [])
                            + [xfm.target.path, xfm.xfm.path, out_xfm.path]))
    return Result(stages=Stages([stage]),
                  output=XfmHandler(xfm=out_xfm, source=xfm.source,
                                    target=xfm.target))


def minc_displacement(xfm : XfmHandler) -> Result[MincAtom]:
    # TODO: add dir argument
    # TODO: this coercion is lame
    output_grid = xfmToMinc(xfm.xfm.newname_with_suffix("_displ", ext='.mnc', subdir="tmp"))
    stage = CmdStage(inputs=(xfm.source, xfm.xfm), outputs=(output_grid,),
                     cmd=['minc_displacement', '-clobber', xfm.source.path, xfm.xfm.path, output_grid.path])
    return Result(stages=Stages([stage]), output=output_grid)


def mincblob(op : str, grid : MincAtom, subdir : str = "tmp") -> Result[MincAtom]:
    """
    Low-level mincblob wrapper with the one exception being the determinant option. By
    default the inner clockwork of mincblob subtracts 1 from all determinant values that
    are being calculated. As such, 1 needs to be added to the result of the mincblob call.
    We will do that here, because it makes most sense here.
    >>> stages = mincblob('determinant', MincAtom("/images/img_1.mnc", pipeline_sub_dir="/tmp")).stages
    >>> [s.render() for s in stages]
    ['mincblob -clobber -determinant /images/img_1.mnc /tmp/img_1/img_1_determinant.mnc']
    """
    if op not in ["determinant", "trace", "translation", "magnitude"]:
        raise ValueError('mincblob: invalid operation %s' % op)

    # if we are calculating the determinant, the first file produced is a temp file:
    if op == "determinant":
        out_file = grid.newname_with_suffix("_temp_det", subdir=subdir)
    else:
        out_file = grid.newname_with_suffix('_' + op, subdir=subdir)

    stage = CmdStage(inputs=(grid,), outputs=(out_file,),
                 cmd=['mincblob', '-clobber', '-' + op, grid.path, out_file.path])

    s = Stages([stage])
    # now create the proper determinant if that's what was asked for
    if op == "determinant":
        result_file = s.defer(mincmath(op='add',
                                       const=1,
                                       vols=[out_file],
                                       subdir=subdir,
                                       new_name=grid.filename_wo_ext + "_det"))
    else:
        result_file = out_file

    return Result(stages=s, output=result_file)
    #TODO add a 'before' to newname_with, e.g., before="grid" -> disp._grid.mnc


def det_and_log_det(displacement_grid : MincAtom,
                    fwhm : Optional[float],
                    annotation: str = "") -> Result[Namespace]:  # (det=MincAtom, log_det=MincAtom)]:
    """
    When this function is called, you might (or should) know what kind of
    deformation grid is passed along. This allows you to provide a proper
    annotation for the produced log determinant file. For instance "absolute"
    or "relative" for transformations that include an affine linear part, or
    that have the linear part taken out respectively.
    """
    s = Stages()
    # TODO: naming doesn't correspond with the (automagic) file naming: d-1 <=> det(f), det <=> det+1(f)
    det = s.defer(determinant(s.defer(smooth_vector(source=displacement_grid, fwhm=fwhm))
                              if fwhm else displacement_grid))

    output_filename_wo_ext = displacement_grid.filename_wo_ext + "_log_det" + annotation
    if fwhm:
        output_filename_wo_ext += "_fwhm" + str(fwhm)
    log_det = s.defer(mincmath(op='log',
                               vols=[det],
                               subdir="stats-volumes",
                               new_name=output_filename_wo_ext))
    return Result(stages=s, output=Namespace(det=det, log_det=log_det))


def nlin_part(xfm : XfmHandler, inv_xfm : Optional[XfmHandler] = None) -> Result[XfmHandler]:
    """
    *** = non linear deformations
    --- = linear (affine) deformations

    Input:
    xfm     :     ******------>
    inv_xfm :    <******------ (optional)

    Calculated:
    inv_lin_xfm :      <------

    Returned:
    concat :      ******------> +
                       <------
    equals :      ******>

    Compute the nonlinear part of a transform as follows:
    go forwards across xfm and then backwards across the linear part
    of the inverse xfm (by first calculating the inverse or using the one supplied) 
    Finally, use minc_displacement to compute the resulting gridfile of the purely 
    nonlinear part.

    The optional inv_xfm (which must be the inverse!) is an optimization -
    we don't go looking for an inverse by filename munging and don't programmatically
    keep a log of operations applied, so any preexisting inverse must be supplied explicitly.
    """
    s = Stages()
    inv_xfm = inv_xfm or s.defer(invert_xfmhandler(xfm))
    inv_lin_part = s.defer(lin_from_nlin(inv_xfm)) 
    xfm = s.defer(concat_xfmhandlers([xfm, inv_lin_part]))
    return Result(stages=s, output=xfm)


def nlin_displacement(xfm : XfmHandler, inv_xfm : Optional[XfmHandler] = None) -> Result[MincAtom]:
    """
    See: nlin_part().

    This returns the nonlinear part of the input
    transformation (xfm) in the form of a grid file (vector field).
    All transformations are encapsulated in this field (linear parts
    that are normally specified in the .xfm file are placed in the
    vector field)
    """
    
    s = Stages()
    return Result(stages=s,
                  output=s.defer(minc_displacement(
                                   s.defer(nlin_part(xfm, inv_xfm=inv_xfm)))))


def determinants_at_fwhms(xfms       : List[XfmHandler],  # TODO change to pd.Series to get indexing (hence safer inv_xfm)?
                          blur_fwhms : str, # TODO: change back to List[float]; should unblurred dets be found automatically?
                          inv_xfms   : Optional[List[XfmHandler]] = None)   \
                       -> Result[pd.DataFrame]:  # TODO how to write down a Pandas type here ?!
    """
    The most common way to use this function is by providing
    it with transformations that go from the final average
    to an individual. I.e.:

    *** = non linear deformations
    --- = linear (affine) deformations

    xfm     = final-nlin  ******------> individual_input
    inv_xfm = final-nlin <******------  individual_input

    Takes a transformation (xfm) containing
    both lsq12 (scaling and shearing, the 6-parameter
    rotations/translations should not be part of this) and
    non-linear parts of a subject to a common/shared average
    and returns the determinants of both the (forward) nonlinear
    part of the xfm at the given fwhms as well as the determinants
    of the full (forward) transformation.  The inverse transform
    may optionally be specified to avoid its recomputation (e.g.,
    when passing an inverted xfm to determinants_at_fwhms,
    specify the original here).
    """
    s = Stages()

    inv_xfms = inv_xfms or s.defer([invert_xfmhandler(xfm) for xfm in xfms])

    fwhms = [float(x) for x in blur_fwhms.split(',')]

    df = pd.DataFrame([{"xfm" : xfm, "inv_xfm" : inv_xfm, "fwhm" : fwhm,
                        "nlin_det" : nlin_det, "log_nlin_det" : nlin_log_det,
                        "full_det" : full_det, "log_full_det" : full_log_det }
                       for fwhm in fwhms + [0]  # was: None, but this turns to NaN in Pandas ...
                       for xfm, inv_xfm in zip(xfms, inv_xfms)
                       for full_det_and_log_det in
                         [s.defer(det_and_log_det(displacement_grid=s.defer(minc_displacement(xfm)),
                                                  fwhm=fwhm,
                                                  annotation="_abs"))]
                       for full_det, full_log_det in [(full_det_and_log_det.det, full_det_and_log_det.log_det)]
                       for nlin_det_and_log_det in
                         [s.defer(det_and_log_det(displacement_grid=s.defer(nlin_displacement(xfm,
                                                                                              inv_xfm=inv_xfm)),
                                                  fwhm=fwhm,
                                                  annotation="_rel"))]
                       for nlin_det, nlin_log_det in [(nlin_det_and_log_det.det, nlin_det_and_log_det.log_det)]])
    # TODO this is terrible, and should probably be done with joins, but one gets the idea ...
    # TODO remove 'inv_xfm' column?
    # TODO the return of this function is 'everything', not really just 'determinants_at_fwhms' ...
    return Result(stages=s, output=df)


def determinant(displacement_grid : MincAtom) -> Result[MincAtom]:
    """
    Takes a displacement field (deformation grid, vector field, those are
    all the same thing) and calculates the proper determinant (mincblob()
    takes care of adding 1 to the silly output of running mincblob directly)
    """
    s = Stages()
    det = s.defer(mincblob(op='determinant', grid=displacement_grid))
    return Result(stages=s, output=det)

def smooth_vector(source : MincAtom, fwhm : float) -> Result[MincAtom]:
    outf = source.newname_with_suffix("_smooth_fwhm%s" % fwhm, subdir="tmp") # TODO smooth_displacement_?
    cmd  = ['smooth_vector', '--clobber', '--filter', '--fwhm=%s' % fwhm,
            source.path, outf.path]
    stage = CmdStage(inputs=(source,), outputs=(outf,), cmd=cmd)
    return Result(stages=Stages([stage]), output=outf)

StatsConf = NamedTuple("StatsConf", [('stats_kernels', str)])


def voxel_vote(label_files : List[MincAtom], output_dir : str, name : str = "voted"):  # TODO too stringy ...

    if len(label_files) == 0:
        raise ValueError("can't vote with 0 files")

    out = MincAtom(name=os.path.join(output_dir, "%s.mnc" % name), output_sub_dir=output_dir)  # FIXME better naming

    s = CmdStage(cmd=["voxel_vote"] + [l.path for l in label_files] + [out.path],
                 inputs=tuple(label_files),
                 outputs=(out,))

    return Result(stages=Stages([s]), output=out)
