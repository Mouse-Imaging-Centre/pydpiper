from argparse import Namespace
from typing import List, Optional
import os

import pandas as pd

from pydpiper.core.stages import Stages, CmdStage, Result
from pydpiper.core.util   import NamedTuple
from pydpiper.core.templating import rendered_template_to_command, templating_env
from pydpiper.minc.files  import MincAtom
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.registration import concat_xfmhandlers, mincmath, minc_displacement, determinant
from pydpiper.minc.registration import MincAlgorithms

#TODO find the nicest API (currently determinants_at_fwhms, but still weird)
#and write documentation indicating it

def lin_from_nlin(xfm : XfmHandler) -> Result[XfmHandler]:
    # TODO add dir argument
    out_xfm = xfm.xfm.newname_with_suffix("_linear_part", subdir="tmp")
    stage = CmdStage(inputs=(xfm.moving, xfm.xfm), outputs=(out_xfm,),
                     cmd = (['lin_from_nlin', '-clobber', '-lsq12']
                            + (['-mask', xfm.moving.mask.path] if xfm.moving.mask else [])
                            + [xfm.fixed.path, xfm.xfm.path, out_xfm.path]))
    return Result(stages=Stages([stage]),
                  output=XfmHandler(xfm=out_xfm, source=xfm.moving,
                                    target=xfm.fixed))


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
    inv_xfm = inv_xfm or s.defer(MincAlgorithms.invert_xfmhandler(xfm))
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

    inv_xfms = [s.defer(MincAlgorithms.invert_xfmhandler(xfm)) for xfm in xfms] if inv_xfms is None else inv_xfms

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


def smooth_vector(source : MincAtom, fwhm : float) -> Result[MincAtom]:
    outf = source.newname_with_suffix("_smooth_fwhm%s" % fwhm, subdir="tmp") # TODO smooth_displacement_?
    cmd  = rendered_template_to_command(
        smooth_vector_template.render(source = source.path, outf = outf.path, fwhm = fwhm))
    stage = CmdStage(inputs=(source,), outputs=(outf,), cmd=cmd)
    return Result(stages=Stages([stage]), output=outf)

StatsConf = NamedTuple("StatsConf", [('stats_kernels', str)])

voxel_vote_template = templating_env.get_template("voxel_vote.sh")

def voxel_vote(label_files : List[MincAtom], output_dir : str, name : str = "voted"):  # TODO too stringy ...

    if len(label_files) == 0:
        raise ValueError("can't vote with 0 files")

    out = MincAtom(name=os.path.join(output_dir, "%s.mnc" % name),
                   output_sub_dir=output_dir)  # FIXME better naming

    s = CmdStage(cmd=rendered_template_to_command(voxel_vote_template.render(
                     label_files = [l.path for l in sorted(label_files)],
                     out = out.path)),
                 inputs = tuple(label_files),
                 outputs=(out,))

    return Result(stages=Stages([s]), output=out)
