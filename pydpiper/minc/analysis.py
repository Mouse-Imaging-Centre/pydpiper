from pydpiper.core.stages import Stages, CmdStage, Result
from pydpiper.core.util   import NamedTuple
from pydpiper.minc.files  import MincAtom, xfmToMinc
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.registration import concat_xfmhandlers, invert_xfmhandler

from typing import List, Optional, Tuple

import os

#TODO find the nicest API (currently determinants_at_fwhms, but still weird)
#and write documentation indicating it

def lin_from_nlin(xfm : XfmHandler) -> Result[XfmHandler]:
    # TODO add dir argument
    out_xfm = xfm.xfm.newname_with_suffix("_linear_part")
    stage = CmdStage(inputs=(xfm.source, xfm.xfm), outputs=(out_xfm,),
                     cmd = (['lin_from_nlin', '-clobber', '-lsq12']
                            + (['-mask', xfm.source.mask.path] if xfm.source.mask else [])
                            + [xfm.source.path, xfm.xfm.path, out_xfm.path]))
    stage.set_log_file(os.path.join(out_xfm.pipeline_sub_dir,
                                    out_xfm.output_sub_dir,
                                    "log",
                                    "lin_from_nlin_" + out_xfm.filename_wo_ext + ".log"))
    return Result(stages=Stages([stage]),
                  output=XfmHandler(xfm=out_xfm, source=xfm.source,
                                    target=xfm.target))

def minc_displacement(xfm : XfmHandler) -> Result[MincAtom]:
    # TODO: add dir argument
    # TODO: this coercion is lame
    output_grid = xfmToMinc(xfm.xfm.newname_with_suffix("_displacement", ext='.mnc'))
    stage = CmdStage(inputs=(xfm.source, xfm.xfm), outputs=(output_grid,),
                     cmd=['minc_displacement', '-clobber', xfm.source.path, xfm.xfm.path, output_grid.path])
    stage.set_log_file(os.path.join(output_grid.pipeline_sub_dir,
                                    output_grid.output_sub_dir,
                                    "log",
                                    "minc_displacement_" + output_grid.filename_wo_ext + ".log"))
    return Result(stages=Stages([stage]), output=output_grid)

def mincblob(op : str, grid : MincAtom) -> Result[MincAtom]:
    """
    Low-level mincblob wrapper -- use `determinant` instead to actually compute a determinant ...
    >>> stages = mincblob('determinant', MincAtom("/images/img_1.mnc", pipeline_sub_dir="/tmp")).stages
    >>> [s.render() for s in stages]
    ['mincblob -clobber -determinant /images/img_1.mnc /tmp/img_1/img_1_determinant.mnc']
    """
    # FIXME could automatically add 1 here for determinant; what about others?
    if op not in ["determinant", "trace", "translation", "magnitude"]:
        raise ValueError('mincblob: invalid operation %s' % op)
    out_grid = grid.newname_with_suffix('_' + op)
    s = CmdStage(inputs=(grid,), outputs=(out_grid,),
                 cmd=['mincblob', '-clobber', '-' + op, grid.path, out_grid.path])
    s.set_log_file(os.path.join(out_grid.pipeline_sub_dir,
                                out_grid.output_sub_dir,
                                "log",
                                "mincblob_" + out_grid.filename_wo_ext + ".log"))
    return Result(stages=Stages([s]), output=out_grid)
    #TODO add a 'before' to newname_with, e.g., before="grid" -> disp._grid.mnc

def det_and_log_det(displacement_grid : MincAtom, fwhm : float) -> Result[Tuple[MincAtom, MincAtom]]:
    s = Stages()
    # TODO: naming doesn't correspond with the (automagic) file naming: d-1 <=> det(f), det <=> det+1(f)
    det = s.defer(determinant(s.defer(smooth_vector(source=displacement_grid, fwhm=fwhm))
                           if fwhm else displacement_grid))
    log_det = s.defer(mincmath(op='log', vols=[det]))
    return Result(stages=s, output=(det, log_det))

def nlin_part(xfm : XfmHandler, inv_xfm : Optional[XfmHandler] = None) -> Result[XfmHandler]:
    """Compute the nonlinear part of a transform as follows:
    go forwards across xfm and then backwards across the linear part
    of the inverse xfm (by first calculating the inverse or using the one supplied) 
    Finally, use minc_displacement to compute the resulting gridfile of the purely 
    nonlinear part.

    The optional inv_xfm (which must be the inverse!) is an optimization -
    we don't go looking for an inverse by filename munging and don't programmatically
    keep a log of operations applied, so any preexisting inverse must be supplied explicitly."""
    s = Stages()
    inv_xfm = inv_xfm or s.defer(invert(xfm))
    inv_lin_part = s.defer(lin_from_nlin(inv_xfm)) 
    xfm = s.defer(concat_xfmhandlers([xfm, inv_lin_part]))
    return Result(stages=s, output=xfm)

def nlin_displacement(xfm : XfmHandler, inv_xfm : Optional[XfmHandler] = None) -> Result[MincAtom]:
    # "minc_displacement <=< nlin_part"
    
    s = Stages()
    return Result(stages=s,
                  output=s.defer(minc_displacement(
                         s.defer(nlin_part(xfm, inv_xfm)))))

def determinants_at_fwhms(xfm        : XfmHandler,
                          blur_fwhms : str, # TODO: change back to List[float]
                          inv_xfm    : Optional[XfmHandler] = None)   \
                       -> Result[Tuple[List[Tuple[float, Tuple[MincAtom, MincAtom]]],  \
                                       List[Tuple[float, Tuple[MincAtom, MincAtom]]]]]:
    """
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
    # "(nlin_displacement, minc_displacement) <=< invert"
    s = Stages()

    inv_xfm = inv_xfm or s.defer(invert(xfm))

    nlin_disp = s.defer(nlin_displacement(xfm=xfm, inv_xfm=inv_xfm))
    full_disp = s.defer(minc_displacement(xfm))
    # TODO add the option to add additional xfm?  (do we even need this? used in mbm etc but might be trivial)
   
    fwhms = [float(x) for x in blur_fwhms.split(',')]
 
    nlin_dets = [(fwhm, s.defer(det_and_log_det(nlin_disp, fwhm))) for fwhm in fwhms]
    full_dets = [(fwhm, s.defer(det_and_log_det(full_disp, fwhm))) for fwhm in fwhms]
    # won't work when additional xfm is specified for nlin_dets:
    #(nlin_dets, full_dets) = [[(fwhm, s.defer(det_and_log_det(disp, fwhm))) for fwhm in blur_fwhms]
    #                          for disp in (nlin_disp, full_disp)]

    #could return a different isomorphic presentation of this big structure...
    return Result(stages=s, output=(nlin_dets, full_dets))
    
def mincmath(op       : str,
             vols     : List[MincAtom],
             const    : Optional[float] = None,
             new_name : Optional[str]   = None) -> Result[MincAtom]:
    """
    Low-level/stupid interface to mincmath
    """
    _const = str(const) if const is not None else ""  # type: Optional[str]

    if new_name:
        name = new_name
    elif len(vols) == 1:
        name = vols[0].filename_wo_ext + "_" + op + "_" + _const
    else:
        name = (op + '_' + ((_const + '_') if _const else '') +
             '_'.join([vol.filename_wo_ext for vol in vols]))

    outf = vols[0].newname_with_fn(lambda x: name)
    s = CmdStage(inputs=tuple(vols), outputs=(outf,),
                 cmd=(['mincmath', '-clobber', '-2']
                   + (['-const', _const] if _const else [])
                   + ['-' + op] + [v.path for v in vols] + [outf.path]))
    s.set_log_file(os.path.join(outf.pipeline_sub_dir,
                                outf.output_sub_dir,
                                "log",
                                "mincmath_" + outf.filename_wo_ext + ".log"))
    return Result(stages=Stages([s]), output=outf)

def determinant(displacement_grid : MincAtom) -> Result[MincAtom]:
    s = Stages()
    det_m_1 = s.defer(mincblob(op='determinant', grid=displacement_grid))
    det = s.defer(mincmath(op='add', const=1, vols=[det_m_1]))
    return Result(stages=s, output=det)

def smooth_vector(source : MincAtom, fwhm : float) -> Result[MincAtom]:
    outf = source.newname_with_suffix("_smooth_fwhm%s" % fwhm) # TODO smooth_displacement_?
    cmd  = ['smooth_vector', '--clobber', '--filter', '--fwhm=%s' % fwhm,
            source.path, outf.path]
    stage = CmdStage(inputs=(source,), outputs=(outf,), cmd=cmd)
    stage.set_log_file(os.path.join(outf.pipeline_sub_dir,
                                    outf.output_sub_dir,
                                    "log",
                                    "smooth_vector_" + outf.filename_wo_ext + ".log"))
    return Result(stages=Stages([stage]), output=outf)

StatsConf = NamedTuple("StatsConf", [('stats_kernels', str)])
