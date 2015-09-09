from pydpiper.core.stages import Stages, CmdStage
from pydpiper.core.containers import Result
from pydpiper.minc.files import MincAtom
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.registration import concat, invert

#TODO find the nicest API (currently determinants_at_fwhms, but still weird)
#and write documentation indicating it

def lin_from_nlin(xfm): # xfm -> xfm
    # TODO add dir argument
    out_xfm = xfm.xfm.newname_with_suffix("_linear_part")
    stage = CmdStage(inputs=[xfm.source, xfm.xfm], outputs=[out_xfm],
                     cmd = ['lin_from_nlin', '-clobber', '-lsq12'] \
                         + (['-mask', xfm.source.mask] if xfm.source.mask else []) \
                         + [xfm.source.path, xfm.xfm.path, out_xfm.path])
    return Result(stages=Stages([stage]),
                  output=XfmHandler(xfm=out_xfm, source=xfm.source,
                                    target=xfm.target))

def minc_displacement(xfm): # xfm -> mnc
    # TODO add dir argument
    output_grid = xfm.xfm.newname_with_suffix("_displacement", ext='.mnc')
    stage = CmdStage(inputs=[xfm.source, xfm.xfm], outputs=[output_grid],
                     cmd=['minc_displacement', '-clobber', xfm.source.path, xfm.xfm.path, output_grid.path])
    return Result(stages=Stages([stage]), output=output_grid)

def mincblob(op, grid): # str, mnc -> mnc
    """
    Low-level mincblob wrapper -- use `determinant` instead to actually compute a determinant ...
    >>> stages = mincblob('determinant', MincAtom("/images/img_1.mnc", output_dir="/tmp")).stages
    >>> stages.pop().render()
    'mincblob -clobber -determinant /images/img_1.mnc /tmp/img_1/img_1_determinant.mnc'
    """
    # FIXME could automatically add 1 here for determinant; what about others?
    if op not in ["determinant", "trace", "translation", "magnitude"]:
        raise ValueError('mincblob: invalid operation %s' % op)
    out_grid = grid.newname_with_suffix('_' + op)
    s = CmdStage(inputs=[grid], outputs=[out_grid],
                 cmd=['mincblob', '-clobber', '-' + op, grid.get_path(), out_grid.get_path()])
    return Result(stages=Stages([s]), output=out_grid)
    #TODO add a 'before' to newname_with, e.g., before="grid" -> disp._grid.mnc

def det_and_log_det(displacement_grid, fwhm):
    s = Stages()
    # NB naming doesn't correspond with the (automagic) file naming: d-1 <=> det(f), det <=> det+1(f)
    det = s.defer(determinant(s.defer(smooth_vector(source=displacement_grid, fwhm=fwhm))
                           if fwhm else displacement_grid))
    log_det = s.defer(mincmath(op='log', vols=[det]))
    return Result(stages=s, output={ 'det': det, 'log_det': log_det})

def nlin_part(xfm, inv_xfm=None): # xfm -> xfm
    """Compute the nonlinear part of a transform as follows:
    go backwards across xfm (by first calculating the inverse or using the one supplied)
    and then forwards across only the linear part.  Finally, use minc_displacement
    to compute the resulting gridfile of the purely nonlinear part.

    The optional inv_xfm (which must be the inverse!) is an optimization -
    we don't go looking for an inverse by filename munging and don't programmatically
    keep a log of operations applied, so any preexisting inverse must be supplied explicitly."""
    s = Stages()
    lin_part = s.defer(lin_from_nlin(xfm))
    inv_xfm = inv_xfm or s.defer(invert(xfm))
    xfm = s.defer(concat([inv_xfm, lin_part]))
    return Result(stages=s, output=xfm)

def nlin_displacement(xfm, inv_xfm=None): # xfm -> mnc
    # "minc_displacement <=< nlin_part"
    s = Stages()
    return Result(stages=s, output=s.defer(minc_displacement(s.defer(nlin_part(xfm, inv_xfm)))))

def determinants_at_fwhms(xfm, blur_fwhms):
    # "(nlin_displacement, minc_displacement) <=< invert"
    s = Stages()

    inv_xfm = s.defer(invert(xfm))

    nlin_disp = s.defer(nlin_displacement(xfm=inv_xfm, inv_xfm=xfm))
    full_disp = s.defer(minc_displacement(inv_xfm))
    # TODO add the option to add additional xfm?  (do we even need this? used in mbm etc but might be trivial)
    
    #for blur in blur_fwhms: # throws away the results
    #    s.defer(det_and_log_det(nlin_disp, blur))
    #    s.defer(det_and_log_det(full_disp, blur))
    nlin_dets = [(fwhm, s.defer(det_and_log_det(nlin_disp, fwhm))) for fwhm in blur_fwhms]
    full_dets = [(fwhm, s.defer(det_and_log_det(full_disp, fwhm))) for fwhm in blur_fwhms]
    # won't work when additional xfm is specified for nlin_dets:
    #(nlin_dets, full_dets) = [[(fwhm, s.defer(det_and_log_det(disp, fwhm))) for fwhm in blur_fwhms]
    #                          for disp in (nlin_disp, full_disp)]

    #could return a different isomorphic presentation of this big structure...
    return Result(stages=s, output=(nlin_dets, full_dets))
    
def mincmath(op, vols, const=None, new_name=None):
    """Low-level/stupid interface to mincmath"""
    const = str(const) if const is not None else None
    name = new_name if new_name else \
      ('_' + op + '_' + ((const + '_') if const else '') + '_'.join([vol.name for vol in vols]))
    outf = vols[0].newname_with_suffix(name)
    s = CmdStage(inputs=[vols], outputs=[outf],
                 cmd=['mincmath', '-clobber', '-2'] \
                 + (['-const', const] if const else []) \
                 + ['-' + op] + [v.path for v in vols])
    return Result(stages=Stages([s]), output=outf)

def determinant(displacement_grid):
    s = Stages()
    det_m_1 = s.defer(mincblob(op='determinant', grid=displacement_grid))
    det = s.defer(mincmath(op='add', const=1, vols=[det_m_1]))
    return Result(stages=s, output=det)

def smooth_vector(source, fwhm):
    outf = source.newname_with_suffix("_smooth_fwhm%s" % fwhm) # TODO smooth_displacement_?
    cmd  = ['smooth_vector', '--clobber', '--filter', str(fwhm), source.path, outf.path]
    stage = CmdStage(inputs=[source], outputs=[outf], cmd=cmd)
    return Result(stages=Stages([stage]), output=outf)

