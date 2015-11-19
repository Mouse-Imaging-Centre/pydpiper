import copy
import os.path
import shlex
import subprocess
import re
import sys

import typing
from typing import NamedTuple, Sequence, Tuple, TypeVar

from pydpiper.minc.files            import MincAtom, XfmAtom
from pydpiper.minc.containers       import XfmHandler, Registration
from pydpiper.core.stages     import CmdStage, cmd_stage, Result, Stages
from pydpiper.core.util       import flatten
from pydpiper.core.conversion import InputFile, OutputFile

from pyminc.volumes.factory   import volumeFromFile  # type: ignore

# TODO canonicalize all names of form 'source', 'src', 'src_img', 'dest', 'target', ... in program

# TODO should these atoms/modules be remade as distinct classes with public outf and stage fields (only)?
# TODO output_dir (work_dir?) isn't used but could be useful; assert not(subdir and output_dir)?
def mincblur(img      : MincAtom,
             fwhm     : MincAtom,
             gradient : bool = False,
             subdir   : str = 'tmp') -> Result[MincAtom]:
    """
    >>> img = MincAtom(name='/images/img_1.mnc', pipeline_sub_dir='/scratch/some_pipeline_processed/')
    >>> img_blur = mincblur(img=img, fwhm='0.056')
    >>> img_blur.output.path
    '/scratch/some_pipeline_processed/img_1/tmp/img_1_fwhm0.056_blur.mnc'
    >>> [i.render() for i in img_blur.stages]
    ['mincblur -clobber -no_apodize -fwhm 0.056 /images/img_1.mnc /scratch/some_pipeline_processed/img_1/tmp/img_1_fwhm0.056.mnc']
    """
    suffix = "_dxyz" if gradient else "_blur"
    outf  = img.newname_with_suffix("_fwhm%s" % fwhm + suffix, subdir=subdir)
    stage = CmdStage(
              inputs = [img], outputs = [outf],
              #this used to a function render_fn : conf, inf, outf -> string
              # drop last 9 chars from output filename since mincblur
              # automatically adds "_blur.mnc/_dxyz.mnc" and Python
              # won't lift this length calculation automatically ...
              cmd = shlex.split('mincblur -clobber -no_apodize -fwhm %s %s %s' % (fwhm, img.path, outf.path[:-9]))
                  + (['-gradient'] if gradient else []))
    return Result(stages=Stages([stage]), output=outf)

def mincresample_simple(img             : MincAtom,
                        xfm             : XfmAtom,
                        like            : MincAtom,
                        extra_flags     : Sequence[str] = [],
                        new_name_wo_ext : str = None,
                        subdir          : str = None) -> Result[MincAtom]:
    """
    Resample an image, ignoring mask/labels
    ...
    new_name_wo_ext -- string indicating a user specified file name (without extension)
    """
    
    if not subdir:
        subdir = 'resampled'
        
    if not new_name_wo_ext:
        outf = img.newname_with_fn(lambda _old: xfm.filename_wo_ext + '-resampled', subdir=subdir)
    else:
        # we have the output filename without extension. This should replace the entire
        # current "base" of the filename. 
        outf = img.newname_with_fn(lambda _old: new_name_wo_ext, subdir=subdir)
    
    stage = CmdStage(
              inputs = [xfm, like, img], 
              outputs = [outf],
              cmd = ['mincresample', '-clobber', '-2',
                     '-transform %s' % xfm.path,
                     '-like %s' % like.path,
                     img.path, outf.path] + extra_flags)
    return Result(stages=Stages([stage]), output=outf)

# TODO mincresample_simple could easily be replaced by a recursive call to mincresample
def mincresample(img  : MincAtom,
                 xfm  : XfmAtom,
                 like : MincAtom,
                 extra_flags     : Sequence[str] = [],
                 new_name_wo_ext : str = None,
                 subdir          : str = None) -> Result[MincAtom]:
    """
    ...
    new_name_wo_ext -- string indicating a user specified file name (without extension)
    subdir          -- string indicating which subdirectory to output the file in:
    
    >>> img  = MincAtom('/tmp/img_1.mnc')
    >>> like = MincAtom('/tmp/img_2.mnc')
    >>> xfm  = XfmAtom('/tmp/trans.xfm')
    >>> stages, resampled = mincresample(img=img, xfm=xfm, like=like, extra_flags=[])
    >>> [x.render() for x in stages]
    ['mincresample -clobber -2 -transform /tmp/trans.xfm -like /tmp/img_2.mnc /tmp/img_1.mnc /micehome/bdarwin/git/pydpiper/img_1/resampled/trans-resampled.mnc']
    """
    
    s = Stages()
    
    # when resampling a mask, make sure we use nearest neighbor resampling:
    # and watch out... make sure the copy is a true copy to avoid the original
    # extra_flags list being modified when mask_extra_flags is modified.
    mask_extra_flags = copy.copy(extra_flags)
    if '-sinc' in mask_extra_flags:
        mask_extra_flags.remove('-sinc')
    mask_extra_flags.append('-nearest_neighbour')
    new_mask   = s.defer(mincresample_simple(img.mask, xfm, like, mask_extra_flags, 
                                             new_name_wo_ext, subdir)) if img.mask is not None else None
    # and with labels we have to ensure that the values are not scaled 
    labels_extra_flags = copy.copy(mask_extra_flags)
    labels_extra_flags.append('-keep_real_range')
    new_labels = s.defer(mincresample_simple(img.labels, xfm, like, labels_extra_flags, 
                                             new_name_wo_ext, subdir)) if img.labels is not None else None
    new_img    = s.defer(mincresample_simple(img, xfm, like, extra_flags, 
                                             new_name_wo_ext, subdir))
    
    # Note that new_img can't be used for anything until the mask/label files are also resampled.
    # This shouldn't create a problem with stage dependencies as long as masks/labels appear in inputs/outputs of CmdStages.
    # (If this isn't automatic, a relevant helper function would be trivial.)
    new_img.mask   = new_mask
    new_img.labels = new_labels
    return Result(stages=s, output=new_img)

def xfmconcat(xfms : Sequence[XfmAtom],
              name : str = None) -> Result[XfmAtom]:
    """
    >>> stages, xfm = xfmconcat([MincAtom('/tmp/%s' % i, pipeline_sub_dir='/scratch') for i in ['t1.xfm', 't2.xfm']])
    >>> [s.render() for s in stages]
    ['xfmconcat /tmp/t1.xfm /tmp/t2.xfm /scratch/t1/concat_of_t1_and_t2.xfm']
    """
    if len(xfms) == 0:
        raise ValueError("`xfmconcat` arg `xfms` was empty (can't concat zero files)")
    elif len(xfms) == 1:
        return Result(stages=Stages(), output=xfms[0])
    else:
        if name:
            outf = xfms[0].newname_with_fn(lambda _: name)
        else:
            outf = xfms[0].newname_with_fn(
                lambda _orig: "concat_of_%s" % "_and_".join([x.filename_wo_ext for x in xfms])) #could do names[1:] if dirname contains names[0]?
        stage = CmdStage(
            inputs = xfms, outputs = [outf],
            cmd = shlex.split('xfmconcat %s %s' % (' '.join([x.path for x in xfms]), outf.path)))
        return Result(stages=Stages([stage]), output=outf)


#
#
# TODO: do we need this function at all??
#       the concept seems odd. It's a concat and a resample?
#
#
def concat(ts          : Sequence[XfmHandler],
           name        : str = None,
           extra_flags : Sequence[str] = []) -> Result[XfmHandler]: # TODO remove extra flags OR make ['-sinc'] default
    """
    xfmconcat lifted to work on XfmHandlers instead of MincAtoms
    """
    s = Stages()
    t = s.defer(xfmconcat([t.xfm for t in ts], name=name))
    res = s.defer(mincresample(img=ts[0].source,
                            xfm=t,
                            like=ts[-1].target,
                            extra_flags=["-sinc"])) #TODO move extra flags to mincresample?
    return Result(stages=s,
                  output=XfmHandler(source=ts[0].source,
                                    target=ts[-1].target,
                                    xfm=t,
                                    resampled=res))

def nu_estimate(src : MincAtom) -> Result[MincAtom]:
    out = src.newname_with_suffix("_nu_estimate", ext=".imp")

    # TODO finish dealing with masking as per lines 436-448 of the old LSQ6.py.  (note: the 'singlemask' option there is never used)
    # (currently we don't allow for a single mask or using the initial model's mask if the inputs don't have them)
    cmd = CmdStage(inputs = [src], outputs = [out],
                   cmd = shlex.split("nu_estimate -clobber -distance 8 -iterations 100 -stop 0.001 -fwhm 0.15 -shrink 4 -lambda 5.0e-02")
                       + (['-mask', src.mask.path] if src.mask else []) + [src.path, out.path])
    return Result(stages=Stages([cmd]), output=out)

def nu_evaluate(img : MincAtom,
                field : FileAtom) -> Result[MincAtom]:
    out = img.newname_with_suffix("_nuc")
    cmd = CmdStage(inputs = [img, field], outputs = [out],
                   cmd = ['nu_evaluate', '-clobber', '-mapping', field.path, img.path, out.path])
    return Result(stages=Stages([cmd]), output=out)

def nu_correct(src : MincAtom) -> Result[MincAtom]:
    s = Stages()
    return Result(stages=s, output=s.defer(nu_evaluate(src, s.defer(nu_estimate(src)))))

INormalizeConf = NamedTuple('INormalizeConf',
                            [('const',  int),
                             ('method', str)]) # TODO: should be enum

default_inormalize_conf = INormalizeConf(const=1000, method='ratioOfMedians')
# NB the inormalize default is actually '-medianOfRatios'
# FIXME how do we want to deal with situations where our defaults differ from the tools' defaults,
# and in the latter case should we output the actual settings if the user doesn't explicitly set them?
# should we put defaults into the classes or populate with None (which will often raise an error if disallowed)
# and create default objects?

def inormalize(src  : MincAtom,
               conf : INormalizeConf) -> Result[MincAtom]:
    out = src.newname_with_suffix('_inorm')
    cmd = CmdStage(inputs = [src], outputs = [out],
                   cmd = shlex.split('inormalize -clobber -const %s -%s' % (conf.const, conf.method))
                       + (['-mask', src.mask.path] if src.mask else [])
                       + [src.path, out.path])
    return Result(stages=Stages([cmd]), output=out)


# TODO: a lot of these things were Instance((int,float)) so that
# formatting would be preserved, e.g., a value of 10 wouldn't be printed as
# 10.0 ... now, though, defining an IntFloat alias has no effect since
# no dynamic tests are done.  Would such an alias be useful documentation?
# Unclear ...
RotationalMinctraccConf = NamedTuple('RotationalMinctraccConf',
  [("blur_factor", float),
   ("resample_step_factor", float),
   ("registration_step_factor", float),
   ("w_translations_factor", float),
   ("rotational_range", float),
   ("rotational_interval", float),
   ("temp_dir", str)])

default_rotational_minctracc_conf = RotationalMinctraccConf(
  blur_factor=10,
  resample_step_factor=4,
  registration_step_factor=10,
  w_translations_factor=8,
  rotational_range=50,
  rotational_interval=10,
  temp_dir="/dev/shm")

def get_rotational_minctracc_conf(resolution, rotation_params=None, rotation_range=None,
                                  rotation_interval=None, rotation_tmp_dir=None,
                                  subject_matter=None):
    """
    The parameters for rotational minctracc are set based on the resolution of the input files.
    If not explicitly specified, most parameters are muliples of that resolution.
    
    resolution - int/float indicating the resolution of the input files
    rotation_params - a list with 4 integer elements indicating multiples of the resolution: 
                     [blur factor,
                     resample factor,
                     registration factor,
                     w_translation factor]
     
    """

#FIXME consistently require that masks are explicitely added to inputs array (or not?)
def rotational_minctracc(source : MincAtom,
                         targe  : MincAtom,
                         conf   : RotationalMinctraccConf,
                         resolution      : float,
                         mask            : MincAtom = None,
                         resample_source : bool = False):
    """
    source     -- MincAtom (does not have to be blurred)
    target     -- MincAtom (does not have to be blurred)
    conf       -- RotationalMinctraccConf
    resolution -- (float) resolution at which the registration happens, used
                  to determine all parameters for rotation_minctracc 
    mask       -- MincAtom (optional argument to specify a mask)
    
    This function runs a rotational_minctracc.py call on its two input 
    files.  That program performs a 6 parameter (rigid) registration
    by doing a brute force search in the x,y,z rotation space.  Normally
    the input files have unknown orientation.
    
    simplex -- this simplex will be set based on the resolution provided.
               for the mouse brain we want it to be 1mm. We have mouse brain
               files at either 0.056mm or 0.04mm resultion. For now we will
               determine the value for the simplex by multiplying the 
               resultion by 20.
        
    There are a number of parameters that have to be set and this 
    will be done using factors that depend on the resolution of the
    input files. Here is the list:
        
    argument to be set   --  default (factor)  -- (for 56 micron, translates to)
            blur                    10                    (560 micron) 
        resample stepsize            4                    (224 micron)
      registration stepsize         10                    (560 micron)
        w_translations               8                    (448 micron)
         
    Specifying -1 for the blur argument will result in retrieving an unblurred file.
    The two other parameters that can be set are (in degrees) have defaults:
        
        rotational range          50
        rotational interval       10
        
    Whether or not a mask will be used is based on the presence of a mask 
    in the target file (target.mask).  Alternatively, a mask can be specified using the
    mask argument.
    """
    
    s = Stages()

    # convert the factors into units appropriate for rotational_minctracc (i.e., mm)
    blur_stepsize          = resolution * conf.blur_factor
    resample_stepsize      = resolution * conf.resample_step_factor   
    registration_stepsize  = resolution * conf.registration_step_factor
    w_translation_stepsize = resolution * conf.w_translations_factor 
    
    # blur input files
    blurred_src = s.defer(mincblur(source, blur_stepsize))
    blurred_dest = s.defer(mincblur(target, blur_stepsize))

    out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                                         "%s_rotational_minctracc_to_%s.xfm" % (source.filename_wo_ext, target.filename_wo_ext)),
                      pipeline_sub_dir=source.pipeline_sub_dir,
                      output_sub_dir=source.output_sub_dir)
    
    
    # use the target mask if around, or overwrite this mask with the mask
    # that is explicitly provided:
    mask_for_command = target.mask if target.mask else mask
    cmd = CmdStage(inputs = [source, target] + ([mask_for_command] if mask_for_command else []), outputs = [out_xfm],
        cmd = ["rotational_minctracc.py", 
               "-t", conf.temp_dir, 
               "-w", str(w_translation_stepsize),
               "-s", str(resample_stepsize),
               "-g", str(registration_stepsize),
               "-r", str(conf.rotational_range),
               "-i", str(conf.rotational_interval),
               "--simplex", str(resolution * 20),
               blurred_src.path,
               blurred_dest.path,
               out_xfm.path,
               "/dev/null"] + (['-m', mask_for_command.path] if mask_for_command else []))
    s.update(Stages([cmd]))
    
    resampled = NotImplemented
    if resample_source:
        resampled = s.defer(mincresample(img=source, xfm=out_xfm, like=target, extra_flags=['-sinc']))
        
    return Result(stages=s,
                  output=XfmHandler(source=source,
                                    target=target,
                                    xfm=out_xfm,
                                    resampled=resampled))

R3 = Tuple[float, float, float]

# TODO: move is_nonlinear out of _base_part or even decompose a minctracc
# conf into a pair of configurations?
_base_part   = [("step_sizes", float),
                ("blur_resolution", float),
                ("use_masks", bool),
                ("is_nonlinear", bool)]

_linear_part = [("simplex", float),
                ("transform_type", str),
                # TODO: Enum(None, 'lsq3', 'lsq6', 'lsq7', 'lsq9', 'lsq10', 'lsq12', 'procrustes')
                ("tolerance", float),
                ("w_rotations", R3),
                ("w_translations", R3),
                ("w_scales", R3),
                ("w_shear", R3)]

_nonlinear_part = [("iterations", int),
                   ("use_simplex", bool),
                   ("stiffness", float),
                   ("weight", float),
                   ("similarity", float),
                   ("objective", str),
                   # TODO: Enum(None,'xcorr','diff','sqdiff','label', 'chamfer','corrcoeff','opticalflow')
                   ("lattice_diameter", R3),
                   ("sub_lattice", int),
                   ("max_def_magnitude", R3)]
    
MinctraccConf = NamedTuple('MinctraccConf', _base_part + _linear_part + _nonlinear_part)

LinearMinctraccConf = NamedTuple("LinearMinctraccConf", _base_part + _linear_part)

NonlinearMinctraccConf = NamedTuple("NonlinearMinctraccConf", _base_part + _nonlinear_part)

# should we even keep (hence render) the defaults which are the same as minctracc's?
# does this even get used in the pipeline?  LSQ6/12 get their own
# protocols, and many settings here are the minctracc default
def default_linear_minctracc_conf(transform_type : str) -> MinctraccConf:
    return LinearMinctraccConf(is_nonlinear=False,
                               blur_resolution=None,
                               step_sizes=(0.5,) * 3,
                               use_masks=True,
                               simplex=1,
                               transform_type=transform_type,
                               tolerance=0.001,
                               w_scales=(0.02,) * 3,
                               w_shear=(0.02,) * 3,
                               w_rotations=(0.0174533,) * 3,
                               w_translations=(1.0,) * 3)

default_lsq6_minctracc_conf, default_lsq12_minctracc_conf = [default_linear_minctracc_conf(x) for x in ('lsq6', 'lsq12')]

# TODO: I'm not sure about these defaults.. they should be
# based on the resolution of the input files. This will 
# somewhat work for mouse brains, but is a very coarse 
# registration. What's its purpose?
#
#
# FIXME wrapping this in a function gives a weird weird error ...
#def default_nonlinear_minctracc_conf():
step_size = 0.5
step_sizes = (step_size,) * 3
default_nonlinear_minctracc_conf = NonlinearMinctraccConf(
    is_nonlinear=True,
    step_sizes=step_sizes,
    blur_resolution=step_size,
    use_masks=True,
    iterations=40,
    similarity=0.8,
    use_simplex=True,
    stiffness=0.98,
    weight=0.8,
    objective='corrcoeff',
    lattice_diameter=tuple((x * 3 for x in step_sizes)),
    sub_lattice=6,
    max_def_magnitude=None)

#TODO move to utils
def space_sep(x):
    if isinstance(x, list) or isinstance(x, tuple): #ugh
        return ' '.join(map(str,x))
    else:
        return str(x)

# TODO: add memory estimation hook
def minctracc(source    : MincAtom,
              target    : MincAtom,
              conf      : MinctraccConf,
              transform : XfmAtom = None,
              transform_name_wo_ext : str  = None, 
              generation            : int  = None,
              resample_source       : bool = False):
    # FIXME Currently broken as neither linear nor nonlinear minctracc
    # configuration objects have all the fields required by this function ...
    """
    source
    target
    conf
    transform
    transform_name_wo_ext -- to use for the output transformation (without the extension)
    generation            -- if provided, the transformation name will be:
                             source.filename_wo_ext + "_mincANTS_nlin-" + generation
    resample_source       -- whether or not to resample the source file 
    
    
    minctracc functionality:
    
    LSQ6 -- (conf.is_nonlinear == False) and (conf.transform_type == "lsq6")
    For the 6 parameter alignment we have 
    
    LSQ12 -- (conf.is_nonlinear == False) and (conf.transform_type == "lsq12")
    
    NLIN -- conf.is_nonlinear == True
    
    
    """
    
    s = Stages()
    
    if not conf.transform_type in [None, 'pat', 'lsq3', 'lsq6', 'lsq7', 'lsq9', 'lsq10', 'lsq12', 'procrustes']:
        raise ValueError("minctracc: invalid transform type %s" % conf.transform_type) # not needed if MinctraccConfig(Atom) given
    # FIXME this line should produce a MincAtom (or FileAtom?? doesn't matter if
    # we use only structural properties) with null mask/labels
    if transform_name_wo_ext:
        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                                            "%s.xfm" % (transform_name_wo_ext)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)
    elif generation:
        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                                            "%s_minctracc_nlin-%s.xfm" % (source.filename_wo_ext, generation)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)
    else:
        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                                            "%s_minctracc_to_%s.xfm" % (source.filename_wo_ext, target.filename_wo_ext)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)
    
    source_for_minctracc = source
    target_for_minctracc = target
    if conf.blur_resolution is not None:
        source_for_minctracc = s.defer(mincblur(source, conf.blur_resolution,
                                                gradient=True))
        target_for_minctracc = s.defer(mincblur(target, conf.blur_resolution,
                                                gradient=True))
    
    stage = cmd_stage(flatten(
              ['minctracc', '-clobber', '-debug', '-xcorr'],
              (['-transformation', InputFile(transform.path)] if transform else []),
              (['-' + conf.transform_type] if conf.transform_type else []),
              (['-use_simplex'] if conf.use_simplex is not None else []),
              # FIXME add -est_centre, -est_translations/-identity if not transform (else add transform) !!
              flatten(*[[f,space_sep(v)] for (f,v) in
               [
                ['-step',                  conf.step_sizes],
                ['-simplex',               conf.simplex],
                ['-tol',                   conf.tolerance],
                ['-w_shear',               conf.w_shear],
                ['-w_scales',              conf.w_scales],
                ['-w_rotations',           conf.w_rotations],
                ['-w_translations',        conf.w_translations],
                ['-iterations',            conf.iterations],
                ['-similarity_cost_ratio', conf.similarity],
                ['-lattice_diameter',      conf.lattice_diameter],
                ['-sub_lattice',           conf.sub_lattice]
               ] if v is not None]),
              (['-nonlinear %s' % (conf.objective if conf.objective else '')] if conf.is_nonlinear else []),
              (['-source_mask', InputFile(source.mask.path)] if source.mask and conf.use_masks else []),
              (['-model_mask',  InputFile(target.mask.path)] if target.mask and conf.use_masks else []),
              [InputFile(source_for_minctracc.path), InputFile(target_for_minctracc.path), 
               OutputFile(out_xfm.path)]))

    s.add(stage)
    
    resampled = NotImplemented
    if resample_source:
        resampled = s.defer(mincresample(img=source, xfm=out_xfm, like=target, extra_flags=['-sinc']))

    return Result(stages=s,
                  output=XfmHandler(source=source,
                                    target=target,
                                    xfm=out_xfm,
                                    resampled=resampled))

SimilarityMetricConf = NamedTuple('SimilarityMetricConf',
    [("metric", str),
     ("weight", float),
     ("blur_resolution", float),
     ("radius_or_bins", float),
     ("use_gradient_image", bool)])

default_similarity_metric_conf = SimilarityMetricConf(
    metric="CC",
    weight=1.0,
    blur_resolution=None,
    radius_or_bins=3,
    use_gradient_image=False)

MincANTSConf = NamedTuple("MincANTSConf",
                          [("iterations", str),
                           ("transformation_model", str),
                           ("regularization", str),
                           ("use_mask", bool),
                           ("default_resolution", float),
                           ("sim_metric_confs", Sequence[SimilarityMetricConf])])

# we don't supply a resolution default here because it's preferable
# to take resolution from initial target instead
mincANTS_default_conf = MincANTSConf(
    iterations="100x100x100x150",
    transformation_model="'Syn[0.1]'",
    regularization="'Gauss[2,1]'",
    use_mask=True,
    default_resolution=None,
    sim_metric_confs=[default_similarity_metric_conf,
                      default_similarity_metric_conf._replace(use_gradient_image=False)])

def mincANTS(source : MincAtom,
             target : MincAtom,
             conf   : MincANTSConf,
             transform_name_wo_ext : str  = None,
             generation            : int  = None,
             resample_source       : bool = False) -> Result[XfmHandler]:
    """
    ...
    transform_name_wo_ext -- to use for the output transformation (without the extension)
    generation            -- if provided, the transformation name will be:
                             source.filename_wo_ext + "_mincANTS_nlin-" + generation
    resample_source       -- whether or not to resample the source file   
    
    Construct a single call to mincANTS.
    Also does blurring according to the specified options
    since the cost function might use these.
    """
    s = Stages()
    
    if transform_name_wo_ext:
        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                                            "%s.xfm" % (transform_name_wo_ext)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)
    elif generation:
        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                                            "%s_mincANTS_nlin-%s.xfm" % (source.filename_wo_ext, generation)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)
    else:
        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                                            "%s_mincANTS_to_%s.xfm" % (source.filename_wo_ext, target.filename_wo_ext)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)

    similarity_cmds = []
    similarity_inputs = set()
    for sim_metric_conf in conf.sim_metric_confs:
        if sim_metric_conf.blur_resolution is not None:
            src  = s.defer(mincblur(source, fwhm=sim_metric_conf.blur_resolution,
                                    gradient=sim_metric_conf.use_gradient_image))
            dest = s.defer(mincblur(target, fwhm=sim_metric_conf.blur_resolution,
                                    gradient=sim_metric_conf.use_gradient_image))
        else:
            src  = source
            dest = target
        similarity_inputs.add(src)
        similarity_inputs.add(dest)
        inner = ','.join([src.path, dest.path,
                          str(sim_metric_conf.weight), str(sim_metric_conf.radius_or_bins)])
        subcmd = "'" + "".join([sim_metric_conf.metric, '[', inner, ']']) + "'"
        similarity_cmds.extend(["-m", subcmd])
    stage = CmdStage(inputs = [source, target] + list(similarity_inputs) + ([target.mask] if target.mask else []),
                     # hard to use cmd_stage wrapper here due to complicated subcmds ...
                     outputs = [out_xfm],
                     cmd = ['mincANTS', '3',
                            '--number-of-affine-iterations', '0']
                         + similarity_cmds
                         + ['-t', conf.transformation_model,
                            '-r', conf.regularization,
                            '-i', conf.iterations,
                            '-o', out_xfm.path]
                         + (['-x', target.mask.path] if conf.use_mask and target.mask else []))
    s.add(stage)
    resampled = NotImplemented
    if resample_source:
        resampled = s.defer(mincresample(img=source, xfm=out_xfm, like=target, extra_flags=['-sinc']))
    return Result(stages=s,
                  output=XfmHandler(source=source,
                                    target=target,
                                    xfm=out_xfm,
                                    resampled=resampled))

#def lsq12_NLIN_build_model(...):
#    raise NotImplemented

#def NLIN_build_model(imgs, initial_target, reg_method, nlin_dir, confs):
#    functions = { 'mincANTS'  : mincANTS_NLIN,
#                  'minctracc' : minctracc_NLIN }
#
#    function  = functions[reg_method]  #...[conf.nlin_reg_method] ???
#
#    return function(imgs=imgs, initial_target=initial_target, nlin_dir=nlin_dir, confs=confs)
    
    
def mincANTS_NLIN_build_model(imgs           : Sequence[MincAtom],
                              initial_target : MincAtom,
                              confs          : Sequence[MincANTSConf],
                              nlin_dir       : str) -> Sequence[XfmHandler]:
    """
    This functions runs a hierarchical mincANTS registration on the input
    images (imgs) creating an unbiased average.
    The mincANTS configuration (confs) that is passed in, should be 
    passed in an array of configurations for each of the levels/generations.
    After each round of registrations, an average is created out of the 
    resampled input files, which is then used as the target for the next
    round of registrations. 
    """
    s = Stages()
    avg = initial_target
    avg_imgs = []
    for i, conf in enumerate(confs):
        xfms = [s.defer(mincANTS(source=img, target=avg, conf=conf, generation=i+1,resample_source=True)) for img in imgs]
        # number the generations starting at 1, enumerate will start at 0
        avg  = s.defer(mincaverage([xfm.resampled for xfm in xfms], name_wo_ext='nlin-%d' % (i+1), output_dir=nlin_dir))
        avg_imgs.append(avg)
    return Result(stages=s, output=Registration(xfms=xfms, avg_img=avg, avg_imgs=avg_imgs))

def LSQ12_NLIN(source, target, conf):
    raise NotImplementedError

def intrasubject_registrations(subj, conf): # Subject, lsq12_nlin_conf -> Result(..., (Registration)(xfms))
    """
    subj -- Subject (has a intersubject_registration_time_pt and a time_pt_dict 
            that maps timepoints to individual subjects
    conf -- MincANTSConf 
    
    This function returns a dictionary mapping a time point to a transformation
    {time_pt_1 : transform_from_1_to_2, ..., time_pt_n-1 : transform_from_n-1_to_n}
    and as such the dictionary is one element smaller than the number of time points
    """
    # TODO: somehow the output of this function should provide us with
    # easy access from a MincAtom to an XfmHandler from time_pt N to N+1 
    # either here or in the chain() function 
    
    # TODO: can't do this due to circular references...
    #if not isinstance(subj, Subject):
    #    raise ValueError("The input to intrasubject_registrations (subj) is not of type Subject.")
    s = Stages()
    timepts = sorted(subj.time_pt_dict.items())
    timepts_indices = [index for index,subj_atom in timepts]
    # we need to find the index of the common time point and for that we
    # should only look at the first element of the tuples stored in timepts
    index_of_common_time_pt = timepts_indices.index(subj.intersubject_registration_time_pt)
    time_pt_to_xfms = []
    
    for source_index in range(len(timepts) - 1):
        time_pt_to_xfms.append((timepts_indices[source_index],
                                s.defer(mincANTS(source=timepts[source_index][1],
                                                 target=timepts[source_index + 1][1],
                                                 conf=conf,
                                                 resample_source=True))))
    return Result(stages=s, output=(time_pt_to_xfms,index_of_common_time_pt))


#def multilevel_registration(source, target, registration_function, conf, curr_dir, transform=None):
#    ...

def multilevel_minctracc(source    : MincAtom,
                         target    : MincAtom,
                         confs     : Sequence[MinctraccConf],
                         curr_dir  : str,
                         transform : XfmAtom = None) -> Result[XfmHandler]:
    # TODO fold curr_dir into conf?
    p = Stages()
    for conf in confs:
        # having the basic cmdstage fns act on single items rather than arrays is a bit inefficient,
        # e.g., we create many blur stages (which will later be eliminated, but still ...)
        src_blur  = p.defer(mincblur(source, conf.blur_resolution)) # TODO use conf.use_gradients
        dest_blur = p.defer(mincblur(target, conf.blur_resolution))
        transform = p.defer(minctracc(src_blur, dest_blur, conf=conf, transform=transform))
    return Result(stages=p,
                  output=XfmHandler(xfm=transform,
                                    source=src_blur,
                                    target=dest_blur,
                                    resampled=None))

#"""Multilevel registration of many images to a single target"""
#def multilevel_minctracc_all(sources, target, conf, resolutions, transforms=None):
#    p = Stages()
#    transforms = transforms or [None] * len(sources)
#    for res in resolutions:
#        ss_blurred = (p.add_stages(mincblur(s, res)) for s in sources)
#        t_blurred  = p.add_stages(mincblur(target, res))
#        transforms = [p.extract_stages(minctracc(s, t, conf, res, transform=t_blurred))
#                      for (s,t) in zip(ss_blurred, transforms)]
#    return Result(stages=p, output=transforms)


def multilevel_pairwise_minctracc(imgs       : Sequence[MincAtom], 
                                  conf       : MinctraccConf, 
                                  #transforms : Sequence[] = None, 
                                  like       : MincAtom   = None,
                                  curr_dir   : str        = ".") -> Result[Sequence[XfmHandler]]:
    """Pairwise registration of all images"""
    p = Stages()
    output_dir = os.path.join(curr_dir, 'pairs')
    def avg_xfm_from(src_img):
        """Compute xfm from src_img to each other img, average them, and resample along the result"""
        # TODO to save creation of lots of duplicate blurs, could use multilevel_minctracc_all,
        # being careful not to register the img against itself
        xfms = [p.defer(multilevel_minctracc(src_img, target_img, conf=conf, curr_dir=output_dir))
                for target_img in imgs if src_img != target_img]   # TODO src_img.name != ....name ??
        avg_xfm = p.defer(xfmaverage(xfms, output_dir=curr_dir))
        res  = p.defer(mincresample(img=src_img,
                                 xfm=avg_xfm,
                                 like=like or src_img,
                                 extra_flags=["-sinc"]))
        return XfmHandler(xfm = avg_xfm, source = src_img,
                          target = None, resampled = res) ##FIXME the None here borks things interface-wise ...
                                                          ## does putting `target = res` make sense? could a sum be used?
    return Result(stages=p, output=[avg_xfm_from(img) for img in imgs])

MultilevelMinctraccConf = NamedTuple('MultilevelMinctraccConf',
  [('resolution', float),   # TODO: used to choose step size...shouldn't be here
   ('single_gen_confs', MinctraccConf), # list of minctracc confs for each generation; could fold res/transform_type into these ...
   ('transform_type', str)])

# TODO move LSQ12 stuff to an LSQ12 file
LSQ12_default_conf = MultilevelMinctraccConf(transform_type='lsq12', resolution = NotImplemented,
  single_gen_confs = [])

""" Pairwise lsq12 registration, returning array of transforms and an average image
    Assumption: given that this is a pairwise registration, we assume that all the input
                files to this function have the same shape (i.e. resolution, and dimension sizes.
                This allows us to use any of the input files as the likefile for resampling if
                none is provided. """
# TODO all this does is call multilevel_pairwise_minctracc and then return an average; fold into that procedure?
# TODO eliminate/provide default val for resolutions, move resolutions into conf, finish conf ...

def lsq12_pairwise(imgs : Sequence[MincAtom],
                   conf : LSQ12Conf,
                   lsq12_dir : str,
                   like : MincAtom = None) -> Result[Any]: # TODO: FIXME (Any)
    output_dir = os.path.join(lsq12_dir, 'lsq12')
    #conf.transform_type='-lsq12' # hack ... copy? or set external to lsq12 call ? might be better
    p = Stages()
    xfms = p.defer(multilevel_pairwise_minctracc(imgs=imgs, conf=conf, like=like, curr_dir=output_dir))
    avg_img  = p.defer(mincaverage([x.resampled for x in xfms], output_dir=output_dir))
    return Result(stages = p, output = Registration(avg_imgs=[avg_img], avg_img=avg_img, xfms=xfms))

K = TypeVar('K')

""" This is a direct copy of lsq12_pairwise, with the only difference being that
    that takes dictionaries as input for the imgs, and returns the xfmhandlers as
    dictionaries as well. This is necessary (amongst others) for the registration_chain
    as using dictionaries is the easiest way to keep track of the input files. """
def lsq12_pairwise_on_dictionaries(imgs      : Dict[K, MincAtom],
                                   conf      : LSQ12Conf,
                                   lsq12_dir : str,
                                   like      : MincAtom = None):
    l  = [(k,v) for k, v in sorted(imgs.items())]  # type: List[Tuple[K, MincAtom]]
    ks = [k for k, _ in l]
    vs = [v for _, v in l]
    stages, outputs = lsq12_pairwise(imgs=vs, conf=conf, curr_dir=lsq12_dir, like=like)
    return Result(stages=stages, output=Registration(avg_imgs=outputs.avg_imgs,
                                                      avg_img=outputs.avg_img,
                                                      xfms=dict(zip(ks, outputs.xfms))))

def mincaverage(imgs, name_wo_ext="average", output_dir='.', copy_header_from_first_input=False):
    """
    By default mincaverage only copies header information over to the output
    file that is shared by all input files (xspace, yspace, zspace information etc.)
    In some cases however, you might want to have some of the non-standard header
    information. Use the copy_header_from_first_input argument for this, and it
    will add the -copy_header flag to mincaverage
    """
    if len(imgs) == 0:
        raise ValueError("`mincaverage` arg `imgs` is empty (can't average zero files)")
    # TODO: propagate masks/labels??
    avg = MincAtom(name = os.path.join(output_dir, '%s.mnc' % name_wo_ext), orig_name = None)
    sdfile = MincAtom(name = os.path.join(output_dir, '%s_sd.mnc' % name_wo_ext), orig_name = None)
    additional_flags = []
    if copy_header_from_first_input:
        additional_flags = ['-copy_header']
    s = CmdStage(inputs=imgs, outputs=[avg, sdfile],
          cmd = ['mincaverage', '-clobber', '-normalize',
                 '-max_buffer_size_in_kb', '409620'] + additional_flags +
                 ['-sdfile', sdfile.path] + 
                 [img.path for img in imgs] + 
                 [avg.path])
    return Result(stages=Stages([s]), output=avg)

def xfmaverage(xfms       : Sequence[MincAtom],
               output_dir : str) -> Result[XfmAtom]:
    if len(xfms) == 0:
        raise ValueError("`xfmaverage` arg `xfms` is empty (can't average zero files)")

    # TODO: the path here is probably not right...
    outf  = XfmAtom(name=os.path.join(output_dir, 'transforms/average.xfm'), orig_name=None)
    stage = CmdStage(inputs=xfms, outputs=[outf],
                     cmd=["xfmaverage"] + [x.xfm.path for x in xfms] + [outf.path])
    return Result(stages=Stages([stage]), output=outf)

def xfminvert(xfm : XfmAtom) -> XfmAtom:
    inv_xfm = xfm.newname_with_suffix('_inverted')
    s = CmdStage(inputs=[xfm], outputs=[inv_xfm],
                 cmd=['xfminvert', '-clobber', xfm.path, inv_xfm.path])
    return Result(stages=Stages([s]), output=inv_xfm)

def invert(xfm : XfmHandler) -> XfmHandler:
    """xfminvert lifted to work on XfmHandlers instead of MincAtoms"""
    s = Stages()
    inv_xfm = s.defer(xfminvert(xfm.xfm))
    return Result(stages=s, #FIXME add a resampling stage to get rid of null `resampled` field?
                  output=XfmHandler(xfm=inv_xfm, source=xfm.target, target=xfm.source, resampled=NotImplemented))

def can_read_MINC_file(filename : str) -> bool:
    """Can the MINC file `filename` with be read with `mincinfo`?"""
    
    mincinfoCmd = subprocess.Popen(["mincinfo", filename],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   shell=False)
    # if the following returns anything other than None, the string matched, 
    # and thus indicates that the file could not be read
    return re.search("Unable to open file", mincinfoCmd.stdout.read()) is not None


def check_MINC_input_files(args : Sequence[MincAtom]):
    """
    This is a general function that checks MINC input files to a pipeline. It uses
    the program mincinfo to test whether the input files are readable MINC files,
    and ensures that the input files have distinct filenames. (A directory is created
    for each inputfile based on the filename without extension, which means that 
    filenames need to be distinct)
    
    args: and array of strings (pointing to the input files)
    """ 
    if len(args) < 1:
        raise ValueError("\nError: no input files are provided. Exiting...\n")
    else:
        # here we should also check that the input files can be read
        issuesWithInputs = 0
        for inputF in args:
            if not can_read_MINC_file(inputF):
                print("\nError: can not read input file: " + str(inputF) + "\n", file=sys.stderr)
                issuesWithInputs = 1
        if issuesWithInputs:
            raise ValueError("Error: issues reading input files. Exiting...\n")
    # lastly we should check that the actual filenames are distinct, because
    # directories are made based on the basename
    seen = set()
    for inputF in args:
        fileBase = os.path.splitext(os.path.basename(inputF))[0]
        if fileBase in seen:
            raise ValueError("\nError: the following name occurs at least twice in the "
                             "input file list:\n" + str(fileBase) + ".mnc\nPlease provide "
                             "unique names for all input files.\n")
        seen.add(fileBase)

def get_parameters_for_rotational_minctracc(resolution,
                                            rotation_tmp_dir=None, rotation_range=None, 
                                            rotation_interval=None, rotation_params=None):
    """
    returns the proper combination of a rotation_minctracc configuration and
    a value for the resolution such that the "mousebrain" option for 
    the --lsq6-large-rotations-parameters flag works.
    """
    rotational_configuration = default_rotational_minctracc_conf
    resolution_for_rot = resolution
    # for mouse brains we have fixed parameters:
    if rotation_params == "mousebrain":
        # we want to set the parameters such that 
        # blur          ->  560 micron (10 * resolution)
        # resample step ->  224 micron ( 4 * resolution)
        # registr. step ->  560 micron (10 * resolution)
        # w_trans       ->  448 micron ( 8 * resolution)
        # simplex       -> 1120 micron (20 * resolution)
        # so we simply need to set the resolution to establish this:
        resolution_for_rot = 0.056
    else:
        # use the parameters provided
        if rotation_tmp_dir:
            rotational_configuration.temp_dir = rotation_tmp_dir
        if rotation_range:
            rotational_configuration.rotational_range = rotation_range
        if rotation_interval:
            rotational_configuration.rotational_interval = rotation_interval
        if rotation_params:
            rotational_configuration.blur_factor              = float(rotation_params.split(',')[0])
            rotational_configuration.resample_step_factor     = float(rotation_params.split(',')[1])
            rotational_configuration.registration_step_factor = float(rotation_params.split(',')[2])
            rotational_configuration.w_translations_factor    = float(rotation_params.split(',')[3])
            
    return rotational_configuration, resolution_for_rot
                
    
def lsq6(imgs        : Sequence[MincAtom],
         target      : MincAtom,
         lsq6_method : str,
         resolution  : float,
         rotation_tmp_dir : str   = None,
         rotation_range   : float = None, 
         rotation_interval: float = None,
         rotation_params  : Any   = None) -> Result[Sequence[XfmHandler]]:
    """
    imgs
    target
    lsq6_method  -- from the options, can be: "lsq6_simple", "lsq6_centre_estimation",
                            or "lsq6_large_rotations"
                            
    when the lsq6_method is lsq6_large_rotations, these specify:
    rotation_tmp_dir  -- temp directory used for I/O in rotational_minctracc
    rotation_range    -- range of x,y,z-search space in degrees
    rotation_interval -- step size in degrees along range
    rotation_params   -- list of 4 values (or "mousebrain"), see rotational_minctracc for more info
    """
    # make sure all variables that are passed along are of the appropriate type:
    s = Stages()
    xfm_handlers_to_target = []
    
    ############################################################################
    # alignment
    ############################################################################
    #
    # Calling rotational_minctracc
    #
    if lsq6_method == "lsq6_large_rotations":
        rotational_configuration, resolution_for_rot = \
            get_parameters_for_rotational_minctracc(resolution, rotation_tmp_dir,
                                                    rotation_range, rotation_interval,
                                                    rotation_params) 
        # now call rotational_minctracc on all input images 
        xfm_handlers_to_target = [s.defer(rotational_minctracc(img, target,
                                                               rotational_configuration,
                                                               resolution_for_rot))
                                  for img in imgs]
    #
    # Center estimation
    #
    if lsq6_method == "lsq6_centre_estimation":
        raise NotImplementedError("lsq6_centre_estimation is not implemented yet...")
    #
    # Simple lsq6 (files already roughly aligned)
    #
    if lsq6_method == "lsq6_simple":
        raise NotImplementedError("lsq6_simple is not implemented yet...")
    
    ############################################################################
    # TODO: resample input files ???
    ############################################################################
    
    return Result(stages=s, output=xfm_handlers_to_target)

RegistrationTargets = NamedTuple("RegistrationTargets",
    # what does this mean?
    [("registration_standard", MincAtom),
     ("xfm_to_standard", XfmAtom),
     ("registration_native", MincAtom)
    ])

def lsq6_nuc_inorm(imgs : Sequence[MincAtom],
                   registration_targets : RegistrationTargets,
                   lsq6_method : str,
                   resolution  : float,
                   rotation_tmp_dir : str = None,
                   rotation_range : float = None, 
                   rotation_interval : float = None,
                   rotation_params : Any = None):
    """
    imgs                 -- 
    registration_targets -- ???
    lsq6_method          -- from the options, can be: "lsq6_simple", "lsq6_centre_estimation",
                            or "lsq6_large_rotations"
    """       
    s = Stages()
    
    # 1) run the actual 6 parameter registration
    init_target = registration_targets.registration_standard if not registration_targets.registration_native \
                    else registration_targets.registration_native
    xfm_handlers_source_imgs_to_lsq6_target = s.defer(lsq6(imgs, init_target, lsq6_method, resolution,
                                                           rotation_tmp_dir, rotation_range, 
                                                           rotation_interval, rotation_params))
        
    # 2) concatenate the native_to_standard transform if we have this transform
    xfms_to_final_target_space = []
    if registration_targets.xfm_to_standard:
        xfms_to_final_target_space = [s.defer(xfmconcat([first_xfm.xfm, 
                                                        registration_targets.xfm_to_standard]))
                                              for first_xfm in xfm_handlers_source_imgs_to_lsq6_target]
    else:
        xfms_to_final_target_space = [xfm_handler.xfm for xfm_handler in xfm_handlers_source_imgs_to_lsq6_target]
    
    # 3) resample the input to the final lsq6 space
    #    we should go back to basics in terms of the file name that we create here. It should
    #    be fairly basic. Something along the lines of:
    #    {orig_file_base}_resampled_lsq6.mnc
    filenames_wo_ext_lsq6 = [img.filename_wo_ext + "_resampled_lsq6" for img in imgs]
    imgs_in_lsq6_space = [s.defer(mincresample(img=native_img,
                                               xfm=xfm_to_lsq6,
                                               like=registration_targets.registration_standard,
                                               extra_flags=['-sinc'],
                                               new_name_wo_ext=filename_wo_ext))
                          for native_img,xfm_to_lsq6,filename_wo_ext in zip(imgs,xfms_to_final_target_space,filenames_wo_ext_lsq6)]
    
    # 4) NUC? (TODO: open up NUC parameters)
    # 4a) if we have an initial model, resample the mask to native space and run NUC
    # 4b) if not, simply apply NUC. 
    
    # 5) INORM? apply to the result of 4
    
    # 6) if we have an initial model, and we ran NUC or INORM, resample this file i final
    #    LSQ6 space
    
    # 7) return Result(stages=s, output=Registration(xfms=xfms, avg_img=avg, avg_imgs=avg_imgs))

# class RegistrationTargets(object):
#     """
#     This class can be used for the following options:
#     --init-model
#     --lsq6-target
#     --bootstrap 
#     """
#     registration_standard      = Instance(MincAtom)
#     xfm_to_standard            = Instance(XfmAtom)
#     registration_native        = Instance(MincAtom)


    
def get_registration_targets_from_init_model(init_model_standard_file : str,
                                             output_dir    : str,
                                             pipeline_name : str) -> RegistrationTargets:
    """
    An initial model can have two forms:
    
    1) a model that only has standard space files
    name.mnc --> File in standard registration space.
    name_mask.mnc --> Mask for name.mnc
    
    2) a model with both standard space and native space files
    name.mnc --> File in standard registration space.
    name_mask.mnc --> Mask for name.mnc
    name_native.mnc --> File in native scanner space.
    name_native_mask.mnc --> Mask for name_native.mnc
    name_native_to_standard.xfm --> Transform from native space to standard space
    """
    # initialize the MincAtoms that will be used for the RegistrationTargets
    registration_standard = None
    xfm_to_standard = None
    registration_native = None
    
    # the output directory for files related to the initial model:
    init_model_output_dir = os.path.join(output_dir, pipeline_name + "_init_model")
    
    # first things first, is this a nice MINC file:
    if not can_read_MINC_file(init_model_standard_file):
        raise ValueError("\nError: can not read the following initial model file: %s\n" % init_model_standard_file)
    init_model_dir, standard_file_base =  os.path.split(os.path.splitext(init_model_standard_file)[0])
    init_model_standard_mask = os.path.join(init_model_dir, standard_file_base + "_mask.mnc")
    # this mask file is a prerequisite, so we need to test it
    if not can_read_MINC_file(init_model_standard_mask):
        raise ValueError("\nError (initial model): can not read/find the mask file for the standard space: %s\n" 
               % init_model_standard_mask)
    
    registration_standard = MincAtom(name=init_model_standard_file,
                                     orig_name=init_model_standard_file,
                                     mask=MincAtom(name=init_model_standard_mask,
                                                   orig_name=init_model_standard_mask,
                                                   pipeline_sub_dir=init_model_output_dir),
                                     pipeline_sub_dir=init_model_output_dir)
    
    # check to see if we are dealing with option 2), an initial model with native files
    init_model_native_file = os.path.join(init_model_dir, standard_file_base + "_native.mnc")
    init_model_native_mask = os.path.join(init_model_dir, standard_file_base + "_native_mask.mnc")
    init_model_native_to_standard = os.path.join(init_model_dir, standard_file_base + "_native_to_standard.xfm")
    if os.path.exists(init_model_native_file):
        if not can_read_MINC_file(init_model_native_file):
            raise ValueError("\nError: can not read the following initial model file: %s\n" % init_model_native_file)
        if not can_read_MINC_file(init_model_native_mask):
            raise ValueError("\nError: can not read the following initial model file (required native mask): %s\n" 
                   % init_model_native_mask)
        registration_native = MincAtom(name=init_model_native_file,
                                       orig_name=init_model_native_file,
                                       mask=MincAtom(name=init_model_native_mask,
                                                     orig_name=init_model_output_dir,
                                                     pipeline_sub_dir=init_model_output_dir),
                                       pipeline_sub_dir=init_model_output_dir)
        if not os.path.exists(init_model_native_to_standard):
            raise ValueError("\nError: can not read the following initial model file (required transformation when native "
                   "files exist): %s\n" % init_model_native_to_standard)
        xfm_to_standard = XfmAtom(name=init_model_native_to_standard,
                                   orig_name=init_model_native_to_standard,
                                   pipeline_sub_dir=init_model_output_dir)
    
    return RegistrationTargets(registration_standard=registration_standard,
                               xfm_to_standard=xfm_to_standard,
                               registration_native=registration_native)

def verify_correct_lsq6_target_options(init_model  : MincAtom,
                                       lsq6_target : MincAtom,
                                       bootstrap   : bool) -> None:
    """
    This function can be called using the parameters that are set using 
    the flags:
    --init-model
    --lsq6-target
    --bootstrap
    
    it will check that exactly one of the options is provided and raises 
    an error otherwise
    """
    # check how many options have been specified that can be used as the initial target
    number_of_target_options = sum((bootstrap   != False,
                                    init_model  != None,
                                    lsq6_target != None))
    if number_of_target_options == 0:
        raise ValueError("\nError: please specify a target for the 6 parameter alignment. "
                         "Options are: --lsq6-target, --init-model, --bootstrap.\n")
    if number_of_target_options > 1:
        raise ValueError("\nError: please specify only one of the following options: "
                         "--lsq6-target, --init-model, --bootstrap. Don't know which "
                         "target to use...\n")
    


def get_registration_targets(init_model    : str,
                             lsq6_target   : str,
                             bootstrap     : bool,
                             output_dir    : str,
                             pipeline_name : str,
                             first_input_file : str = None):
    """
    init_model       : value of the flag --init-model (is None, or the name
                       of a MINC file in standard space
    lsq6_target      : value of the flag --lsq6-target (is None, or the name
                       of a target MINC file)
    bootstrap        : value of the flag --bootstrap
    output_dir       : value of the flag --output-dir (top level directory 
                       of the entire process
    pipeline_name    : value of the flag  --pipeline-name
    first_input_file : is None or the name of the first input file. This argument
                       only needs to be specified when --bootstrap is True
    """
    # first check that exactly one of the target methods was chosen
    verify_correct_lsq6_target_options(init_model, lsq6_target, bootstrap)
    
    # if we are dealing with either an lsq6 target or a bootstrap model
    # create the appropriate directories for those
    if lsq6_target is not None:
        if not can_read_MINC_file(lsq6_target):
            raise ValueError("\nError (lsq6 target): can not read MINC file: %s\n" % lsq6_target)
        target_file = MincAtom(name=lsq6_target,
                               orig_name=lsq6_target,
                               pipeline_sub_dir=os.path.join(output_dir, pipeline_name +
                                                             "_target_file"))
        return RegistrationTargets(registration_standard=target_file,
                                   xfm_to_standard=None,
                                   registration_native=None)
    if bootstrap:
        if not first_input_file:
            raise ValueError("\nError: (developer's error) the function "
                             "get_registration_targets is called with bootstrap=True "
                             "but the first input file to the pipeline was not passed "
                             "along. Don't know which file to use as target for LSQ6.\n")
        if not can_read_MINC_file(first_input_file):
            raise ValueError("\nError (bootstrap file): can not read MINC file: %s\n" % first_input_file)
        bootstrap_file = MincAtom(name=first_input_file,
                                  orig_name=first_input_file,
                                  pipeline_sub_dir=os.path.join(output_dir, pipeline_name +
                                                                "_bootstrap_file"))
        return(RegistrationTargets(registration_standard=bootstrap_file))
    
    if init_model:
        return get_registration_targets_from_init_model(init_model, output_dir, pipeline_name)

 
def get_resolution_from_file(input_file : str) -> float:
    """
    input_file -- string pointing to an existing MINC file
    """
    # quite important is that this file actually exists...
    if not can_read_MINC_file(input_file):
        raise IOError("\nError: can not read input file: %s\n" % input_file)
    
    image_resolution = volumeFromFile(input_file).separations

    return min([abs(x) for x in image_resolution])
