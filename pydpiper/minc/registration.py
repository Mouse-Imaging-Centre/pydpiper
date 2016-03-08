import csv
import os.path
import os
import random
import shlex
import subprocess
import sys
import inspect
import re

from configargparse import Namespace
from typing import Any, cast, Dict, Generic, Iterable, List, Optional, Set, Tuple, TypeVar, Union

from pydpiper.core.files import FileAtom
from pydpiper.core.stages import CmdStage, Result, Stages
from pydpiper.core.util import pairs, AutoEnum, NamedTuple
from pydpiper.minc.files import MincAtom, XfmAtom
from pydpiper.minc.containers import XfmHandler

from pyminc.volumes.factory import volumeFromFile  # type: ignore



R3 = Tuple[float, float, float]


class LinearTransType(AutoEnum):
    lsq3 = ()
    lsq6 = ()
    lsq7 = ()
    lsq9 = ()
    lsq10 = ()
    lsq12 = ()
    procrustes = ()


class Objective(AutoEnum):
    # named so that <...>.name returns the appropriate string to pass to minctracc
    xcorr = diff = sqdiff = label = chamfer = corrcoeff = opticalflow = ()


class InputSpace(AutoEnum):
    native = lsq6 = lsq12 = ()


class TargetType(AutoEnum):
    initial_model = bootstrap = target = pride_of_models = ()


RegistrationConf = NamedTuple("RegistrationConf", [("input_space", InputSpace),
                                                   ("resolution", float),
                                                   ("subject_matter", Optional[str]),
                                                   #("target_type", TargetType),
                                                   #("target_file", Optional[str])
                                                   ])


LSQ6Conf = NamedTuple("LSQ6Conf", [("run_lsq6", bool),
                                   ("lsq6_method", str),
                                   ("rotation_tmp_dir", Optional[str]),
                                   ("rotation_range", Optional[float]),
                                   ("rotation_interval", Optional[float]),
                                   ("rotation_params", Optional[str]),
                                   ("copy_header_info", bool),
                                   #("bootstrap", bool),
                                   #("init_model", Optional[str]),
                                   #("lsq6_target", Optional[str]),
                                   ("target_type", TargetType),
                                   ("target_file", Optional[str]),
                                   ("inormalize", bool),
                                   ("nuc", bool),
                                   ("protocol_file", Optional[str]),
                                   ])


LSQ12Conf = NamedTuple('LSQ12Conf', [('run_lsq12', bool),
                                     ('max_pairs', Optional[int]),  # should these be handles, not strings?
                                     ('like_file', Optional[str]),   # in that case, parser could open the file...
                                     ('protocol', Optional[str])])


LinearMinctraccConf = NamedTuple("LinearMinctraccConf",
                                 [("simplex", float),
                                  ("transform_type", Optional[LinearTransType]),
                                  ("tolerance", float),
                                  ("w_rotations", R3),
                                  ("w_translations", R3),
                                  ("w_scales", R3),
                                  ("w_shear", R3)])

# TODO writing a LinearMinctraccConf is annoying b/c of the nested structure,
# so write linear_minctracc_conf : ... -> MinctraccConf

NonlinearMinctraccConf = NamedTuple("NonlinearMinctraccConf",
                                    [("iterations", int),
                                     ("use_simplex", bool),
                                     ("stiffness", float),
                                     ("weight", float),
                                     ("similarity", float),
                                     ("objective", Optional[Objective]),
                                     ("lattice_diameter", R3),
                                     ("sub_lattice", int),
                                     ("max_def_magnitude", R3)])

MinctraccConf = NamedTuple('MinctraccConf',
                           [("step_sizes", R3),
                            ("blur_resolution", float),
                            ("use_masks", bool),
                            ("use_gradient", bool),
                            ("linear_conf", Optional[LinearMinctraccConf]),
                            ("nonlinear_conf", Optional[NonlinearMinctraccConf])])

class MultilevelMinctraccConf(object):
    def __init__(self, confs: List[MinctraccConf]):
        self.confs = confs


def atoms_from_same_subject(atoms: List[FileAtom]):
    """
    Determines whether the list of atoms provided come
    from the same subject. This is based on the
    output_sub_dir field.
    """
    # if all atoms belong to the same subject, they will share the same
    # output_sub_dir
    first_output_sub_dir = atoms[0].output_sub_dir
    all_from_same_sub = True
    for atom in atoms:
        if atom.output_sub_dir != first_output_sub_dir:
            all_from_same_sub = False
    return all_from_same_sub

# TODO canonicalize all names of form 'source', 'src', 'src_img', 'dest', 'target', ... in program

# TODO should these atoms/modules be remade as distinct classes with public outf and stage fields (only)?
# TODO output_dir (work_dir?) isn't used but could be useful; assert not(subdir and output_dir)?

# FIXME mincblur has a slight weirdness that it returns a single file determined by the gradient parameter,
# but the command-line tool actually creates both files if gradient=True.  As a result, there's a possibility that
# two distinct mincblur stages may try to write to the same blur file, since that output is not tracked in the
# mincblur stage running with -gradient.  Obvious solution: just return both outputs.
# Maybe this is one cause of the empty/corrupted files occasionally encountered?
def mincblur(img: MincAtom,
             fwhm: float,
             gradient: bool = False,
             subdir: str = 'tmp') -> Result[MincAtom]:
    """
    >>> img = MincAtom(name='/images/img_1.mnc', pipeline_sub_dir='/scratch/some_pipeline_processed/')
    >>> img_blur = mincblur(img=img, fwhm=0.056)
    >>> img_blur.output.path
    '/scratch/some_pipeline_processed/img_1/tmp/img_1_fwhm0.056_blur.mnc'
    >>> [i.render() for i in img_blur.stages]
    ['mincblur -clobber -no_apodize -fwhm 0.056 /images/img_1.mnc /scratch/some_pipeline_processed/img_1/tmp/img_1_fwhm0.056']
    """
    suffix = "_dxyz" if gradient else "_blur"
    outf = img.newname_with_suffix("_fwhm%s" % fwhm + suffix, subdir=subdir)
    stage = CmdStage(
        inputs=(img,), outputs=(outf,),
        # drop last 9 chars from output filename since mincblur
        # automatically adds "_blur.mnc/_dxyz.mnc" and Python
        # won't lift this length calculation automatically ...
        cmd=shlex.split('mincblur -clobber -no_apodize -fwhm %s %s %s' % (fwhm, img.path, outf.path[:-9]))
            + (['-gradient'] if gradient else []))
    stage.set_log_file(os.path.join(outf.pipeline_sub_dir,
                                    outf.output_sub_dir,
                                    "log",
                                    "mincblur_" + outf.filename_wo_ext + ".log"))
    return Result(stages=Stages((stage,)), output=outf)


def mincaverage(imgs: List[MincAtom],
                name_wo_ext: str = "average",
                output_dir: str = '.',
                avg_file: Optional[MincAtom] = None,
                copy_header_from_first_input: bool = False):
    """
    By default mincaverage only copies header information over to the output
    file that is shared by all input files (xspace, yspace, zspace information etc.)
    In some cases however, you might want to have some of the non-standard header
    information. Use the copy_header_from_first_input argument for this, and it
    will add the -copy_header flag to mincaverage.
    """
    if len(imgs) == 0:
        raise ValueError("`mincaverage` arg `imgs` is empty (can't average zero files)")
    # TODO: propagate masks/labels??

    # the output_dir basically gives us the equivalent of the pipeline_sub_dir for
    # regular input files to a pipeline, so use that here
    avg = avg_file or MincAtom(name=os.path.join(output_dir, '%s.mnc' % name_wo_ext),
                               orig_name=None,
                               pipeline_sub_dir=output_dir)
    # if the average was provided as a MincAtom there is probably a output_sub_dir
    # set. However, this is something we only really use for input files. All averages
    # and related files to directly into the _lsq6, _lsq12 and _nlin directories. That's
    # why we'll create this MincAtom here if avg_file was provided:
    sdfile = MincAtom(name=os.path.join(avg_file.dir, '%s_sd.mnc' % avg_file.filename_wo_ext),
                      orig_name=None,
                      pipeline_sub_dir=avg_file.pipeline_sub_dir) if avg_file else \
             MincAtom(name=os.path.join(output_dir, '%s_sd.mnc' % name_wo_ext),
                      orig_name=None,
                      pipeline_sub_dir=output_dir)
    additional_flags = ["-copy_header"] if copy_header_from_first_input else []  # type: List[str]
    s = CmdStage(inputs=tuple(imgs), outputs=(avg, sdfile),
                 cmd=['mincaverage', '-clobber', '-normalize',
                      '-max_buffer_size_in_kb', '409620'] + additional_flags +
                     ['-sdfile', sdfile.path] +
                     sorted([img.path for img in imgs]) +
                     [avg.path])
    s.set_log_file(os.path.join(avg.pipeline_sub_dir,
                                avg.output_sub_dir,
                                "log",
                                "mincaverage_" + avg.filename_wo_ext + ".log"))
    return Result(stages=Stages([s]), output=avg)


def pmincaverage(imgs: List[MincAtom],
                 name_wo_ext: str = "average",
                 output_dir: str = '.',
                 copy_header_from_first_input: bool = False):
    if copy_header_from_first_input:
        # TODO: should be logged, not printed?
        print("Warning: pmincaverage doesn't implement copy_header; use mincaverage instead", sys.stderr)
    avg = MincAtom(name=os.path.join(output_dir, "%s.mnc" % name_wo_ext), orig_name=None)
    s = CmdStage(inputs=tuple(imgs), outputs=(avg,),
                 cmd=["pmincaverage", "--clobber"] + sorted([img.path for img in imgs]) + [avg.path])
    s.set_log_file(os.path.join(avg.pipeline_sub_dir,
                                avg.output_sub_dir,
                                "log",
                                "pmincaverage_" + avg.filename_wo_ext + ".log"))
    return Result(stages=Stages([s]), output=avg)


class Interpolation(AutoEnum):
    trilinear = tricubic = sinc = nearest_neighbour = ()


def mincresample_simple(img: MincAtom,
                        xfm: XfmAtom,
                        like: MincAtom,
                        extra_flags: Tuple[str] = (),
                        interpolation: Optional[Interpolation] = None,
                        new_name_wo_ext: str = None,
                        subdir: str = None) -> Result[MincAtom]:
    """
    Resample an image, ignoring mask/labels
    ...
    new_name_wo_ext -- string indicating a user specified file name (without extension)
    """

    if not subdir:
        subdir = 'resampled'

    if not new_name_wo_ext:
        # FIXME the path to `outf` is wrong.  For instance, resampling a mask file ends up in the initial model
        # FIXME directory instead of in the resampled directory.
        # FIXME At the same time, `like.newname(...)` doesn't work either, for similar reasons.
        # FIXME Not clear if there's a general "automagic" fix for this.
        outf = img.newname(name=xfm.filename_wo_ext + '-resampled', subdir=subdir)
    else:
        # we have the output filename without extension. This should replace the entire
        # current "base" of the filename. 
        outf = img.newname(name=new_name_wo_ext, subdir=subdir)

    stage = CmdStage(
        inputs=(xfm, like, img),
        outputs=(outf,),
        cmd=['mincresample', '-clobber', '-2',
             '-transform %s' % xfm.path,
             '-like %s' % like.path,
             img.path, outf.path]
            + (['-' + interpolation.name] if interpolation else [])
            + list(extra_flags))
    stage.set_log_file(os.path.join(outf.pipeline_sub_dir,
                                    outf.output_sub_dir,
                                    "log",
                                    "mincresample_" + outf.filename_wo_ext + ".log"))
    return Result(stages=Stages([stage]), output=outf)


# TODO mincresample_simple could easily be replaced by a recursive call to mincresample
def mincresample(img: MincAtom,
                 xfm: XfmAtom,
                 like: MincAtom,
                 extra_flags: Tuple[str] = (),
                 interpolation: Interpolation = None,
                 new_name_wo_ext: str = None,
                 subdir: str = None) -> Result[MincAtom]:
    """
    ...
    new_name_wo_ext -- string indicating a user specified file name (without extension)
    subdir          -- string indicating which subdirectory to output the file in:
    
    >>> img  = MincAtom('/tmp/img_1.mnc')
    >>> like = MincAtom('/tmp/img_2.mnc')
    >>> xfm  = XfmAtom('/tmp/trans.xfm')
    >>> stages, resampled = mincresample(img=img, xfm=xfm, like=like)
    >>> [x.render() for x in stages]
    ['mincresample -clobber -2 -transform /tmp/trans.xfm -like /tmp/img_2.mnc /tmp/img_1.mnc /micehome/bdarwin/git/pydpiper/img_1/resampled/trans-resampled.mnc']
    """

    s = Stages()

    # don't scale label values:
    label_extra_flags = (extra_flags
                         + (('-keep_real_range',) if "-keep_real_range"
                                                    not in extra_flags else ()))

    if not subdir:
        subdir = 'resampled'

    # we need to get the filename without extension here in case we have
    # masks/labels associated with the input file. When that's the case,
    # we supply its name with "_mask" and "_labels" for which we need
    # to know what the main file will be resampled as
    if not new_name_wo_ext:
        new_name_wo_ext = xfm.filename_wo_ext + '-resampled'

    new_img = s.defer(mincresample_simple(img=img, xfm=xfm, like=like,
                                          extra_flags=extra_flags,
                                          new_name_wo_ext=new_name_wo_ext,
                                          subdir=subdir))
    new_img.mask = s.defer(mincresample_simple(img=img.mask, xfm=xfm, like=like,
                                               extra_flags=extra_flags,
                                               interpolation=Interpolation.nearest_neighbour,
                                               new_name_wo_ext=new_name_wo_ext + "_mask",
                                               subdir=subdir)) if img.mask is not None else None
    new_img.labels = s.defer(mincresample_simple(img=img.labels, xfm=xfm, like=like,
                                                 extra_flags=label_extra_flags,
                                                 interpolation=Interpolation.nearest_neighbour,
                                                 new_name_wo_ext=new_name_wo_ext + "_labels",
                                                 subdir=subdir)) if img.labels is not None else None

    # Note that new_img can't be used for anything until the mask/label files are also resampled.
    # This shouldn't create a problem with stage dependencies as long as masks/labels appear in inputs/outputs of CmdStages.
    # (If this isn't automatic, a relevant helper function would be trivial.)
    # TODO: can/should this be done semi-automatically? probably ...
    return Result(stages=s, output=new_img)


def xfmconcat(xfms: List[XfmAtom],
              name: str = None) -> Result[XfmAtom]:
    """
    >>> stages, xfm = xfmconcat([XfmAtom('/tmp/%s' % i, pipeline_sub_dir='/scratch') for i in ['t1.xfm', 't2.xfm']])
    >>> [s.render() for s in stages]
    ['xfmconcat /tmp/t1.xfm /tmp/t2.xfm /scratch/t1/concat_of_t1_and_t2.xfm']
    """
    if len(xfms) == 0:
        raise ValueError("`xfmconcat` arg `xfms` was empty (can't concat zero files)")
    elif len(xfms) == 1:
        return Result(stages=Stages(), output=xfms[0])
    else:
        if name:
            outf = xfms[0].newname(name=name, subdir="transforms")
        elif atoms_from_same_subject(xfms):
            # we can reduce the length of the concatenated filename, because we do not
            # need repeats of the base part of the filename
            commonprefix = os.path.commonprefix([xfm.filename_wo_ext for xfm in xfms])
            outf_name = commonprefix + "_concat"
            for xfm in xfms:
                # only add the part of each of the files that is not
                # captured by the common prefix
                outf_name += "_" + xfm.filename_wo_ext[len(commonprefix):]
            outf = xfms[0].newname(name=outf_name, subdir="transforms")
        else:
            outf = xfms[0].newname(name="concat_of_%s" % "_and_".join([x.filename_wo_ext for x in xfms]),
                                   subdir="transforms")
                   # could do names[1:] if dirname contains names[0]?
        stage = CmdStage(
            inputs=tuple(xfms), outputs=(outf,),
            cmd=shlex.split('xfmconcat -clobber %s %s' % (' '.join([x.path for x in xfms]), outf.path)))

        stage.set_log_file(os.path.join(outf.pipeline_sub_dir,
                                        outf.output_sub_dir,
                                        "log",
                                        "xfmconcat_" + outf.filename_wo_ext + ".log"))
        return Result(stages=Stages([stage]), output=outf)


#
#
# TODO: do we need this function at all??
#       the concept seems odd. It's a concat and a resample?
#
#
def concat_xfmhandlers(xfms: List[XfmHandler],
                       name: str = None,
                       interpolation: Optional[Interpolation] = None,  # remove or make `sinc` default?
                       extra_flags: Tuple[str] = ()) -> Result[XfmHandler]:
    """
    xfmconcat lifted to work on XfmHandlers instead of XfmAtoms.
    To avoid unnecessary resamplings, we will resample the source
    file of the first XfmHandler with the concated transform of all
    of them creating the resampled file for the output XfmHandler
    """
    s = Stages()
    t = s.defer(xfmconcat([t.xfm for t in xfms], name=name))
    res = s.defer(mincresample(img=xfms[0].source,
                               xfm=t,
                               like=xfms[-1].target,
                               interpolation=interpolation,
                               extra_flags=extra_flags))
    return Result(stages=s,
                  output=XfmHandler(source=xfms[0].source,
                                    target=xfms[-1].target,
                                    xfm=t,
                                    resampled=res))


def nu_estimate(src: MincAtom,
                resolution: float,
                mask: Optional[MincAtom] = None,
                subject_matter: Optional[str] = None) -> Result[MincAtom]:  # TODO: make subject_matter an Enum
    """
    subject_matter -- can be either "mousebrain" or "human". If "mousebrain", the 
                      distance parameter will be set to 8 times the resolution,
                      if "human" it will be 200 times.
    """
    out = src.newname_with_suffix("_nu_estimate", ext=".imp")

    # TODO finish dealing with masking as per lines 436-448 of the old LSQ6.py.  (note: the 'singlemask' option there is never used)
    # (currently we don't allow for a single mask or using the initial model's mask if the inputs don't have them)

    # there are at least one parameter that should vary with the resolution of the input 
    # files, and that is: -distance 
    # The defaults for human brains are: 200
    # for mouse brains we use          :   8
    distance_value = 150 * resolution
    if subject_matter is not None:
        if subject_matter == "mousebrain":
            distance_value = 8
        elif subject_matter == "human":
            distance_value = 200
        else:
            raise ValueError(
                "The value for subject_matter in nu_estimate should be either 'human' or 'mousebrain'. It is: '%s.'" % subject_matter)

    mask_for_nu_est = src.mask if src.mask else mask

    cmd = CmdStage(inputs=(src,) + ((mask_for_nu_est,) if mask_for_nu_est else ()),
                   outputs=(out,),
                   cmd=shlex.split(
                       "nu_estimate -clobber -iterations 100 -stop 0.001 -fwhm 0.15 -shrink 4 -lambda 5.0e-02")
                       + ["-distance", str(distance_value)] + (
                       ['-mask', mask_for_nu_est.path] if mask_for_nu_est else []) + [src.path, out.path])
    cmd.set_log_file(os.path.join(out.pipeline_sub_dir,
                                  out.output_sub_dir,
                                  "log",
                                  "nu_estimate_" + out.filename_wo_ext + ".log"))
    return Result(stages=Stages([cmd]), output=out)


def nu_evaluate(img: MincAtom,
                field: FileAtom) -> Result[MincAtom]:
    out = img.newname_with_suffix("_N")
    cmd = CmdStage(inputs=(img, field), outputs=(out,),
                   cmd=['nu_evaluate', '-clobber', '-mapping', field.path, img.path, out.path])
    cmd.set_log_file(os.path.join(out.pipeline_sub_dir,
                                  out.output_sub_dir,
                                  "log",
                                  "nu_evaluate_" + out.filename_wo_ext + ".log"))
    return Result(stages=Stages([cmd]), output=out)


def nu_correct(src: MincAtom,
               resolution: float,
               mask: Optional[MincAtom] = None,
               subject_matter: Optional[str] = None) -> Result[MincAtom]:
    s = Stages()
    return Result(stages=s, output=s.defer(nu_evaluate(src, s.defer(nu_estimate(src, resolution,
                                                                                mask=mask,
                                                                                subject_matter=subject_matter)))))


class INormalizeMethod(AutoEnum):
    # the unusual capitalization here is a hack to avoid having to implement __repr__
    # when passing these to the shell - don't change it!
    ratioOfMeans = ratioOfMedians = meanOfRatios = meanOfLogRatios = medianOfRatios = ()


INormalizeConf = NamedTuple('INormalizeConf',
                            [('const', int),
                             ('method', INormalizeMethod)])

default_inormalize_conf = INormalizeConf(const=1000, method=INormalizeMethod.ratioOfMedians)


# NB the inormalize default is actually '-medianOfRatios'
# FIXME how do we want to deal with situations where our defaults differ from the tools' defaults,
# and in the latter case should we output the actual settings if the user doesn't explicitly set them?
# should we put defaults into the classes or populate with None (which will often raise an error if disallowed)
# and create default objects?

def inormalize(src: MincAtom,
               conf: INormalizeConf,
               mask: Optional[MincAtom] = None) -> Result[MincAtom]:
    """
    Note: if a mask is specified through the "mask" parameter, it will have
    precedence over the mask that might be associated with the 
    src file.
    """
    out = src.newname_with_suffix('_I')

    mask_for_inormalize = mask or src.mask

    cmd = CmdStage(inputs=(src,), outputs=(out,),
                   cmd=shlex.split('inormalize -clobber -const %s -%s' % (conf.const, conf.method.name))
                       + (['-mask', mask_for_inormalize.path] if mask_for_inormalize else [])
                       + [src.path, out.path])
    cmd.set_log_file(os.path.join(out.pipeline_sub_dir,
                                  out.output_sub_dir,
                                  "log",
                                  "inormalize_" + out.filename_wo_ext + ".log"))
    return Result(stages=Stages([cmd]), output=out)


def xfmaverage(xfms: List[XfmAtom],
               output_dir: str = None,
               output_filename_wo_ext: str = None) -> Result[XfmAtom]:
    """
    Currently the only thing that this function can deal with is
    a list of XfmAtoms that all come from the same subject.
    It will use the newname_with_fn() to create the output XfmAtom
    to ensure that the pipeline_sub_dir attributes etc. are set
    correctly
    """
    if len(xfms) == 0:
        raise ValueError("`xfmaverage` arg `xfms` is empty (can't average zero files)")

    # if all transformations belong to the same subject, we should place the average
    # transformation in the directories belonging to that file
    all_from_same_sub = atoms_from_same_subject(xfms)

    # TODO: the path here is probably not right...
    if not output_filename_wo_ext:
        output_filename_wo_ext= "average_xfm"
    if all_from_same_sub:
        outf = xfms[0].newname(name=output_filename_wo_ext, subdir="transforms", ext=".xfm")
    else:
        # it's actually not very clear at this point what to do... will this ever
        # be used? Hope not... :-)
        raise NotImplementedError("Aha... you are trying to use xfmaverage on "
                                  "a list of transformations that are located "
                                  "in several directories. We currently have not "
                                  "implemented the code to deal with this.")
    #else:
    #    outf = XfmAtom(name=os.path.join(output_dir, 'transforms', output_filename), orig_name=None)

    stage = CmdStage(inputs=tuple(xfms), outputs=(outf,),
                     cmd=["xfmavg", "-clobber"] + sorted([x.path for x in xfms]) + [outf.path])
    stage.set_log_file(os.path.join(outf.pipeline_sub_dir,
                                    outf.output_sub_dir,
                                    "log",
                                    "xfmavg_" + outf.filename_wo_ext + ".log"))
    return Result(stages=Stages([stage]), output=outf)


def xfminvert(xfm: XfmAtom) -> Result[XfmAtom]:
    inv_xfm = xfm.newname_with_suffix('_inverted')  # type: XfmAtom
    s = CmdStage(inputs=(xfm,), outputs=(inv_xfm,),
                 cmd=['xfminvert', '-clobber', xfm.path, inv_xfm.path])
    s.set_log_file(os.path.join(inv_xfm.pipeline_sub_dir,
                                inv_xfm.output_sub_dir,
                                "log",
                                "xfminvert_" + inv_xfm.filename_wo_ext + ".log"))
    return Result(stages=Stages([s]), output=inv_xfm)


def invert_xfmhandler(xfm: XfmHandler) -> Result[XfmHandler]:
    """
    xfminvert lifted to work on XfmHandlers instead of MincAtoms
    """
    # TODO: are we potentially creating some issues here... consider the following case:
    # TODO: minctracc src.mnc final_nlin.mnc some_trans.xfm
    # TODO: the xfmhandler will have:
    # TODO: source = src.mnc
    # TODO: target = final_nlin.mnc
    # TODO: xfm    = some_trans.xfm
    #
    # TODO: however, when we generate the inverse of this transform, we might want the
    # TODO: result to be:
    # TODO: source = src_resampled_to_final_nlin.mnc
    # TODO: target = src.mnc
    # TODO: xfm    = some_trans_inverted.xfm
    #
    # TODO: instead of have "final_nlin.mnc" as the source for this XfmHandler... we might
    # TODO: run into issues here...
    s = Stages()
    inv_xfm = s.defer(xfminvert(xfm.xfm))  # type: XfmAtom
    return Result(stages=s,
                  output=XfmHandler(xfm=inv_xfm,
                                    source=xfm.target, target=xfm.source, resampled=None))


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


# FIXME consistently require that masks are explicitely added to inputs array (or not?)
def rotational_minctracc(source: MincAtom,
                         target: MincAtom,
                         conf: RotationalMinctraccConf,
                         resolution: float,
                         mask: MincAtom = None,  # TODO: add mask, resample_source to conf ??
                         resample_source: bool = False,
                         output_name_wo_ext: str = None) -> Result[XfmHandler]:
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
               resolution by 20.
        
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
    blur_stepsize = resolution * conf.blur_factor
    resample_stepsize = resolution * conf.resample_step_factor
    registration_stepsize = resolution * conf.registration_step_factor
    w_translation_stepsize = resolution * conf.w_translations_factor

    # blur input files
    blurred_src = s.defer(mincblur(source, blur_stepsize))
    blurred_dest = s.defer(mincblur(target, blur_stepsize))

    if not output_name_wo_ext:
        output_name_wo_ext = "%s_rotational_minctracc_to_%s" % (source.filename_wo_ext, target.filename_wo_ext)

    out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                                        output_name_wo_ext + ".xfm"),
                      pipeline_sub_dir=source.pipeline_sub_dir,
                      output_sub_dir=source.output_sub_dir)

    # use the target mask if around, or overwrite this mask with the mask
    # that is explicitly provided:
    # TODO: shouldn't this use the mask if it's provided rather than the target mask?
    mask_for_command = target.mask if target.mask else mask
    cmd = CmdStage(inputs=(blurred_src, blurred_dest) + cast(tuple, ((mask_for_command,) if mask_for_command else ())),
                   # if-expression not recognized as a tuple; see mypy/issues/622
                   outputs=(out_xfm,),
                   cmd=["rotational_minctracc.py",
                        "-t", conf.temp_dir,  # TODO don't use a given option if not supplied (i.e., None)
                        "-w", ','.join([str(w_translation_stepsize)]*3),
                        "-s", str(resample_stepsize),
                        "-g", str(registration_stepsize),
                        "-r", str(conf.rotational_range),
                        "-i", str(conf.rotational_interval),
                        "--simplex", str(resolution * 20),
                        blurred_src.path,
                        blurred_dest.path,
                        out_xfm.path,
                        "/dev/null"] + (['-m', mask_for_command.path] if mask_for_command else []))
    cmd.set_log_file(os.path.join(out_xfm.pipeline_sub_dir,
                                  out_xfm.output_sub_dir,
                                  "log",
                                  "rotation_minctracc_" + out_xfm.filename_wo_ext + ".log"))
    s.update(Stages([cmd]))

    resampled = (s.defer(mincresample(img=source, xfm=out_xfm, like=target,
                                      interpolation=Interpolation.sinc))
                 if resample_source else None)

    # This is a point where it's likely for the input file not to have a mask associated with it
    # (given that this is the lsq6 alignment). If the target has a mask, or there was a mask
    # provided, we can add it to the resampled file.
    if resampled:
        if not resampled.mask and mask_for_command:
            resampled.mask = mask_for_command

    return Result(stages=s,
                  output=XfmHandler(source=source,
                                    target=target,
                                    xfm=out_xfm,
                                    resampled=resampled))


# should we even keep (hence render) the defaults which are the same as minctracc's?
# does this even get used in the pipeline?  LSQ6/12 get their own
# protocols, and many settings here are the minctracc default
def default_linear_minctracc_conf(transform_type: LinearTransType) -> LinearMinctraccConf:
    return LinearMinctraccConf(simplex=1,  # TODO simplex=1 -> simplex_factor=20?
                               transform_type=transform_type,
                               tolerance=0.0001,
                               w_scales=(0.02, 0.02, 0.02),
                               w_shear=(0.02, 0.02, 0.02),
                               w_rotations=(0.0174533, 0.0174533, 0.0174533),
                               w_translations=(1.0, 1.0, 1.0))


default_lsq6_minctracc_conf, default_lsq12_minctracc_conf = \
    [default_linear_minctracc_conf(x) for x in (LinearTransType.lsq6, LinearTransType.lsq12)]

# TODO: I'm not sure about these defaults.. they should be
# based on the resolution of the input files. This will 
# somewhat work for mouse brains, but is a very coarse 
# registration. What's its purpose?
# (Could also use a None here and/or make this explode via a property if accessed;
# see XfmHandler `resampled` field)
_step_size = 0.5
_step_sizes = (_step_size, _step_size, _step_size)  # type: R3
_diam = 3 * _step_size
_lattice_diameter = (_diam, _diam, _diam)  # type: R3
default_nonlinear_minctracc_conf = NonlinearMinctraccConf(
    # step_sizes=_step_sizes,
    # blur_resolution=_step_size,
    # use_masks=True,
    iterations=40,
    similarity=0.8,
    use_simplex=True,
    stiffness=0.98,
    weight=0.8,
    objective=Objective.corrcoeff,
    lattice_diameter=_lattice_diameter,
    sub_lattice=6,
    max_def_magnitude=None)


# TODO: move to utils?
def space_sep(xs) -> List[str]:
    # return ' '.join(map(str,x))
    return [str(x) for x in xs]

def parse_bool(s):
    _s = s.lower()
    if _s == "true":
        return True
    elif _s == "false":
        return False
    else:
        raise ParseError("Bad boolean: %s" % s)

def parse_n(p, n):
    def f(s):
        result = parse_many(p)(s)
        if len(result) == n:
            return result
        else:
            raise ParseError("Wrong number of values")
    return f

def parse_minctracc_lin_protocol_file(filename : str, minctracc_conf=default_linear_minctracc_conf) \
        -> MultilevelMinctraccConf:
    with open(filename, 'r') as f:
        return parse_minctracc_lin_protocol(csv.read(f), minctracc_conf)

def parse_minctracc_lin_protocol(f, minctracc_conf : LinearMinctraccConf = default_linear_minctracc_conf) \
        -> MultilevelMinctraccConf:
    """Use the resulting list to `.replace` the default values.  Needs to return a full MinctraccConf
    in order to encode blur and step information."""

    # parsers to use for each row of the protocol file
    parsers = {"blur"               : float,
               "step"               : float,
               "gradient"           : parse_bool,
               "simplex"            : float,
               "transform_type"     : LinearTransType,
               "tolerance"          : float,
               "w_rotations"        : float,
               "w_translations"     : float,
               "w_scales"           : float,
               "w_shear"            : float}

    # mapping from protocol file names to Python field names of the MinctraccConf fields
    #names = {"blur" : "blur_resolution",
    #         "step" : "step",
    #         "gradient" : "gradient"} and the other ones go in the linear part of the configuration
    params = list(parsers.keys())

    # build a mapping from (Python, not file) field names to a list of values (one for each generation)
    d = {}
    for l in f:
        k, *vs = l
        if k not in params:
            raise ParseError("Unrecognized parameter: %s" % k)
        else:
            #new_k = names[k]
            if k in d:
                raise ParseError("Duplicate key: %s" % k)
            else:
                d[k] = [parsers[k](v) for v in vs]

    # some error checking ...
    if not all_equal(d.values(), by=len):
        raise ParseError("Invalid minctracc configuration: all params must have the same number of generations.")
    if len(d) == 0:
        raise ParseError("Empty file ...")   # TODO should this really be an error?
    if "memory_required" in d:
        print("Warning: don't currently use the memory ...")  # doesn't have to be same length -> can crash code below
        del d["memory_required"]

    vs = list(d.values())
    l = len(vs[0])

    # LinearMinctraccConf = NamedTuple("LinearMinctraccConf",
    #                              [("simplex", float),
    #                               ("transform_type", Optional[LinearTransType]),
    #                               ("tolerance", float),
    #                               ("w_rotations", R3),
    #                               ("w_translations", R3),
    #                               ("w_scales", R3),
    #                               ("w_shear", R3)])
    
    # convert a mapping of options to _single_ values to a single-generation minctracc configuration object:
    def convert_single_gen(single_gen_params) -> LinearMinctraccConf:  # TODO name this better ...
        # TODO check for/catch IndexError ... a bit hard to use zip since some params may not be defined ...
        linear_attrs = { k : v for k, v in single_gen_params.items() if k not in ('blur_resolution', 'step', 'gradient')}

        linear_conf  =  default_lsq6_conf.replace(**linear_attrs)
        return MinctraccConf(blur_resolution=single_gen_params["blur"],
                             use_gradient=single_gen_params["gradient"],
                             step_sizes=single_gen_params["step"],
                             use_masks=True, #FIXME
                             linear_conf=linear_conf,
                             nonlinear_conf=None)
                             
    return MultilevelMinctraccConf([convert_single_gen({ key : vs[j] for key, vs in d.items() }) for j in range(l)])

# TODO: add memory estimation hook
def minctracc(source: MincAtom,
              target: MincAtom,
              conf: MinctraccConf,
              transform: Optional[XfmAtom] = None,
              transform_name_wo_ext: Optional[str] = None,
              transform_info: Optional[List[str]] = None,
              generation: Optional[int] = None,
              resample_source: bool = False) -> Result[XfmHandler]:
    """
    source -- "raw", blurring will happen here
    target -- "raw", blurring will happen here
    conf
    transform
    transform_name_wo_ext -- to use for the output transformation (without the extension)
    generation            -- if provided, the transformation name will be:
                             source.filename_wo_ext + "_minctracc_nlin-" + generation
    resample_source       -- whether or not to resample the source file 
    
    
    minctracc functionality:
    
    LSQ6 -- conf.nonlinear is None and conf.linear.transform_type == "lsq6"
    For the 6 parameter alignment we have 
    
    LSQ12 -- conf.nonlinear is None and conf.linear.transform_type == "lsq12"
    
    NLIN -- conf.linear is None and conf.nonlinear is not None
    """

    s = Stages()

    lin_conf = conf.linear_conf  # type: LinearMinctraccConf
    nlin_conf = conf.nonlinear_conf  # type: NonlinearMinctraccConf

    if lin_conf is None and nlin_conf is None:
        raise ValueError("minctracc: no linear or nonlinear configuration specified")

    if lin_conf is not None and lin_conf.transform_type not in LinearTransType.__members__:
        raise ValueError("minctracc: invalid transform type %s" % lin_conf.transform_type)
    # TODO: probably not needed since we're using an enum
    if transform_name_wo_ext:
        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                                            "%s.xfm" % (transform_name_wo_ext)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)
    # the generation provided can be 0 (zero), but that's a proper generation,
    # so we should explicitly test for "is not None" here.
    elif generation is not None:
        if lin_conf:
            trans_type = lin_conf.transform_type
        else:
            trans_type = "nlin"

        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                                            "%s_minctracc_to_%s_%s_%s.xfm" %
                                            (source.filename_wo_ext,
                                             target.filename_wo_ext,
                                             trans_type,
                                             generation)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)
    else:
        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                                            "%s_minctracc_to_%s.xfm" % (
                                            source.filename_wo_ext, target.filename_wo_ext)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)

    source_for_minctracc = source
    target_for_minctracc = target
    if conf.blur_resolution is not None:
        source_for_minctracc = s.defer(mincblur(source, conf.blur_resolution,
                                                gradient=conf.use_gradient))
        target_for_minctracc = s.defer(mincblur(target, conf.blur_resolution,
                                                gradient=conf.use_gradient))

    # TODO: FIXME: currently broken in the presence of None fields; should fall back to our defaults
    # and/or minctracc's own defaults.
    stage = CmdStage(cmd=['minctracc', '-clobber', '-debug', '-xcorr']  # TODO: remove hard-coded `xcorr`?
                         + (['-transformation', transform.path] if transform
                            else (transform_info if transform_info else ["-identity"]))
                         + (['-' + lin_conf.transform_type]
                            if lin_conf and lin_conf.transform_type else [])
                         + (['-use_simplex']
                            if nlin_conf and nlin_conf.use_simplex is not None else [])
                         + (['-step'] + space_sep(conf.step_sizes))
                         + ((['-simplex', str(lin_conf.simplex)]
                             + ['-tol', str(lin_conf.tolerance)]
                             + ['-w_shear'] + space_sep(lin_conf.w_shear)
                             + ['-w_scales'] + space_sep(lin_conf.w_scales)
                             + ['-w_rotations'] + space_sep(lin_conf.w_rotations)
                             + ['-w_translations'] + space_sep(lin_conf.w_translations))
                            if lin_conf is not None else [])
                         + ((['-iterations', str(nlin_conf.iterations)]
                             + ['-similarity_cost_ratio', str(nlin_conf.similarity)]
                             + ['-sub_lattice', str(nlin_conf.sub_lattice)]
                             + ['-lattice_diameter'] + space_sep(nlin_conf.lattice_diameter))
                            if nlin_conf is not None else [])
                         + (['-nonlinear %s' % (nlin_conf.objective if nlin_conf.objective else '')]
                            if nlin_conf else [])
                         + (['-source_mask', source.mask.path]
                            if source.mask and conf.use_masks else [])
                         + (['-model_mask', target.mask.path]
                            if target.mask and conf.use_masks else [])
                         + ([source_for_minctracc.path, target_for_minctracc.path, out_xfm.path]),
                     inputs=(source_for_minctracc, target_for_minctracc) +
                            ((transform,) if transform else ()) +
                            ((source.mask,) if source.mask and conf.use_masks else ()) +
                            ((target.mask,) if target.mask and conf.use_masks else ()),
                     outputs=(out_xfm,))
    stage.set_log_file(os.path.join(out_xfm.pipeline_sub_dir,
                                    out_xfm.output_sub_dir,
                                    "log",
                                    "minctracc_" + out_xfm.filename_wo_ext + ".log"))
    s.add(stage)

    # note accessing a None resampled field from an XfmHandler is an error (by property magic),
    # so the possibility of `resampled` being None isn't as annoying as it looks
    resampled = (s.defer(mincresample(img=source, xfm=out_xfm, like=target,
                                      interpolation=Interpolation.sinc))
                 if resample_source else None)

    return Result(stages=s,
                  output=XfmHandler(source=source,
                                    target=target,
                                    xfm=out_xfm,
                                    resampled=resampled))


SimilarityMetricConf = NamedTuple('SimilarityMetricConf',
                                  [("metric", str),
                                   ("weight", float),
                                   ("radius_or_bins", float),
                                   ("use_gradient_image", bool)])


default_similarity_metric_conf = SimilarityMetricConf(
    metric="CC",
    weight=1.0,
    radius_or_bins=3,
    use_gradient_image=False)

MincANTSConf = NamedTuple("MincANTSConf",
                          [("file_resolution", float),
                           ("iterations", str),
                           ("transformation_model", str),
                           ("regularization", str),
                           ("use_mask", bool),
                           ("sim_metric_confs", List[SimilarityMetricConf])])

class MultilevelMincANTSConf(object):
    def __init__(self, confs: List[MincANTSConf]):
        self.confs = confs

# we don't supply a resolution default here because it's preferable
# to take resolution from initial target instead
mincANTS_default_conf = MincANTSConf(
    iterations="100x100x100x150",
    transformation_model="'Syn[0.1]'",
    regularization="'Gauss[2,1]'",
    use_mask=True,
    file_resolution=None,
    sim_metric_confs=[default_similarity_metric_conf,
                      default_similarity_metric_conf.replace(use_gradient_image=True)])  # type: MincANTSConf

def parse_many(parser, sep=','):
    def f(st):
        return tuple(parser(s) for s in st.split(sep))
    return f

def parse_nullable(parser):
    def p(st):
        if st == "None":
            return None
        else:
            return parser(st)
    return p

class ParseError(ValueError): pass

def parse_mincANTS_protocol_file(f, mincANTS_conf=mincANTS_default_conf) -> MultilevelMincANTSConf:
    """Use the resulting list to `.replace` the default values."""

    # parsers to use for each row of the protocol file
    parsers = {"blur"               : parse_many(parse_nullable(float)),
               "gradient"           : parse_many(parse_bool),
               "similarity_metric"  : parse_many(str),
               "weight"             : parse_many(float),
               "radius_or_histo"    : parse_many(float),
               "transformation"     : str,
               "regularization"     : str,
               "iterations"         : str,
               "useMask"            : bool,
               "memoryRequired"     : float}

    # mapping from protocol file names to Python field names of the mincANTS and similarity metric configurations
    names = {"blur" : "blur", # not needed since blur is deleted...
             "gradient" : "use_gradient_image",
             "similarity_metric" : "metric",
             "weight" : "weight",
             "radius_or_histo" : "radius_or_bins",
             "transformation" : "transformation_model",
             "regularization" : "regularization",
             "iterations" : "iterations",
             "useMask" : "use_mask",
             "memoryRequired" : "memory_required"}
    params = list(parsers.keys())

    # build a mapping from (Python, not file) field names to a list of values (one for each generation)
    d = {}
    for l in f:
        k, *vs = l
        if k not in params:
            raise ParseError("Unrecognized parameter: %s" % k)
        else:
            new_k = names[k]
            if new_k in d:
                raise ParseError("Duplicate key: %s" % k)
            else:
                d[new_k] = [parsers[k](v) for v in vs]

    # some error checking ...
    if not all_equal(d.values(), by=len):
        raise ParseError("Invalid mincANTS configuration: all params must have the same number of generations.")
    if len(d) == 0:
        raise ParseError("Empty file ...")   # TODO should this really be an error?
    if "blur" in d:
        print("Warning: no longer using `blur` even though it's specified in the protocol ...")
        # TODO should be a logger.warning, not a print
        del d["blur"]
    if "memory_required" in d:
        print("Warning: don't currently use the memory ...")  # doesn't have to be same length -> can crash code below
        del d["memory_required"]

    vs = list(d.values())
    l = len(vs[0])

    # convert a mapping of options to _single_ values to a single-generation mincANTS configuration object:
    def convert_single_gen(single_gen_params) -> MincANTSConf:  # TODO name this better ...
        # TODO check for/catch IndexError ... a bit hard to use zip since some params may not be defined ...
        sim_metric_names = {"use_gradient_image", "metric", "weight", "radius_or_bins"}
        # TODO duplication; e.g., parsers = sim_metric_parsers U <...>
        sim_metric_params = {k : v for k, v in single_gen_params.items() if k in sim_metric_names}
        other_attrs       = {k : v for k, v in single_gen_params.items() if k not in sim_metric_names}
        if len(sim_metric_params) > 0:
            sim_metric_values = list(sim_metric_params.values())
            if not all_equal(sim_metric_values, by=len):
                raise ParseError("All parts of the objective function specification must be the same length ...")
            sim_metric_params = [{ k : v[j] for k, v in sim_metric_params.items() } for j in range(len(sim_metric_values[0]))]
            # TODO could warn here if a given param is missing from a given metric specification
            sim_metric_confs = [default_similarity_metric_conf.replace(**s) for s in sim_metric_params]
        else:
            sim_metric_confs = []

        return mincANTS_default_conf.replace(sim_metric_confs=sim_metric_confs,
                                             #resolution=NotImplemented, #ugh...don't know this yet
                                             **other_attrs)
    return MultilevelMincANTSConf([convert_single_gen({ key : vs[j] for key, vs in d.items() }) for j in range(l)])

def all_equal(xs, by=lambda x: x):
    return len(set((by(x) for x in xs))) == 1

def mincANTS(source: MincAtom,
             target: MincAtom,
             conf: MincANTSConf,
             transform_name_wo_ext: str = None,
             generation: int = None,
             resample_source: bool = False) -> Result[XfmHandler]:
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
        name = os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                            "%s.xfm" % (transform_name_wo_ext))
    elif generation != None:
        name = os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                            "%s_mincANTS_nlin-%s.xfm" % (source.filename_wo_ext, generation))
    else:
        name = os.path.join(source.pipeline_sub_dir, source.output_sub_dir, 'transforms',
                            "%s_mincANTS_to_%s.xfm" % (source.filename_wo_ext, target.filename_wo_ext))
    out_xfm = XfmAtom(name=name, pipeline_sub_dir=source.pipeline_sub_dir, output_sub_dir=source.output_sub_dir)

    similarity_cmds = []       # type: List[str]
    similarity_inputs = set()  # type: Set[MincAtom]
    # TODO: similarity_inputs should be a set, but `MincAtom`s aren't hashable
    for sim_metric_conf in conf.sim_metric_confs:
        if conf.file_resolution is not None and sim_metric_conf.use_gradient_image:
            src = s.defer(mincblur(source, fwhm=conf.file_resolution,
                                   gradient=sim_metric_conf.use_gradient_image))
            dest = s.defer(mincblur(target, fwhm=conf.file_resolution,
                                    gradient=sim_metric_conf.use_gradient_image))
        elif conf.file_resolution is None and sim_metric_conf.use_gradient_image:
            # the file resolution is not set, however we want to use the gradients
            # for this similarity metric...
            raise ValueError("A similarity metric in the mincANTS configuration "
                            "wants to use the gradients, but the file resolution for the "
                            "configuration has not been set.")
        else:
            src = source
            dest = target
        similarity_inputs.add(src)
        similarity_inputs.add(dest)
        inner = ','.join([src.path, dest.path,
                          str(sim_metric_conf.weight), str(sim_metric_conf.radius_or_bins)])
        subcmd = "'" + "".join([sim_metric_conf.metric, '[', inner, ']']) + "'"
        similarity_cmds.extend(["-m", subcmd])
    stage = CmdStage(
        inputs=(source, target) + tuple(similarity_inputs) + cast(tuple, ((target.mask,) if target.mask else ())),
        # need to cast to tuple due to mypy bug; see mypy/issues/622
        outputs=(out_xfm,),
        cmd=['mincANTS', '3',
             '--number-of-affine-iterations', '0']
            + similarity_cmds
            + ['-t', conf.transformation_model,
               '-r', conf.regularization,
               '-i', conf.iterations,
               '-o', out_xfm.path]
            + (['-x', target.mask.path] if conf.use_mask and target.mask else []))
    stage.set_log_file(os.path.join(out_xfm.pipeline_sub_dir,
                                    out_xfm.output_sub_dir,
                                    "log",
                                    "mincANTS_" + out_xfm.filename_wo_ext + ".log"))
    s.add(stage)
    resampled = (s.defer(mincresample(img=source, xfm=out_xfm, like=target,
                                      interpolation=Interpolation.sinc))
                 if resample_source else None)  # type: Optional[MincAtom]
    return Result(stages=s,
                  output=XfmHandler(source=source,
                                    target=target,
                                    xfm=out_xfm,
                                    resampled=resampled))

T = TypeVar('T')


class WithAvgImgs(Generic[T]):
    def __init__(self,
                 output: T,
                 avg_imgs: List[MincAtom],
                 avg_img: MincAtom) -> None:
        self.output = output
        self.avg_imgs = avg_imgs
        self.avg_img = avg_img


def minctracc_NLIN_build_model(imgs: List[MincAtom],
                               initial_target: MincAtom,
                               conf: MultilevelMinctraccConf,
                               nlin_dir: str) -> Result[WithAvgImgs[List[XfmHandler]]]:  # TODO: add resolution parameter:
    if len(conf.confs) == 0:
        raise ValueError("No configurations supplied ...")
    s = Stages()
    avg = initial_target
    avg_imgs = []
    for i, conf in enumerate(conf.confs, start=1):
        xfms = [s.defer(minctracc(source=img, target=avg, conf=conf, generation=i, resample_source=True))
                for img in imgs]
        avg = s.defer(mincaverage([xfm.resampled for xfm in xfms], name_wo_ext='nlin-%d' % i, output_dir=nlin_dir))
        avg_imgs.append(avg)
    return Result(stages=s, output=WithAvgImgs(output=xfms, avg_img=avg, avg_imgs=avg_imgs))


def mincANTS_NLIN_build_model(imgs: List[MincAtom],
                              initial_target: MincAtom,
                              conf: MultilevelMincANTSConf,
                              nlin_dir: str,
                              mincaverage = mincaverage) -> Result[WithAvgImgs[List[XfmHandler]]]:
    """
    This functions runs a hierarchical mincANTS registration on the input
    images (imgs) creating an unbiased average.
    The mincANTS configuration `confs` that is passed in should be
    a list of configurations for each of the levels/generations.
    After each round of registrations, an average is created out of the 
    resampled input files, which is then used as the target for the next
    round of registrations. 
    """
    if len(conf.confs) == 0:
        raise ValueError("No configurations supplied ...")
    s = Stages()
    avg = initial_target
    avg_imgs = []  # type: List[MincAtom]
    for i, conf in enumerate(conf.confs, start=1):
        xfms = [s.defer(mincANTS(source=img, target=avg, conf=conf, generation=i, resample_source=True))
                for img in imgs]
        avg = s.defer(mincaverage([xfm.resampled for xfm in xfms], name_wo_ext='nlin-%d' % i, output_dir=nlin_dir))
        avg_imgs.append(avg)
    return Result(stages=s, output=WithAvgImgs(output=xfms, avg_img=avg, avg_imgs=avg_imgs))

# some stuff for the registration chain.
# The Subject class moved here since `intrasubject_registrations` was also here.


def LSQ12_mincANTS_nlin(source: MincAtom,
                        target: MincAtom,
                        linear_conf: MinctraccConf,
                        nlin_conf: MincANTSConf):
    """
    Runs a 12 parameter registration followed by a non linear mincANTS registration
    from the source to the target. This is a *non* model building function, which
    can be used for intra-subject registrations in the registration_chain, or in
    MAGeT for instance
    """
    # TODO: currently this only works for mincANTS for the non linear part

    s = Stages()
    # we need to resample the source file in this case, because that will
    # be the input for the non linear stage
    lsq12_transform_handler = s.defer(multilevel_minctracc(source,
                                                           target,
                                                           conf=linear_conf,
                                                           resample_input=True))

    nlin_transform_handler = s.defer(mincANTS(source=lsq12_transform_handler.resampled,
                                              target=target,
                                              conf=nlin_conf,
                                              resample_source=True))

    full_transform = s.defer(concat_xfmhandlers([lsq12_transform_handler,
                                                 nlin_transform_handler],
                                                name=source.filename_wo_ext + "_to_" +
                                                target.filename_wo_ext + "_lsq12_mincANTS_nlin"))
    return Result(stages=s, output=full_transform)


V = TypeVar('V')


class Subject(Generic[V]):
    """
    A Subject contains the intersubject_registration_time_pt and a dictionary
    that maps timepoints to scans/data of type `V` related to this Subject.
    (Here V could be - for instance - str, FileAtom/MincAtom or XfmHandler).
    """

    def __init__(self,
                 intersubject_registration_time_pt: int,
                 time_pt_dict: Optional[Dict[int, V]] = None) -> None:
        # TODO: change the time_pt datatype to decimal or rational to allow, e.g., 18.5?
        self.intersubject_registration_time_pt = intersubject_registration_time_pt  # type: int
        self.time_pt_dict = time_pt_dict or dict()  # type: Dict[int, V]

    # compare by fields, not pointer
    def __eq__(self, other) -> bool:
        return (self is other or
                (self.__class__ == other.__class__
                 and self.intersubject_registration_time_pt == other.intersubject_registration_time_pt
                 and self.time_pt_dict == other.time_pt_dict))

    # ugh; also, should this be type(self) == ... ?

    # TODO: change name? This might not be an 'image'
    @property
    def intersubject_registration_image(self) -> V:
        return self.time_pt_dict[self.intersubject_registration_time_pt]

    def __repr__(self) -> str:
        return ("Subject(inter_sub_time_pt: %s, time_pt_dict keys: %s ... (values not shown))"
                % (self.intersubject_registration_time_pt, self.time_pt_dict.keys()))


def intrasubject_registrations(subj: Subject,
                               linear_conf: MinctraccConf,
                               nlin_conf: MincANTSConf) \
        -> Result[Tuple[List[Tuple[int, XfmHandler]], int]]:
    """
    
    subj -- Subject (has a intersubject_registration_time_pt and a time_pt_dict 
            that maps timepoints to individual subjects
    
    Returns a dictionary mapping a time point to a transformation
    {time_pt_1 : transform_from_1_to_2, ..., time_pt_n-1 : transform_from_n-1_to_n};
    note this is one element smaller than the number of time points.
    """
    # TODO: somehow the output of this function should provide us with
    # easy access from a MincAtom to an XfmHandler from time_pt N to N+1 
    # either here or in the chain() function

    s = Stages()
    timepts = sorted(subj.time_pt_dict.items())  # type: List[Tuple[int, MincAtom]]
    timepts_indices = [index for index, _subj_atom in timepts]  # type: List[int]
    # we need to find the index of the common time point and for that we
    # should only look at the first element of the tuples stored in timepts
    index_of_common_time_pt = timepts_indices.index(subj.intersubject_registration_time_pt)  # type: int

    time_pt_to_xfms = [(timepts_indices[source_index],
                        s.defer(LSQ12_mincANTS_nlin(source=src[1],
                                                    target=dest[1],
                                                    linear_conf=linear_conf,
                                                    nlin_conf=nlin_conf)))
                       for source_index, (src, dest) in enumerate(pairs(timepts))]
    return Result(stages=s, output=(time_pt_to_xfms, index_of_common_time_pt))


# def multilevel_registration(source, target, registration_function, conf, curr_dir, transform=None):
#    ...


# TODO: this is very static right now, but I just want to get things running

_lin_conf_1 = LinearMinctraccConf(simplex=2.8,
                                  transform_type="lsq12",
                                  tolerance=0.0001,
                                  w_translations=(0.4,0.4,0.4),
                                  w_rotations=(0.0174533,0.0174533,0.0174533),
                                  w_scales=(0.02,0.02,0.02),
                                  w_shear=(0.02,0.02,0.02))

default_lsq12_multilevel_minctracc_level1 = MinctraccConf(step_sizes=(0.9,0.9,0.9),
                                                          blur_resolution=0.28,
                                                          use_masks=True,
                                                          use_gradient=False,
                                                          linear_conf=_lin_conf_1,
                                                          nonlinear_conf=None)

default_lsq12_multilevel_minctracc_level2 = MinctraccConf(step_sizes=(0.46,0.46,0.46),
                                                          blur_resolution=0.19,
                                                          use_masks=True,
                                                          use_gradient=True,
                                                          linear_conf=_lin_conf_1.replace(simplex=1.4),
                                                          nonlinear_conf=None)

default_lsq12_multilevel_minctracc_level3 = MinctraccConf(step_sizes=(0.3,0.3,0.3),
                                                          blur_resolution=0.14,
                                                          use_masks=True,
                                                          use_gradient=False,
                                                          linear_conf=_lin_conf_1.replace(simplex=0.9),
                                                          nonlinear_conf=None)
default_lsq12_multilevel_minctracc = MultilevelMinctraccConf([default_lsq12_multilevel_minctracc_level1,
                                                              default_lsq12_multilevel_minctracc_level2,
                                                              default_lsq12_multilevel_minctracc_level3])

def multilevel_minctracc(source: MincAtom,
                         target: MincAtom,
                         conf: MultilevelMinctraccConf,
                         transform: Optional[XfmAtom] = None,
                         transform_info: Optional[List[str]] = None,
                         resample_input: bool = False) -> Result[XfmHandler]:
    if len(conf.confs) == 0:  # not a "registration" at all; also, src_blur/target_blur will be undefined ...
        raise ValueError("No configurations supplied")
    s = Stages()
    last_resampled = None
    for idx, conf in enumerate(conf.confs):
        transform_handler = s.defer(minctracc(source, target, conf=conf,
                                              transform=transform,
                                              transform_info=transform_info if idx == 0 else None,
                                              generation=idx,
                                              resample_source=resample_input))
        transform = transform_handler.xfm
        # why potentially throw away the resampled image?
        last_resampled = transform_handler.resampled if resample_input else None
    return Result(stages=s,
                  output=XfmHandler(xfm=transform,
                                    source=source,
                                    target=target,
                                    resampled=last_resampled))


# """Multilevel registration of many images to a single target"""
# def multilevel_minctracc_all(sources, target, conf, resolutions, transforms=None):
#    p = Stages()
#    transforms = transforms or [None] * len(sources)
#    for res in resolutions:
#        ss_blurred = (p.add_stages(mincblur(s, res)) for s in sources)
#        t_blurred  = p.add_stages(mincblur(target, res))
#        transforms = [p.extract_stages(minctracc(s, t, conf, res, transform=t_blurred))
#                      for (s,t) in zip(ss_blurred, transforms)]
#    return Result(stages=p, output=transforms)


def multilevel_pairwise_minctracc(imgs: List[MincAtom],
                                  conf: MultilevelMinctraccConf,
                                  # transforms : List[] = None,
                                  max_pairs : int,
                                  # max_pairs doesn't even make sense for a non-pairwise MinctraccConf,
                                  # suggesting that the pairwise conf being a list of confs is inappropriate
                                  like: MincAtom = None,
                                  output_dir_for_avg: str = ".",
                                  output_name_for_avg: str = None) -> Result[WithAvgImgs[List[XfmHandler]]]:
    """Pairwise registration of all images.
    max_pairs - number of images to register each image against. (Currently we might register against one fewer.)

    If no output_name_for_avg is supplied, the output name will be:

    avg_{transform_type}                          -- in case all the configurations have the same transformation type
    ave_{transform_type_1}_..._{transform_type_n} -- otherwise
    """
    s = Stages()

    if max_pairs < 2:
        raise ValueError("must register at least two pairs")

    if len(imgs) < 2:
        raise ValueError("currently need at least two images")
        # otherwise 0 imgs passed to mincavg (could special-case)

    confs = conf.confs
    # the name of the average file that is produced by this function:
    if not output_name_for_avg:
        all_same_transform_type = True
        first_transform_type = confs[0].linear_conf.transform_type if confs[0].linear_conf else "nlin"
        alternate_name = "avg"
        for stage in confs:
            current_transform_type = stage.linear_conf.transform_type if stage.linear_conf else "nlin"
            if current_transform_type != first_transform_type:
                all_same_transform_type = False
            alternate_name += "_" + current_transform_type
        if all_same_transform_type:
            output_name_for_avg = "avg_" + first_transform_type
        else:
            output_name_for_avg = alternate_name

    final_avg = MincAtom(name=os.path.join(output_dir_for_avg, output_name_for_avg + ".mnc"),
                         pipeline_sub_dir=output_dir_for_avg)

    def avg_xfm_from(src_img     : MincAtom,
                     target_imgs : List[MincAtom]):
        """Compute xfm from src_img to each target img, average them, and resample along the result"""
        # TODO to save creation of lots of duplicate blurs, could use multilevel_minctracc_all,
        # being careful not to register the img against itself
        # FIXME: do something about the configuration, currently it all seems a bit broken...
        xfms = [s.defer(multilevel_minctracc(src_img, target_img,
                                             conf=default_lsq12_multilevel_minctracc))
                for target_img in target_imgs if src_img != target_img]  # TODO src_img.name != ....name ??

        avg_xfm = s.defer(xfmaverage([xfm.xfm for xfm in xfms],
                                     output_filename_wo_ext="%s_avg_lsq12" % src_img.filename_wo_ext))

        res = s.defer(mincresample(img=src_img,
                                   xfm=avg_xfm,
                                   like=like or src_img,
                                   interpolation=Interpolation.sinc))
        return XfmHandler(xfm=avg_xfm, source=src_img,
                          target=final_avg, resampled=res)  ##FIXME the None here borks things interface-wise ...
        # does putting `target = res` make sense? could a sum be used?

    if max_pairs is None or max_pairs >= len(imgs):
        avg_xfms = [avg_xfm_from(img, target_imgs=imgs) for img in imgs]
    else:
        # FIXME this will be the same across all runs of the program, but also across calls with the same inputs ...
        random.seed(tuple((img.path for img in imgs)))
        avg_xfms = [avg_xfm_from(img, target_imgs=random.sample(imgs, max_pairs)) for img in imgs]
                      # FIXME might use one fewer image than `max_pairs`...

    # FIXME the average computed in lsq12_pairwise is now rather redundant -- resolve by returning this one?
    final_avg = s.defer(mincaverage([xfm.resampled for xfm in avg_xfms], avg_file=final_avg))

    return Result(stages=s, output=WithAvgImgs(avg_imgs=[final_avg], avg_img=final_avg, output=avg_xfms))

# MultilevelMinctraccConf = NamedTuple('MultilevelMinctraccConf',
#  [#('resolution', float),   # TODO: used to choose step size...shouldn't be here
#   ('single_gen_confs', MinctraccConf) # list of minctracc confs for each generation; could fold res/transform_type into these ...
# ('transform_type', str)])
#  ])
# OR:


# TODO move LSQ12 stuff to an LSQ12 file?
# LSQ12_default_conf = MultilevelMinctraccConf(transform_type='lsq12', resolution = NotImplemented,
#                                              single_gen_confs = [])


""" Pairwise lsq12 registration, returning array of transforms and an average image
    Assumption: given that this is a pairwise registration, we assume that all the input
                files to this function have the same shape (i.e. resolution, and dimension sizes.
                This allows us to use any of the input files as the likefile for resampling if
                none is provided. """


# TODO all this does is call multilevel_pairwise_minctracc and then return an average; fold into that procedure?
# TODO eliminate/provide default val for resolutions, move resolutions into conf, finish conf ...
def lsq12_pairwise(imgs: List[MincAtom],
                   conf: MultilevelMinctraccConf,  # TODO: override transform_type field?
                   lsq12_conf: LSQ12Conf,
                   lsq12_dir: str,
                   like: MincAtom = None,
                   mincaverage = mincaverage) -> Result[WithAvgImgs[List[XfmHandler]]]:
    # conf.transform_type='-lsq12' # hack ... copy? or set external to lsq12 call ? might be better
    s = Stages()
    avgs_and_xfms = s.defer(multilevel_pairwise_minctracc(imgs=imgs, conf=conf, like=like,
                                                 output_dir_for_avg=lsq12_dir,
                                                 max_pairs=lsq12_conf.max_pairs))
    return Result(stages=s, output=avgs_and_xfms)


K = TypeVar('K')


def lsq12_pairwise_on_dictionaries(imgs: Dict[K, MincAtom],
                                   conf: LSQ12Conf,
                                   lsq12_dir: str,
                                   like: Optional[MincAtom] = None) \
        -> Result[WithAvgImgs[Dict[K, XfmHandler]]]:
    """ This is a direct copy of lsq12_pairwise, with the only difference being that
    it takes dictionaries as input for the imgs, and returns the xfmhandlers as
    dictionaries as well. This is necessary (amongst others) for the registration_chain
    as using dictionaries is the easiest way to keep track of the input files. """
    s = Stages()
    l = [(k, v) for k, v in sorted(imgs.items())]  # type: List[Tuple[K, MincAtom]]
    ks = [k for k, _ in l]
    vs = [v for _, v in l]
    res = s.defer(lsq12_pairwise(imgs=vs, conf=conf, lsq12_dir=lsq12_dir, like=like))
    return Result(stages=s, output=WithAvgImgs(output=dict(zip(ks, res.output)),
                                               avg_imgs=res.avg_imgs,
                                               avg_img=res.avg_img))


def lsq12_nlin_build_model(imgs       : List[MincAtom],
                           lsq12_conf : LSQ12Conf,
                           lsq12_dir  : str,
                           nlin_dir   : str,
                           nlin_conf  : Union[MultilevelMinctraccConf, MultilevelMincANTSConf],
                           resolution : float) -> Result[WithAvgImgs[List[XfmHandler]]]:
    """
    Runs both a pairwise lsq12 registration followed by a non linear
    registration procedure on the input files.
    """
    s = Stages()

    # TODO: make sure that we pass on a correct configuration for lsq12_pairwise
    # it should be able to get this passed in....
    lsq12_result = s.defer(lsq12_pairwise(imgs=imgs, like=None,
                                          lsq12_conf=lsq12_conf,
                                          conf=default_lsq12_multilevel_minctracc, lsq12_dir=lsq12_dir))

    # extract the resampled lsq12 images
    lsq12_resampled_imgs = [xfm_handler.resampled for xfm_handler in  lsq12_result.output]

    if isinstance(nlin_conf, MultilevelMinctraccConf):
        nlin_result = s.defer(minctracc_NLIN_build_model(imgs=lsq12_resampled_imgs,
                                                         initial_target=lsq12_result.avg_img,
                                                         conf=nlin_conf,
                                                         nlin_dir=nlin_dir))
    elif isinstance(nlin_conf, MultilevelMincANTSConf):
        nlin_result = s.defer(mincANTS_NLIN_build_model(imgs=lsq12_resampled_imgs,
                                                        initial_target=lsq12_result.avg_img,
                                                        conf=nlin_conf,
                                                        nlin_dir=nlin_dir))
    else:
        # this should never happen
        raise ValueError("The non linear configuration passed to lsq12_nlin_build_model is neither for minctracc nor for mincANTS.")

    nlin_resampled_imgs = [xfm_handler.resampled for xfm_handler in nlin_result.output]

    # concatenate the transformations from lsq12 and nlin before returning them
    from_imgs_to_nlin_xfms = [s.defer(concat_xfmhandlers(xfms=[lsq12_xfmh, nlin_xfmh],
                                                         name=img.filename_wo_ext + "_lsq12_and_nlin")) for
                              lsq12_xfmh, nlin_xfmh, img in zip(lsq12_result.output,
                                                                     nlin_result.output,
                                                                     imgs)]

    return Result(stages=s, output=WithAvgImgs(output=from_imgs_to_nlin_xfms,
                                               avg_img=nlin_result.avg_img,
                                               avg_imgs=nlin_result.avg_imgs))


def can_read_MINC_file(filename: str) -> bool:
    """Can the MINC file `filename` with be read with `mincinfo`?"""
    # FIXME: opening and closing this file many times can be quite slow on some platforms ...
    # better to change to can_read_MINC_files.
    with open(os.devnull, 'w') as dev_null:
        returncode = subprocess.call(["mincinfo", filename], stdout=dev_null, stderr=dev_null) == 0
    return returncode


def check_MINC_input_files(args: List[str]) -> None:
    """
    This is a general function that checks MINC input files to a pipeline. It uses
    the program mincinfo to test whether the input files are readable MINC files,
    and ensures that the input files have distinct filenames. (A directory is created
    for each input file based on the filename without extension, which means that 
    filenames need to be distinct)
    
    args: names of input files
    """
    if len(args) < 1:
        raise ValueError("\nNo input files are provided.\n")
    else:
        # here we should also check that the input files can be read
        issuesWithInputs = 0
        for inputF in args:
            if not can_read_MINC_file(inputF):  # FIXME this will be quite slow on SciNet, etc.
                print("\nError: can not read input file: " + str(inputF) + "\n", file=sys.stderr)
                issuesWithInputs = 1
        if issuesWithInputs:
            raise ValueError("\nIssues reading input files.\n")
    # lastly we should check that the actual filenames are distinct, because
    # directories are made based on the basename
    seen = set()  # type: Set[str]
    for inputF in args:
        fileBase = os.path.splitext(os.path.basename(inputF))[0]
        if fileBase in seen:
            raise ValueError("\nThe following name occurs at least twice in the "
                             "input file list:\n" + str(fileBase) + ".mnc\nPlease provide "
                                                                    "unique names for all input files.\n")
        seen.add(fileBase)

def check_MINC_files_have_equal_dimensions_and_resolution(args: List[str],
                                                          additional_msg: str = "") -> bool:
    """
    Reads input files and compares the dimension length, start and stepsizes of the
    first input file against all the others. Raises an error if they are not all
    the same
    """
    if len(args) < 2:
        return True

    first_file = volumeFromFile(args[0])
    img_dimensions_first = ([int(element) for element in first_file.getSizes()])
    img_separations_first = first_file.separations
    img_starts_first = first_file.starts
    for other_img in args[1:]:
        other_volume = volumeFromFile(other_img)
        if not img_dimensions_first   == ([int(element) for element in other_volume.getSizes()]) or \
            not img_separations_first == other_volume.separations or \
            not img_starts_first      == other_volume.starts :
            print("\nThe input files do not all have the same "
                  "dimensions/starts/step sizes. The first input "
                  "file:\n", str(args[0]), " differs from:\n",
                  str(other_img), "\n")
            raise ValueError("Not all input images have similar bounding boxes. "
                             + additional_msg)


# data structures to hold setting for the parameter settings we know about:
mousebrain = {'resolution': 0.056}
human = {'resolution': 1.00}
# we want to set the parameters such that 
# blur          ->  560 micron (10 * resolution)
# resample step ->  224 micron ( 4 * resolution)
# registr. step ->  560 micron (10 * resolution)
# w_trans       ->  448 micron ( 8 * resolution)
# simplex       -> 1120 micron (20 * resolution)
# so we simply need to set the resolution to establish this:
# resolution_for_rot = 0.056
# could also use an Enum here, although a dict (or subclasses) can be extended
known_settings = {'mousebrain': mousebrain,
                  'human': human}


## under construction
# TODO it's silly for rotation_params to be overloaded like this - make another parser flag (--subject-matter)
#def parse_rotational_minctracc_params(rotation_params : str) -> Tuple[RotationalMinctraccConf, float]:
#    if rotation_params in known_settings:
#        resolution = known_settings[rotation_params]
#    else:
#        blur_factor, resample_step_factor, registration_step_factor, w_translations_factor = (
#            [float(x) for x in rotation_params.split(',')])
#    return config, resolution


def get_parameters_for_rotational_minctracc(resolution,  # TODO why do most arguments default to None?
                                            rotation_tmp_dir=None,
                                            rotation_range=None,
                                            rotation_interval=None,
                                            rotation_params=None):
    """
    returns the proper combination of a rotational_minctracc configuration and
    a value for the resolution such that the "mousebrain" option for 
    the --lsq6-large-rotations-parameters flag works.

    The parameters for rotational minctracc are set based on the resolution of the input files.
    If not explicitly specified, most parameters are multiples of that resolution.

    resolution - int/float indicating the resolution of the input files
    rotation_params - a list with 4 integer elements indicating multiples of the resolution:
                     [blur factor,
                     resample factor,
                     registration factor,
                     w_translation factor]
    """
    rotational_configuration = default_rotational_minctracc_conf.maybe_replace(
                                 temp_dir=rotation_tmp_dir,
                                 rotational_range=rotation_range,
                                 rotational_interval=rotation_interval)
    rotational_resolution    = resolution
    # for mouse brains we have fixed parameters:
    # if rotation_params == "mousebrain":
    if rotation_params in known_settings:
        rotational_resolution = known_settings[rotation_params]['resolution']
    else:
        if rotation_params:
            blur_factor, resample_step_factor, registration_step_factor, w_translations_factor = (
                [float(x) for x in rotation_params.split(',')])
            rotational_configuration = rotational_configuration.replace(
                                         blur_factor=blur_factor,
                                         resample_step_factor=resample_step_factor,
                                         registration_step_factor=registration_step_factor,
                                         w_translations_factor=w_translations_factor)

    return rotational_configuration, rotational_resolution


def to_lsq6_conf(lsq6_args : Namespace) -> LSQ6Conf:
    """Convert a Namespace produced by an LSQ6 option parser to an LSQ6Conf.
    This lets us inherit defaults from the parser without duplicating them here,
    and we also change the three flags into a single enum + value to reduce the risk of later errors."""
    target_type, target_file = verify_correct_lsq6_target_options(init_model=lsq6_args.init_model,
                                                                  bootstrap=lsq6_args.bootstrap,
                                                                  lsq6_target=lsq6_args.lsq6_target,
                                                                  pride_of_models=lsq6_args.pride_of_models)
    new_args = lsq6_args.__dict__.copy()
    for k in ["init_model", "bootstrap", "lsq6_target", "pride_of_models"]:
        del new_args[k]
    new_args["target_type"] = target_type
    new_args["target_file"] = target_file
    return LSQ6Conf(**new_args)


def lsq6(imgs: List[MincAtom],
         target: MincAtom,
         resolution: float,
         #output_dir,
         #pipeline_name,
         conf: LSQ6Conf,
         post_alignment_xfm: XfmAtom = None,
         post_alignment_target: MincAtom = None) -> Result[List[XfmHandler]]:
    """
    post_alignment_xfm -- this is a transformation you want to concatenate with the
                          output of the transform aligning the imgs to the target.
                          The most obvious example is a native_to_standard.xfm transformation
                          from an initial model.
    post_alignment_target -- a MincAtom indicating the space that the post alignment
                             transformation brings you into

    The output name of the transformations generated in this function will be:

    imgs[i].output_sub_dir + "_lsq6.xfm"
    """
    s = Stages()
    xfms_to_target = []  # type: List[XfmHandler]

    if post_alignment_xfm and not post_alignment_target:
        raise ValueError("You've provided a post alignment transformation to lsq6() but not a MincAtom indicating the target for this transformation.")

    if not conf.run_lsq6:
        raise ValueError("You silly person... you've called lsq6(), but also specified --no-run-lsq6. That's not a very sensible combination of things.")

    # FIXME this is a stupid function: it's not very safe (note lack of type of argument) and rather redundant ...
    def conf_from_defaults(defaults) -> MultilevelMinctraccConf:
        conf = MultilevelMinctraccConf(
            [MinctraccConf(step_sizes=[defaults["blur_factors"][i] * resolution] * 3,
                           blur_resolution=defaults["blur_factors"][i] * resolution,
                           use_gradient=defaults["gradients"][i],
                           use_masks=True,
                           linear_conf=default_linear_minctracc_conf('lsq6').replace(w_translations=[defaults["translations"][i]] * 3,
                                                                                     simplex=defaults["simplex_factors"][i] * resolution),
                           nonlinear_conf=None)
             for i in range(len(defaults["blur_factors"]))])  # FIXME: don't assume all lengths are equal
        return conf

    ############################################################################
    # alignment - switch on lsq6_method
    ############################################################################
    if conf.lsq6_method == "lsq6_large_rotations":
        # still not convinced this shouldn't go inside rotational_minctracc somehow,
        # though you may want to override ...
        rotational_configuration, resolution_for_rot = \
            get_parameters_for_rotational_minctracc(resolution=resolution,
                                                    rotation_tmp_dir=conf.rotation_tmp_dir,
                                                    rotation_range=conf.rotation_range,
                                                    rotation_interval=conf.rotation_interval,
                                                    rotation_params=conf.rotation_params)
        # now call rotational_minctracc on all input images 
        xfms_to_target = [s.defer(rotational_minctracc(source=img, target=target,
                                                       conf=rotational_configuration,
                                                       resolution=resolution_for_rot,
                                                       output_name_wo_ext=None if post_alignment_xfm else
                                                                          img.output_sub_dir + "_lsq6"))
                          for img in imgs]
    elif conf.lsq6_method == "lsq6_centre_estimation":
        defaults = { 'blur_factors'    : [   90,    35,    17,    9,     4],
                     'simplex_factors' : [  128,    64,    40,   28,    16],
                     'step_factors'    : [   90,    35,    17,    9,     4],
                     'gradients'       : [False, False, False, True, False],
                     'translations'    : [  0.4,   0.4,   0.4,  0.4,   0.4] }
        if conf.protocol_file is not None:
            mt_conf = parse_minctracc_lin_protocol_file(conf.protocol_file, NotImplemented)
        else:
            mt_conf = conf_from_defaults(defaults)
        xfms_to_target = [s.defer(multilevel_minctracc(source=img, target=target, conf=mt_conf,
                                                       transform_info=["-est_center", "-est_translations"]))
                          for img in imgs]
    elif conf.lsq6_method == "lsq6_simple":
        defaults = { 'blur_factors'    : [   17,    9,     4],
                     'simplex_factors' : [   40,   28,    16],
                     'step_factors'    : [   17,    9,     4],
                     'gradients'       : [False, True, False],
                     'translations'    : [  0.4,  0.4,   0.4] }


        if conf.protocol_file is not None:  # FIXME the proliferations of LSQ6Confs vs. MultilevelMinctraccConfs here is very confusing ...
            mt_conf = parse_minctracc_lin_protocol_file(conf.protocol_file, NotImplemented)  # FIXME don't totally ignore confs here ?!
        else:
            mt_conf = conf_from_defaults(defaults)

        xfms_to_target = [s.defer(multilevel_minctracc(source=img, target=target, conf=mt_conf))
                          for img in imgs]
    else:
        raise ValueError("bad lsq6 method: %s" % conf.lsq6_method)

    final_xfmhs = xfms_to_target
    if post_alignment_xfm:
        final_xfmhs = [XfmHandler(xfm=s.defer(xfmconcat([first_xfm.xfm,
                                                         post_alignment_xfm],
                                                         name=first_xfm.xfm.output_sub_dir + "_lsq6")),
                                  target=post_alignment_target,
                                  source=img,
                                  resampled=None)
                       for img, first_xfm in zip(imgs, xfms_to_target)]

    ############################################################################
    # TODO: resample input files ???
    ############################################################################

    return Result(stages=s, output=final_xfmhs)


class RegistrationTargets(object):
    """
    This class can be used for the following options:
    --init-model
    --lsq6-target
    --bootstrap
    """
    # TODO rename registration_standard to standard_space_target or similar?
    def __init__(self,
                 registration_standard: MincAtom,
                 xfm_to_standard: Optional[XfmAtom] = None,
                 registration_native: Optional[MincAtom] = None) -> None:
        self.registration_native = registration_native  # type : MincAtom
        self.registration_standard = registration_standard  # type : Optional[MincAtom]
        self.xfm_to_standard = xfm_to_standard  # type : Optional[XfmAtom]


def lsq6_nuc_inorm(imgs: List[MincAtom],
                   registration_targets: RegistrationTargets,
                   resolution: float,
                   lsq6_options: LSQ6Conf,
                   subject_matter: Optional[str] = None):
    s = Stages()

    # run the actual 6 parameter registration
    init_target = registration_targets.registration_native or registration_targets.registration_standard

    source_imgs_to_lsq6_target_xfms = s.defer(lsq6(imgs=imgs, target=init_target,
                                                   resolution=resolution,
                                                   conf=lsq6_options,
                                                   post_alignment_xfm=registration_targets.xfm_to_standard,
                                                   post_alignment_target=registration_targets.registration_standard))

    # TODO: perhaps this is a bit superfluous
    xfms_to_final_target_space = [xfm_handler.xfm for xfm_handler in source_imgs_to_lsq6_target_xfms]

    # resample the input to the final lsq6 space
    # we should go back to basics in terms of the file name that we create here. It should
    # be fairly basic. Something along the lines of:
    # {orig_file_base}_resampled_lsq6.mnc
    imgs_in_lsq6_space = [s.defer(mincresample(
        img=native_img,
        xfm=xfm_to_lsq6,
        like=registration_targets.registration_standard,
        interpolation=Interpolation.sinc,
        new_name_wo_ext=native_img.filename_wo_ext + "_lsq6"))
                          for native_img, xfm_to_lsq6 in zip(imgs, xfms_to_final_target_space)]

    # we've just performed a 6 parameter alignment between a bunch of input files
    # and a target. The input files could have been the very initial input files to the
    # pipeline, and have no masks associated with them. In that case, and if the target does
    # have a mask, we should add masks to the resampled files now.
    mask_to_add = registration_targets.registration_standard.mask
    for resampled_input in imgs_in_lsq6_space:
        if not resampled_input.mask:
            resampled_input.mask = mask_to_add

    # resample the mask from the initial model to native space
    # we can use it for either the non uniformity correction or
    # for intensity normalization later on
    masks_in_native_space = None  # type: List[MincAtom]
    if registration_targets.registration_standard.mask:
        # we should apply the non uniformity correction in
        # native space. Given that there is a mask, we should
        # resample it to that space using the inverse of the
        # lsq6 transformation we have so far
        masks_in_native_space = [s.defer(mincresample(img=registration_targets.registration_standard.mask,
                                                      xfm=xfm_to_lsq6,
                                                      like=native_img,
                                                      interpolation=Interpolation.nearest_neighbour,
                                                      extra_flags=('-invert',)))
                                 for native_img, xfm_to_lsq6 in zip(imgs, xfms_to_final_target_space)]

    # NUC
    nuc_imgs_in_native_space = None  # type: List[MincAtom]
    if lsq6_options.nuc:
        # if masks are around, they will be passed along to nu_correct,
        # if not we create a list with the same length as the number
        # of images with None values
        # what we get back here is a list of MincAtoms with NUC files
        nuc_imgs_in_native_space = [s.defer(nu_correct(src=native_img,
                                                       resolution=resolution,
                                                       mask=native_img_mask,
                                                       subject_matter=subject_matter))
                                    for native_img, native_img_mask
                                    in zip(imgs,
                                           masks_in_native_space if masks_in_native_space
                                           else [None] * len(imgs))]

    inorm_imgs_in_native_space = None  # type: List[MincAtom]
    if lsq6_options.inormalize:
        # TODO: this is still static
        inorm_conf = default_inormalize_conf
        input_imgs_for_inorm = nuc_imgs_in_native_space if nuc_imgs_in_native_space else imgs
        inorm_imgs_in_native_space = (
            [s.defer(inormalize(src=nuc_img,
                                conf=inorm_conf,
                                mask=native_img_mask))
             for nuc_img, native_img_mask in zip(input_imgs_for_inorm,
                                                 masks_in_native_space or [None] * len(input_imgs_for_inorm))])

    # the only thing left to check is whether we have to resample the NUC/inorm images to LSQ6 space:
    final_resampled_lsq6_files = imgs_in_lsq6_space
    if (lsq6_options.nuc and lsq6_options.inormalize) or lsq6_options.inormalize:  # FIXME 'and' clause is redundant !?
        # the final resampled files should be the normalized files resampled with the 
        # lsq6 transformation
        final_resampled_lsq6_files = [s.defer(mincresample(
                                                img=inorm_img,
                                                xfm=xfm_to_lsq6,
                                                like=registration_targets.registration_standard,
                                                interpolation=Interpolation.sinc,
                                                new_name_wo_ext=inorm_img.filename_wo_ext + "_lsq6"))
                                      for inorm_img, xfm_to_lsq6
                                      in zip(inorm_imgs_in_native_space,
                                             xfms_to_final_target_space)]
    elif lsq6_options.nuc:
        # the final resampled files should be the non uniformity corrected files 
        # resampled with the lsq6 transformation
        nuc_filenames_wo_ext_lsq6 = [nuc_img.filename_wo_ext + "_lsq6" for nuc_img in
                                     nuc_imgs_in_native_space]
        final_resampled_lsq6_files = [s.defer(mincresample(img=nuc_img,
                                                           xfm=xfm_to_lsq6,
                                                           like=registration_targets.registration_standard,
                                                           interpolation=Interpolation.sinc,
                                                           new_name_wo_ext=nuc_filename_wo_ext))
                                      for nuc_img, xfm_to_lsq6, nuc_filename_wo_ext
                                      in zip(nuc_imgs_in_native_space,
                                             xfms_to_final_target_space,
                                             nuc_filenames_wo_ext_lsq6)]
    else:
        # in this case neither non uniformity correction was applied, nor intensity 
        # normalization, so the initialization of the final_resampled_lsq6_files 
        # variable is already correct
        pass

    # we've just performed a 6 parameter alignment between a bunch of input files
    # and a target. The input files could have been the very initial input files to the
    # pipeline, and have no masks associated with them. In that case, and if the target does
    # have a mask, we should add masks to the resampled files now.
    mask_to_add = registration_targets.registration_standard.mask
    for resampled_input in final_resampled_lsq6_files:
        if not resampled_input.mask:
            resampled_input.mask = mask_to_add

        # note that in the return, the registration target is given as "registration_standard".
    # the actual registration might have been between the input file and a potential
    # "registration_native", but since we concatenated that transform with the
    # native_to_standard.xfm, the "registration_standard" file is the correct target
    # with respect to the transformation that's returned
    #
    # TODO: potentially add more to this return. Perhaps we want to pass along 
    #       non uniformity corrected / intensity normalized files in native space?
    return Result(stages=s, output=[XfmHandler(source=src_img,
                                               target=registration_targets.registration_standard,
                                               xfm=lsq6_xfm,
                                               resampled=final_resampled)
                                    for src_img, lsq6_xfm, final_resampled in
                                    zip(imgs, xfms_to_final_target_space, final_resampled_lsq6_files)])


def get_registration_targets_from_init_model(init_model_standard_file: str,
                                             output_dir: str,
                                             pipeline_name: str,
                                             timepoint: str = None) -> RegistrationTargets:
    """
    -- timepoint - this argument is used when a pride of models
                   is specified by the user (registration_chain).
                   In that case several directories should be created,
                   one for each of the time points

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
    # the output directory for files related to the initial model:
    if not timepoint:
        init_model_output_dir = os.path.join(output_dir, pipeline_name + "_init_model")
    else:
        init_model_output_dir = os.path.join(output_dir, pipeline_name + "_" + timepoint +"_init_model")

    # first things first, is this a nice MINC file:
    if not can_read_MINC_file(init_model_standard_file):
        raise ValueError("Error: can not read the following initial model file: %s" % init_model_standard_file)
    init_model_dir, standard_file_base = os.path.split(os.path.splitext(init_model_standard_file)[0])
    init_model_standard_mask = os.path.join(init_model_dir, standard_file_base + "_mask.mnc")
    # this mask file is a prerequisite, so we need to test it
    if not can_read_MINC_file(init_model_standard_mask):
        raise ValueError("Error (initial model): can not read/find the mask file for the standard space: %s"
                         % init_model_standard_mask)

    registration_standard = MincAtom(name=init_model_standard_file,
                                     mask=MincAtom(name=init_model_standard_mask,
                                                   pipeline_sub_dir=init_model_output_dir),
                                     pipeline_sub_dir=init_model_output_dir)

    # check to see if we are dealing with option 2), an initial model with native files
    init_model_native_file = os.path.join(init_model_dir, standard_file_base + "_native.mnc")
    init_model_native_mask = os.path.join(init_model_dir, standard_file_base + "_native_mask.mnc")
    init_model_native_to_standard = os.path.join(init_model_dir, standard_file_base + "_native_to_standard.xfm")
    if os.path.exists(init_model_native_file):
        if not can_read_MINC_file(init_model_native_file):
            raise ValueError("Error: can not read the following initial model file: %s" % init_model_native_file)
        if not can_read_MINC_file(init_model_native_mask):
            raise ValueError("Error: can not read the following initial model file (required native mask): %s"
                             % init_model_native_mask)
        registration_native = MincAtom(name=init_model_native_file,
                                       orig_name=init_model_native_file,
                                       mask=MincAtom(name=init_model_native_mask,
                                                     orig_name=init_model_output_dir,
                                                     pipeline_sub_dir=init_model_output_dir),
                                       pipeline_sub_dir=init_model_output_dir)
        if not os.path.exists(init_model_native_to_standard):
            raise ValueError(
                "\nError: can not read the following initial model file (required transformation when native "
                "files exist): %s\n" % init_model_native_to_standard)
        xfm_to_standard = XfmAtom(name=init_model_native_to_standard,
                                  orig_name=init_model_native_to_standard,
                                  pipeline_sub_dir=init_model_output_dir)
    else:
        registration_native = xfm_to_standard = None

    return RegistrationTargets(registration_standard=registration_standard,
                               xfm_to_standard=xfm_to_standard,
                               registration_native=registration_native)

# TODO should this just accept an object with the three relevant fields?
# TODO rename this function since it now returns interesting stuff ...
def verify_correct_lsq6_target_options(init_model: str,
                                       lsq6_target: str,
                                       bootstrap: bool,
                                       pride_of_models: str = None) -> Tuple[TargetType, Optional[str]]:
    """
    This function can be called using the parameters that are set using 
    the flags:
    --init-model
    --lsq6-target
    --bootstrap
    --pride-of-models (when running the registration_chain.py)
    
    it will check that exactly one of the options is provided, returning a filename (or None) tagged with
    the chosen target type, and raises an error otherwise.
    """

    # the registration_chain.py is the only program that works with the
    # pride_of_models. It is a little cleaner (I think?) to only show this
    # option when the main calling program is actually the registration_chain.py.
    # So let's find out! The stack is first in last out, so we need to last element:
    calling_program = inspect.stack()[-1][1]

    # check how many options have been specified that can be used as the initial target
    number_of_target_options = sum((bootstrap is not False,
                                    init_model is not None,
                                    lsq6_target is not None,
                                    pride_of_models is not None))

    no_target_msg = "Error: please specify a target for the 6 parameter alignment. " \
                    "Options are: --lsq6-target, --init-model, --bootstrap"
    if re.search(calling_program, "registration_chain.py"):
        no_target_msg += ", --pride-of-models"
    too_many_target_msg = "Error: please specify only one of the following options: " \
                          "--lsq6-target, --init-model, --bootstrap"
    if re.search(calling_program, "registration_chain.py"):
        too_many_target_msg += ", --pride-of-models. "
    too_many_target_msg += " Don't know which target to use..."


    if number_of_target_options == 0:
        raise ValueError(no_target_msg)
    if number_of_target_options > 1:
        raise ValueError(too_many_target_msg)

    if init_model:
        return TargetType.initial_model, init_model
    elif bootstrap:
        return TargetType.bootstrap, None
    elif lsq6_target:
        return TargetType.target, lsq6_target
    elif pride_of_models:
        return TargetType.pride_of_models, pride_of_models


# TODO: why is this separate?
def registration_targets(lsq6_conf: LSQ6Conf,
                         app_conf,
                         first_input_file: Optional[str] = None) -> RegistrationTargets:

    target_type   = lsq6_conf.target_type
    target_file   = lsq6_conf.target_file
    output_dir    = app_conf.output_directory
    pipeline_name = app_conf.pipeline_name
    # TODO currently the 'files' must come from the app_conf, but we might want to supply them separately
    # (e.g., for a pipeline which takes files in a csv but for which, e.g., bootstrap is still meaningful)

    # if we are dealing with either an lsq6 target or a bootstrap model
    # create the appropriate directories for those
    if target_type == TargetType.target:
        if not can_read_MINC_file(target_file):
            raise ValueError("Cannot read MINC file: %s" % target_file)
        target_file = MincAtom(name=target_file,
                               pipeline_sub_dir=os.path.join(output_dir, pipeline_name +
                                                             "_target_file"))
        return RegistrationTargets(registration_standard=target_file,
                                   xfm_to_standard=None,
                                   registration_native=None)
    elif target_type == TargetType.bootstrap:
        if target_file is not None:
            raise ValueError("BUG: bootstrap was chosen but a file was specified ...")
        if first_input_file is None:
            raise ValueError("Bootstrap was chosen but no input files supplied "
                             "(possibly a bug: I'm probably only looking at input files specified "
                             "on the command line, so if you supplied them in a CSV "
                             "I might not have noticed (or this might not make sense) ...")
        if not can_read_MINC_file(first_input_file):
            raise ValueError("Error (bootstrap file): can not read MINC file: %s\n" % first_input_file)
        bootstrap_file = MincAtom(name=first_input_file,
                                  pipeline_sub_dir=os.path.join(output_dir, pipeline_name +
                                                                "_bootstrap_file"))
        return RegistrationTargets(registration_standard=bootstrap_file)

    elif target_type == TargetType.initial_model:
        return get_registration_targets_from_init_model(target_file, output_dir, pipeline_name)
    else:
        raise ValueError("Invalid target type: %s" % lsq6_conf.target_type)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_pride_of_models_mapping(pride_top_level_dir: str,
                                output_dir: str,
                                pipeline_name: str):
    """
    Assumptions/requirements for a pride of models:
    -- all initial models have the same resolution

    -- all keys in the dictionary will be mapped as float

    returns:
    dictionary mapping from:
    time_point -> RegistrationTargets
    """
    list_of_initial_model_dirs = os.listdir(pride_top_level_dir)
    common_resolution = None

    pride_of_models_dict = {}
    for init_model_dir in list_of_initial_model_dirs:
        # in order to use the names of the initial model directories as
        # timepoints, they need to be integers (floats would be okay too)
        if not is_number(init_model_dir):
            raise ValueError("Error: the directory names of the initial models "
                             "that make up the pride of models are used to determine "
                             "its timepoint. These directory names need to be either "
                             "integers or floats. Please rename this directory: " +
                             os.path.join(pride_top_level_dir, init_model_dir))

        files_in_dir = next(os.walk(os.path.join(pride_top_level_dir, init_model_dir)))[2]
        commonprefix = os.path.commonprefix(files_in_dir)
        model_in_standard_space = os.path.join(pride_top_level_dir, init_model_dir, commonprefix + ".mnc")
        if not common_resolution:
            common_resolution = get_resolution_from_file(model_in_standard_space)
        else:
            if not common_resolution == get_resolution_from_file(model_in_standard_space):
                raise ValueError("Error: we currently require all the initial models in the "
                                 "pride of models to have the same resolution. The resolution of "
                                 "this file: " + str(model_in_standard_space) + " (" +
                                 str(get_resolution_from_file(model_in_standard_space)) + ") is different "
                                 "from the resolution we found so far: " + str(common_resolution))

        pride_of_models_dict[float(init_model_dir)] = get_registration_targets_from_init_model(model_in_standard_space,
                                                                                        output_dir,
                                                                                        pipeline_name,
                                                                                        init_model_dir)
    return pride_of_models_dict


def get_resolution_from_file(input_file: str) -> float:
    """
    input_file -- string pointing to an existing MINC file
    """
    # quite important is that this file actually exists...
    if not can_read_MINC_file(input_file):
        raise IOError("\nError: can not read input file: %s\n" % input_file)

    image_resolution = volumeFromFile(input_file).separations

    return min([abs(x) for x in image_resolution])


def create_quality_control_images(imgs: List[MincAtom],
                                  create_montage:bool = True,
                                  montage_output:str = None,
                                  montage_dir:str = None,
                                  scaling_factor: int = 20,
                                  message:str = "lsq6"):
    """
    This class takes a list of input files and creates
    a set of quality control (verification) images. Optionally
    these images can be combined in a single montage image for
    easy viewing

    montage_dir -- the main output directory for montage images
                   if provided, all log files will go into a
                   subdirectory called "log" for montage images

    The scaling factor corresponds to the the mincpik -scale
    parameter
    """
    s = Stages()
    individualImages = []
    individualImagesLabeled = []

    if create_montage and montage_output == None:
        print("\nError: createMontage is specified in createQualityControlImages, but no output name for the montage is provided. Exiting...\n")
        sys.exit()

    # for each of the input files, run a mincpik call and create
    # a triplane image.
    for img in imgs:
        img_verification = img.newname_with_suffix("_QC_image",
                                                   subdir="tmp",
                                                   ext=".png")
        # TODO: the memory and procs are set to 0 to ensure that
        # these stages finish soonish. No other stages depend on
        # these, but we do want them to finish as soon as possible
        mincpik_stage = CmdStage(
            inputs=(img,),
            outputs=(img_verification,),
            cmd=["mincpik", "-clobber",
                 "-scale", str(scaling_factor),
                 "-triplanar",
                 img.path, img_verification.path],
            memory=0,
            procs=0)
        mincpik_stage.set_log_file(os.path.join(img_verification.pipeline_sub_dir,
                                                img_verification.output_sub_dir,
                                                "log",
                                                "mincpik_" + img_verification.filename_wo_ext + ".log"))
        s.add(mincpik_stage)
        individualImages.append(img_verification)


        # we should add a label to each of the individual images
        # so it will be easier for the user to identify what
        # which images potentially fail
        img_verification_convert = img.newname_with_suffix("_QC_image_labeled",
                                                           subdir="tmp",
                                                           ext=".png")
        # FIXME: the memory and procs are set to 0 to ensure that
        # these stages finish soonish. No other stages depend on
        # these, but we do want them to finish as soon as possible
        # (Note that this may lead to large memory consumption by individual executors,
        # particularly for large pipelines, and seems unlikely to work at all on the HPF)
        convert_stage = CmdStage(
            inputs=(img_verification,),
            outputs=(img_verification_convert,),
            cmd=["convert",
                 "-label", img.output_sub_dir,
                 img_verification.path,
                 img_verification_convert.path],
            memory=0,
            procs=0)
        convert_stage.set_log_file(os.path.join(img_verification_convert.pipeline_sub_dir,
                                                img_verification_convert.output_sub_dir,
                                                "log",
                                                "convert_" + img_verification_convert.filename_wo_ext + ".log"))
        s.add(convert_stage)
        individualImagesLabeled.append(img_verification_convert)


    # if montageOutput is specified, create the overview image
    if create_montage:
        # TODO: the memory and procs are set to 0 to ensure that
        # these stages finish soonish. No other stages depend on
        # these, but we do want them to finish as soon as possible
        # TODO: maybe... currently inputs and outputs to a pipeline
        # can only be FileAtoms/MincAtoms, so we can't just provide
        # a string to a file for this
        montage_output_fileatom = FileAtom(montage_output)

        montage_stage = CmdStage(
            inputs=tuple(individualImagesLabeled),
            outputs=(montage_output_fileatom,),
            cmd=["montage", "-geometry", "+2+2"] + \
                [labeled_img.path for labeled_img in individualImagesLabeled] + \
                [montage_output_fileatom.path],
            memory=0,
            procs=0)
        if montage_dir:
            montage_stage.set_log_file(os.path.join(montage_dir,
                                                    "log",
                                                    "montage_" + montage_output_fileatom.filename_wo_ext + ".log"))
        else:
            montage_stage.set_log_file(os.path.join(montage_output_fileatom.pipeline_sub_dir,
                                                    montage_output_fileatom.output_sub_dir,
                                                    "log",
                                                    "montage_" + montage_output_fileatom.filename_wo_ext + ".log"))
        message_to_print = "\n* * * * * * *\nPlease consider the following verification "
        message_to_print += "image, showing "
        message_to_print += "%s. " % message
        message_to_print += "\n%s\n" % (montage_output)
        message_to_print += "* * * * * * *\n"
        # the hook needs a return. Given that "print" does not return
        # anything, we need to encapsulate the print statement in a
        # function (which in this case will return None, but that's fine)
        def printMessageForMontage():
            print(message_to_print)
        montage_stage.when_finished_hooks.append(
            lambda : printMessageForMontage())

        s.add(montage_stage)

    return Result(stages=s, output=None)

