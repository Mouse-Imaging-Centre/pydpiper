import csv
import os
import random
import shlex
import subprocess
import sys
import warnings
import time
from operator import mul
import math

from configargparse import Namespace
from typing import Any, cast, Dict, Generic, Iterable, List, Optional, Set, Tuple, TypeVar, Union, Callable

from functools import reduce

from pydpiper.core.files import FileAtom
from pydpiper.core.stages import CmdStage, Result, Stages
from pydpiper.core.util import pairs, AutoEnum, NamedTuple, raise_, flatten
from pydpiper.minc.files import MincAtom, XfmAtom, xfmToMinc
from pydpiper.minc.containers import XfmHandler
from pyminc.volumes.factory import volumeFromFile  # type: ignore

# TODO push down into lsq12_pairwise?
gen = random.Random(137)  # seed must be a small int; see #291


R3 = Tuple[float, float, float]


class LinearTransType(AutoEnum):
    lsq3 = ()
    lsq6 = ()
    lsq7 = ()
    lsq9 = ()
    lsq10 = ()
    lsq12 = ()
    procrustes = ()


class LinearObjectiveFn(AutoEnum):
    # named so that <...>.name returns the appropriate string to pass to minctracc
    xcorr = zscore = ssc = vr = mi = nmi = ()


class NonlinearObjectiveFn(AutoEnum):
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
                                     ('protocol', Optional[str]),
                                     ('generate_tournament_style_lsq12_avg', Optional[bool])])


LinearMinctraccConf = NamedTuple("LinearMinctraccConf",
                                 [("objective", Optional[LinearObjectiveFn]),
                                  ("simplex", float),
                                  ("transform_type", Optional[LinearTransType]),
                                  ("tolerance", float),
                                  ("w_rotations", R3),
                                  ("w_translations", R3),
                                  ("w_scales", R3),
                                  ("w_shear", R3)])

# TODO writing a LinearMinctraccConf is annoying b/c of the nested structure,
# so write linear_minctracc_conf : ... -> MinctraccConf

# TODO this doesn't allow for the various optimization params or objective functions to be set ...
NonlinearMinctraccConf = NamedTuple("NonlinearMinctraccConf",
                                    [("iterations", int),
                                     ("use_simplex", bool),
                                     ("stiffness", float),
                                     ("weight", float),
                                     ("similarity", float),
                                     ("objective", Optional[NonlinearObjectiveFn]),
                                     ("lattice_diameter", Optional[R3]),
                                     ("sub_lattice", int)])

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

# N.B.: mincblur has a slight weirdness in that it might create one or two
# output files.  Due to our semi-manual managing of the filenames without any allocation/gensym, there's always
# the risk that two different parts of a program may try to write to the same output file (manifest at the level
# of output file checking in the Python code if we're honest about outputs, and as weird output file corruption if not).
# In general this might be a problem, but we cross our fingers and hope that in most such cases the two commands
# will be identical and so one will be optimized away.  If our `mincblur` procedure returned the path to _either_
# an image or gradient file, however, it seems less likely that we'd be so fortunate.  As a result, mincblur currently
# returns both files by default, but can be disabled as an optimization.  Better than this would be
# lazily computing the gradient ...
# We currently return a Namespace(img) or Namespace(img, gradient), but dynamically choosing which field to select
# is tedious in Python ...

MincblurMemCfg = NamedTuple("MincblurMemCfg",
                            [('base_mem', float),
                             ('mem_per_voxel', float),
                             ('include_tmpdir', bool),
                             ('tmpdir_factor', float)])

default_mincblur_mem_cfg = MincblurMemCfg(base_mem=1.25e-02, mem_per_voxel=5.9e-9,
                                          tmpdir_factor=2.5, include_tmpdir=True)

def minc_displacement(xfm : XfmHandler) -> Result[MincAtom]:
    # TODO: add dir argument
    # TODO: this coercion is lame
    output_grid = xfmToMinc(xfm.xfm.newname_with_suffix("_displ", ext='.mnc', subdir="tmp"))
    stage = CmdStage(inputs=(xfm.source, xfm.xfm), outputs=(output_grid,),
                     cmd=['minc_displacement', '-clobber', xfm.source.path, xfm.xfm.path, output_grid.path])
    return Result(stages=Stages([stage]), output=output_grid)


def mincblur(img: MincAtom,
             fwhm: float,
             gradient: bool = True,
             subdir: str = 'tmp') -> Result[Namespace]:  # (img=MincAtom, Optional[gradient=MincAtom]):
    """
    >>> img = MincAtom(name='/images/img_1.mnc', pipeline_sub_dir='/scratch/some_pipeline_processed/')
    >>> img_blur = mincblur(img=img, fwhm=0.056)
    >>> img_blur.output.path
    '/scratch/some_pipeline_processed/img_1/tmp/img_1_fwhm0.056_blur.mnc'
    >>> [i.render() for i in img_blur.stages]
    ['mincblur -clobber -no_apodize -fwhm 0.056 /images/img_1.mnc /scratch/some_pipeline_processed/img_1/tmp/img_1_fwhm0.056']
    """

    # Is this the appropriate place for this?
    # the -1 is for compatibility with protocol files, while 0/False might make more sense (but is sort of 'in-band');
    # None is converted to NaN in Pandas data frames, which seems annoying
    if fwhm in (-1, 0, None):
        return Result(stages=Stages(), output=Namespace(img=img))

    # suffix   = "_dxyz" if gradient else "_blur"
    fwhm_str = "_fwhm%s" % fwhm
    out_img      = img.newname_with_suffix(fwhm_str + "_blur", subdir=subdir)
    out_gradient = img.newname_with_suffix(fwhm_str + "_dxyz", subdir=subdir)
    stage = CmdStage(
        inputs=(img,), outputs=(out_img,) + ((out_gradient,) if gradient else ()),
        # drop last 9 chars from output filename since mincblur
        # automatically adds "_blur.mnc" (or "_dxyz.mnc") and Python
        # won't lift this length calculation automatically ...
        cmd=shlex.split('mincblur -clobber -no_apodize -fwhm %s %s %s' % (fwhm, img.path, out_img.path[:-9]))
            + (['-gradient'] if gradient else []))
    #stage.set_log_file(os.path.join(out_img.dir, "..", "log",
    #                                "%s_%s.log" % ("mincblur", out_img.filename_wo_ext)))

    def set_memory(stage, mem_cfg):
        # we pass the stage itself as an argument since the stage will be converted to an old-style CmdStage,
        # so `stage` will have no effect.  In order to receive this argument, hooks must now take a self-argument
        # (instead of no arguments as previously).
        voxels = reduce(mul, volumeFromFile(img.path).getSizes())
        #default_mem = self.mem #hack; see pipeline.addStage method
        stage.setMem((mem_cfg.base_mem + voxels * mem_cfg.mem_per_voxel)
                     * (mem_cfg.tmpdir_factor if mem_cfg.include_tmpdir else 1))
    # FIXME this is the final word; we might want (1) either the executor/system to look at it
    # or (2) a wrapper that enforces some sensible minimum, as with Pydpiper 1.x
    # (but the default_job_mem is not accessible here ... could make exec_options an arg to the hooks ...? crazy)
    stage.when_runnable_hooks.append(lambda s: set_memory(s, default_mincblur_mem_cfg))

    return Result(stages=Stages((stage,)),
                  output=Namespace(img=out_img, gradient=out_gradient) if gradient else Namespace(img=out_img))


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
    s = Stages()

    if len(imgs) == 0:
        raise ValueError("`mincaverage` arg `imgs` is empty (can't average zero files)")

    # the output_dir basically gives us the equivalent of the pipeline_sub_dir for
    # regular input files to a pipeline, so use that here
    avg = avg_file or MincAtom(name=os.path.join(output_dir, '%s.mnc' % name_wo_ext),
                               orig_name=None,
                               pipeline_sub_dir=output_dir)

    # if all input files have masks associated with them, add the combined mask to
    # the average:
    all_inputs_have_masks = all((img.mask for img in imgs))

    if all_inputs_have_masks:
        combined_mask = (MincAtom(name=os.path.join(avg_file.dir, '%s_mask.mnc' % avg_file.filename_wo_ext),
                                  orig_name=None,
                                  pipeline_sub_dir=avg_file.pipeline_sub_dir)
                         if avg_file else
                         MincAtom(name=os.path.join(output_dir, '%s_mask.mnc' % name_wo_ext),
                                  orig_name=None,
                                  pipeline_sub_dir=output_dir))
        s.defer(mincmath(op='max',
                         # set comprehension loses order (OK as max is associative, commutative)
                         # but removes duplicates; 'sorted' preserved determinism:
                         vols=sorted({img_inst.mask for img_inst in imgs}),
                         out_atom=combined_mask))
        avg.mask = combined_mask

    # if the average was provided as a MincAtom there is probably a output_sub_dir
    # set. However, this is something we only really use for input files. All averages
    # and related files to directly into the _lsq6, _lsq12 and _nlin directories. That's
    # why we'll create this MincAtom here if avg_file was provided:
    sdfile = (MincAtom(name=os.path.join(avg_file.dir, '%s_sd.mnc' % avg_file.filename_wo_ext),
                       orig_name=None,
                       pipeline_sub_dir=avg_file.pipeline_sub_dir) if avg_file else
              MincAtom(name=os.path.join(output_dir, '%s_sd.mnc' % name_wo_ext),
                       orig_name=None,
                       pipeline_sub_dir=output_dir))
    additional_flags = ["-copy_header"] if copy_header_from_first_input else []  # type: List[str]
    avg_cmd = CmdStage(inputs=tuple(imgs), outputs=(avg, sdfile),
                       cmd=['mincaverage', '-clobber', '-normalize', '-max_buffer_size_in_kb', '409620'] +
                           additional_flags +
                           ['-sdfile', sdfile.path] +
                           sorted([img.path for img in imgs]) +
                           [avg.path])
    # averages in a pipeline often indicate important progress. Let's report that back to the user
    # in terms of a status update:
    status_update_message = "\n\n*\n* * *\n* * * * *\n* * * * * * *\nStatus update: \nFinished creating the following average:\n" \
                            + str(avg.path) + "\n" + time.ctime() + "\n* * * * * * *\n* * * * *\n* * *\n*\n"

    avg_cmd.when_finished_hooks.append(lambda _: print(status_update_message))

    s.add(avg_cmd)
    return Result(stages=s, output=avg)


PMincAverageMemCfg = NamedTuple("PMincAverageMemCfg", [('base_mem', float), ('mem_per_voxel', float)])
default_pmincaverage_mem_cfg = PMincAverageMemCfg(base_mem=0.5, mem_per_voxel=14.0/(430*13158000.0))


def mincbigaverage(imgs : List[MincAtom],
                   name_wo_ext: str = "average",
                   avgnum: Optional[int] = None,
                   robust: Optional[bool] = None,
                   output_dir: str = '.',
                   avg_file: Optional[MincAtom] = None,
                   sdfile: Optional[str] = None,
                   tmpdir: Optional[str] = None,
                   copy_header_from_first_input: bool = False):


    if len(imgs) == 0:
        raise ValueError("`mincbigaverage` arg `imgs` is empty (can't average zero files)")

    if copy_header_from_first_input:
        # TODO: should be logged, not just printed?
        warnings.warn("Warning: mincbigaverage doesn't implement copy_header; use mincaverage instead")

    s = Stages()

    avg = avg_file or MincAtom(name=os.path.join(output_dir, '%s.mnc' % name_wo_ext),
                               orig_name=None,
                               pipeline_sub_dir=output_dir)

    # TODO use --filelist instead of putting all files on command line?
    avg_cmd = CmdStage(inputs=tuple(imgs), outputs=(avg,),
                       cmd=["mincbigaverage", "-clobber"]
                           + (["--avgnum", avgnum] if avgnum else [])
                           + (["--robust"] if robust else [])
                           + (["--tmpdir", tmpdir] if tmpdir else [])
                           + (["--sdfile", sdfile] if sdfile else [])
                           + sorted([img.path for img in imgs]) + [avg.path])

    # if all input files have masks associated with them, add the combined mask to the average:
    all_inputs_have_masks = all((img.mask for img in imgs))

    if all_inputs_have_masks:
        combined_mask = (MincAtom(name=os.path.join(avg_file.dir, '%s_mask.mnc' % avg_file.filename_wo_ext),
                                  orig_name=None,
                                  pipeline_sub_dir=avg_file.pipeline_sub_dir)
                         if avg_file else
                         MincAtom(name=os.path.join(output_dir, '%s_mask.mnc' % name_wo_ext),
                                  orig_name=None,
                                  pipeline_sub_dir=output_dir))
        s.defer(mincmath(op='max',
                         # set comprehension loses order (OK as max is associative, commutative)
                         # but removes duplicates; 'sorted' preserved determinism:
                         vols=sorted({img_inst.mask for img_inst in imgs}),
                         out_atom=combined_mask))
        avg.mask = combined_mask

    # TODO do we need to set memory?  test!

    # averages in a pipeline often indicate important progress. Let's report that back to the user
    # in terms of a status update:
    status_update_message = "\n\n* * * * * * *\nStatus update: \nFinished creating the following average:\n" \
                            + str(avg.path) + "\n" + time.ctime() + "\n* * * * * * *\n"

    avg_cmd.when_finished_hooks.append(lambda _: print(status_update_message))

    s.add(avg_cmd)

    return Result(stages=s, output=avg)


# FIXME this doesn't implement the avg_file and other mincaverage stuff (other than copy_header ...)
# TODO  maybe there's enough similarity to parametrize over these and maybe others (xfmavg?!)
def pmincaverage(imgs: List[MincAtom],
                 name_wo_ext: str = "average",
                 output_dir: str = '.',
                 avg_file: Optional[MincAtom] = None,
                 copy_header_from_first_input: bool = False):

    ## TODO this now basically duplicates `mincaverage` with a few differences ... refactor! ...

    if len(imgs) == 0:
        raise ValueError("`mincaverage` arg `imgs` is empty (can't average zero files)")

    if copy_header_from_first_input:
        # TODO: should be logged, not just printed?
        warnings.warn("Warning: pmincaverage doesn't implement copy_header; use mincaverage instead")

    s = Stages()

    avg = avg_file or MincAtom(name=os.path.join(output_dir, '%s.mnc' % name_wo_ext),
                               orig_name=None,
                               pipeline_sub_dir=output_dir)

    avg_cmd = CmdStage(inputs=tuple(imgs), outputs=(avg,),
                       cmd=["pmincaverage", "--clobber"] + sorted([img.path for img in imgs]) + [avg.path])


    # if all input files have masks associated with them, add the combined mask to the average:
    all_inputs_have_masks = all((img.mask for img in imgs))

    if all_inputs_have_masks:
        combined_mask = (MincAtom(name=os.path.join(avg_file.dir, '%s_mask.mnc' % avg_file.filename_wo_ext),
                                  orig_name=None,
                                  pipeline_sub_dir=avg_file.pipeline_sub_dir)
                         if avg_file else
                         MincAtom(name=os.path.join(output_dir, '%s_mask.mnc' % name_wo_ext),
                                  orig_name=None,
                                  pipeline_sub_dir=output_dir))
        s.defer(mincmath(op='max',
                         # set comprehension loses order (OK as max is associative, commutative)
                         # but removes duplicates; 'sorted' preserved determinism:
                         vols=sorted({img_inst.mask for img_inst in imgs}),
                         out_atom=combined_mask))
        avg.mask = combined_mask

    def set_memory(st, cfg):
        voxels_per_file = reduce(mul, volumeFromFile(imgs[0].path).getSizes())
        st.setMem(cfg.base_mem + voxels_per_file * cfg.mem_per_voxel * len(imgs))

    avg_cmd.when_runnable_hooks.append(lambda st: set_memory(st, default_pmincaverage_mem_cfg))

    # averages in a pipeline often indicate important progress. Let's report that back to the user
    # in terms of a status update:
    status_update_message = "\n\n*\n* * *\n* * * * *\n* * * * * * *\nStatus update: \nFinished creating the following average:\n" \
                            + str(avg.path) + "\n" + time.ctime() + "\n* * * * * * *\n* * * * *\n* * *\n*\n"

    avg_cmd.when_finished_hooks.append(lambda _: print(status_update_message))

    s.add(avg_cmd)

    return Result(stages=s, output=avg)


def mincreshape(img : MincAtom, args : List[str]):
    out_img=img.newname_with(''.join(args))  # TODO better naming
    stage = CmdStage(inputs=(img,), outputs=out_img,
                     cmd=["mincreshape", "-clobber"] + args + [img.path, out_img.path])
    return Result(stages=Stages([stage]), output=out_img)


class Interpolation(AutoEnum):
    trilinear = tricubic = sinc = nearest_neighbour = ()


def mincresample_simple(img: MincAtom,
                        xfm: XfmAtom,
                        like: MincAtom,
                        extra_flags: Tuple[str] = (),
                        interpolation: Optional[Interpolation] = None,
                        invert: bool = False,
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
        # FIXME Also, using the xfm's filename is wrong, since we might be resampling, e.g., a mask.
        # FIXME We should basically use the same naming scheme as is used to generate the xfm's name but
        # FIXME use the files for resampling, not registration
        outf = img.newname(name=xfm.filename_wo_ext + '-resampled', subdir=subdir)
    else:
        # we have the output filename without extension. This should replace the entire
        # current "base" of the filename. 
        outf = img.newname(name=new_name_wo_ext, subdir=subdir)

    stage = CmdStage(
        inputs=(xfm, like, img),
        outputs=(outf,),
        cmd=(['mincresample', '-clobber', '-2']
             + (['-' + interpolation.name] if interpolation else [])
             + (['-invert'] if invert else [])
             + list(extra_flags)
             + (['-transform %s' % xfm.path]) #if xfm is not identity else [])
             + ['-like %s' % like.path, img.path, outf.path]))

    return Result(stages=Stages([stage]), output=outf)


#class IdentityXfm(object):
#    pass

#identity_xfm = IdentityXfm()

# TODO it's probably better to make -keep_real_range its own argument or something to avoid
#   "-keep_real_rnge" all over the place ...
# TODO mincresample_simple could easily be replaced by a recursive call to mincresample ... what about separate
# wrappers mincresample_mask, mincresample_labels?
def mincresample(img: MincAtom,
                 xfm: XfmAtom,  # TODO: update to handler?
                 like: MincAtom,
                 invert: bool = False,
                 interpolation: Interpolation = None,
                 extra_flags: Tuple[str] = (),
                 new_name_wo_ext: str = None,
                 subdir: str = None) -> Result[MincAtom]:
    """
    ...
    new_name_wo_ext -- string indicating a user specified file name (without extension)
    subdir          -- string indicating which subdirectory to output the file in:

    >>> img1 = MincAtom('/tmp/img_1.mnc')
    >>> img2 = MincAtom('/tmp/img_2.mnc')
    >>> xfm  = XfmAtom('/tmp/trans.xfm')
    >>> stages = mincresample(img=img1, xfm=xfm, like=img2).stages
    >>> [x.render() for x in stages]
    ['mincresample -clobber -2 -transform /tmp/trans.xfm -like /tmp/img_2.mnc /tmp/img_1.mnc img_1/resampled/trans-resampled.mnc']
    >>> stages_ = mincresample(img=img2, xfm=xfm, like=img1, invert=True).stages
    >>> [x.render() for x in stages_]
    ['mincresample -clobber -2 -invert -transform /tmp/trans.xfm -like /tmp/img_1.mnc /tmp/img_2.mnc img_2/resampled/trans-resampled.mnc']
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
        # FIXME this is wrong when invert=True
        new_name_wo_ext = xfm.filename_wo_ext + '-resampled'

    new_img = s.defer(mincresample_simple(img=img, xfm=xfm, like=like,
                                          extra_flags=extra_flags,
                                          invert=invert,
                                          interpolation=interpolation,
                                          new_name_wo_ext=new_name_wo_ext,
                                          subdir=subdir))
    new_img.mask = s.defer(mincresample_simple(img=img.mask, xfm=xfm, like=like,
                                               extra_flags=extra_flags,
                                               interpolation=Interpolation.nearest_neighbour,
                                               invert=invert,
                                               new_name_wo_ext=new_name_wo_ext + "_mask",
                                               subdir=subdir)) if img.mask is not None else None
    new_img.labels = s.defer(mincresample_simple(img=img.labels, xfm=xfm, like=like,
                                                 extra_flags=label_extra_flags,
                                                 interpolation=Interpolation.nearest_neighbour,
                                                 invert=invert,
                                                 new_name_wo_ext=new_name_wo_ext + "_labels",
                                                 subdir=subdir)) if img.labels is not None else None

    # Note that new_img can't be used for anything until the mask/label files are also resampled.
    # This shouldn't create a problem with stage dependencies as long as masks/labels appear in inputs/outputs of CmdStages.
    # (If this isn't automatic, a relevant helper function would be trivial.)
    # TODO: can/should this be done semi-automatically? probably ...
    return Result(stages=s, output=new_img)


# TODO should this be a method on XfmHandlers, allowing use of the default argument syntax to refer to
# the relevant mnc files?  (N.B.: each use site is on files already contained in XfmHandlers and manually dereferenced)
def mincresample_new(img: MincAtom,
                     xfm: XfmAtom,  # TODO: add optional 'img', 'like' params to override those determined from XfmH
                     like: MincAtom,
                     extra_flags: Tuple[str] = (),
                     invert: bool = False,
                     interpolation: Interpolation = None,
                     new_name_wo_ext: Optional[str] = None,
                     postfix: Optional[str] = None,
                     subdir: str = None) -> Result[MincAtom]:

    return mincresample(img=img, xfm=xfm, like=like, invert=invert,
                        interpolation=interpolation, extra_flags=extra_flags,
                        new_name_wo_ext=new_name_wo_ext or ("%s_res_to_%s%s" % (img.filename_wo_ext,
                                                                                like.filename_wo_ext,
                                                                                postfix if postfix else "")),
                        subdir=subdir)


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

        return Result(stages=Stages([stage]), output=outf)


#
#
# TODO: do we need this function at all??
#       the concept seems odd. It's a concat and a resample?
#
#
def concat_xfmhandlers(xfms: List[XfmHandler],
                       name: str = None,
                       resample_source: bool = True,
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
                               extra_flags=extra_flags)) if resample_source else None
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
    out = src.newname_with_suffix("_nu_estimate", ext=".imp", subdir="tmp")


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

    return Result(stages=Stages([cmd]), output=out)

def mincmath(op       : str,
             vols     : List[MincAtom],
             const    : Optional[float]    = None,
             new_name : Optional[str]      = None,
             subdir   : str                = "tmp",
             out_atom : Optional[MincAtom] = None) -> Result[MincAtom]:
    """
    Low-level/stupid interface to mincmath

    out_atom has the highest precedence in terms of the
    resulting MincAtom
    """
    _const = str(const) if const is not None else ""  # type: Optional[str]

    if not out_atom:
        if new_name:
            name = new_name
        elif len(vols) == 1:
            name = vols[0].filename_wo_ext + "_" + op + (("_" + _const) if _const else "")
        else:
            name = (op + '_' + ((_const + '_') if _const else '') +
                 '_'.join([vol.filename_wo_ext for vol in vols]))
        outf = vols[0].newname(name=name, subdir=subdir)
    else:
        outf = out_atom

    s = CmdStage(inputs=tuple(vols), outputs=(outf,),
                 cmd=(['mincmath', '-clobber', '-2']
                   + (['-const', _const] if _const else [])
                   + ['-' + op] + [v.path for v in vols] + [outf.path]))

    return Result(stages=Stages([s]), output=outf)

def nu_evaluate(img: MincAtom,
                field: FileAtom,
                subdir: str = "tmp") -> Result[MincAtom]:
    out = img.newname_with_suffix("_N", subdir=subdir)
    cmd = CmdStage(inputs=(img, field), outputs=(out,),
                   cmd=['nu_evaluate', '-clobber', '-mapping', field.path, img.path, out.path])

    return Result(stages=Stages([cmd]), output=out)

# TODO: could also use the `nu_correct` program instead of doing this:
def nu_correct(src: MincAtom,
               resolution: float,
               mask: Optional[MincAtom] = None,
               subject_matter: Optional[str] = None,
               subdir: str = "tmp") -> Result[MincAtom]:
    s = Stages()
    return Result(stages=s, output=s.defer(nu_evaluate(src, s.defer(nu_estimate(src, resolution,
                                                                                mask=mask,
                                                                                subject_matter=subject_matter)),
                                                       subdir=subdir)))


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
               mask: Optional[MincAtom] = None,
               subdir: str = "tmp") -> Result[MincAtom]:
    """
    Note: if a mask is specified through the "mask" parameter, it will have
    precedence over the mask that might be associated with the 
    src file.
    """
    out = src.newname_with_suffix('_I', subdir=subdir)

    mask_for_inormalize = mask or src.mask

    cmd = CmdStage(inputs=(src, mask_for_inormalize) if mask_for_inormalize else (src,),  # issue a warning if no mask?
                   outputs=(out,),
                   cmd=shlex.split('inormalize -clobber -const %s -%s' % (conf.const, conf.method.name))
                       + (['-mask', mask_for_inormalize.path] if mask_for_inormalize else [])
                       + [src.path, out.path])

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

    return Result(stages=Stages([stage]), output=outf)


def xfminvert(xfm: XfmAtom,
              subdir: str = "transforms") -> Result[XfmAtom]:
    inv_xfm = xfm.newname_with_suffix('_inverted',
                                      subdir=subdir)  # type: XfmAtom
    s = CmdStage(inputs=(xfm,), outputs=(inv_xfm,),
                 cmd=['xfminvert', '-clobber', xfm.path, inv_xfm.path])

    return Result(stages=Stages([s]), output=inv_xfm)


def invert_xfmhandler(xfm: XfmHandler,
                      subdir: str = "transforms") -> Result[XfmHandler]:
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
    inv_xfm = s.defer(xfminvert(xfm.xfm, subdir=subdir))  # type: XfmAtom
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
    blur_factor=5,
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
               files at either 0.056mm or 0.04mm resolution. For now we will
               determine the value for the simplex by multiplying the 
               resolution by 20.
        
    There are a number of parameters that have to be set and this 
    will be done using factors that depend on the resolution of the
    input files. Here is the list:
        
    argument to be set   --  default (factor)  -- (for 56 micron, translates to)
            blur                     5                    (230 micron)
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
    blurred_src = s.defer(mincblur(source, blur_stepsize)).img
    blurred_dest = s.defer(mincblur(target, blur_stepsize)).img

    if not output_name_wo_ext:
        output_name_wo_ext = "%s_rot_mt_to_%s" % (source.filename_wo_ext, target.filename_wo_ext)

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
    return LinearMinctraccConf(objective=LinearObjectiveFn.xcorr,
                               simplex=1,  # TODO simplex=1 -> simplex_factor=20?
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
    objective=NonlinearObjectiveFn.corrcoeff,
    lattice_diameter=None,
    sub_lattice=6)


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

def parse_minctracc_linear_protocol_file(filename : str, transform_type : LinearTransType,
                                         minctracc_conf=default_linear_minctracc_conf) \
        -> MultilevelMinctraccConf:
    with open(filename, 'r') as f:
        return parse_minctracc_linear_protocol(f=csv.reader(f, delimiter=';'),
                                               base_minctracc_conf=minctracc_conf, transform_type=transform_type)

def parse_minctracc_nonlinear_protocol_file(filename : str,
                                            minctracc_conf=default_nonlinear_minctracc_conf):
    with open(filename, 'r') as f:
        return parse_minctracc_nonlinear_protocol(csv.reader(f, delimiter=';'), minctracc_conf)


def thrice_result(f):
    def g(x):
        y = f(x)
        return (y, y, y)  # good question
    return g


def parse_minctracc_protocol(f, base_minctracc_conf, parsers, names,
                             is_ignored_key : Callable[[str], bool],
                             modify : Callable[[MinctraccConf, Any], MinctraccConf]):
    params = list(parsers.keys())

    # build a mapping from (Python, not file) field names to a list of values (one for each generation)
    d = {}
    for l in f:
        k, *vs = l
        if k not in params:
            if is_ignored_key(k):
                print("Warning: key '%s' not used" % k)
            else:
                raise ParseError("Unrecognized parameter: %s" % k)
        else:
            new_k = names[k] if k in names else k
            if new_k in d:
                raise ParseError("Duplicate key: %s/%s" % (k, new_k))
            else:
                d[new_k] = [parsers[k](v) for v in vs]

    # some error checking ...
    if not all_equal(d.values(), by=len):
        raise ParseError("Invalid minctracc configuration: "
                         "all params must be given for the same number of generations.")
    if len(d) == 0:
        raise ParseError("Empty file ...")   # TODO should this really be an error?
    for k in d.copy():  # otherwise d changes size during iteration ...
        if is_ignored_key(k):
            print("Warning: don't currently use '%s'..." % k)  # doesn't have to be same length -> can crash code below
            del d[k]

    vs = list(d.values())
    l = len(vs[0])

    def convert_single_gen(single_gen_params) -> NonlinearMinctraccConf:  # TODO name this better ...
        # TODO check for/catch IndexError ... a bit hard to use zip since some params may not be defined ...
        attrs = { k : v
                  for k, v in single_gen_params.items()
                  if k not in ('blur', 'step', 'gradient')}

        conf  = base_minctracc_conf.replace(**attrs)  # FIXME this is rather unsafe
        return modify(MinctraccConf(blur_resolution=single_gen_params["blur"],
                             use_gradient=single_gen_params["gradient"],
                             step_sizes=(single_gen_params["step"],) * 3,
                             use_masks=True, #FIXME
                             linear_conf=None,
                             nonlinear_conf=None), conf)

    return MultilevelMinctraccConf([convert_single_gen({ key : vs[j] for key, vs in d.items() }) for j in range(l)])


def parse_minctracc_nonlinear_protocol(f,
                                       base_minctracc_conf : NonlinearMinctraccConf = default_nonlinear_minctracc_conf)\
        -> MultilevelMinctraccConf:
    # parsers to use for each row of the protocol file
    parsers = {"blur"               : float,
               "step"               : float,  # use thrice_result here too?
               "gradient"           : parse_bool,
               "iterations"         : int,
               "lattice_diameter"   : thrice_result(float),
               "optimization"       : lambda o: True if o == "-use_simplex" \
                                                     else raise_(NotImplementedError("optimization %s" % o)),
               "simplex"            : float,
               "w_rotations"        : thrice_result(float),  # old protocols have this, but just gets deleted anyway ...
               "w_translations"     : thrice_result(float),
               "w_scales"           : thrice_result(float),
               "w_shear"            : thrice_result(float),
               "stiffness"          : float,
               "weight"             : float,
               "similarity"         : float}

    # mapping from names in the config file to NonlinearMinctraccConf field names
    names = { "optimization" : "use_simplex" }
    return parse_minctracc_protocol(f=f,
                                    parsers=parsers,
                                    names=names,
                                    base_minctracc_conf=default_nonlinear_minctracc_conf,
                                    is_ignored_key=lambda k: k.startswith("w_") or (k in ["memory_required", "simplex"]),
                                    modify=lambda b, c: b.replace(nonlinear_conf=c))


def parse_minctracc_linear_protocol(f, transform_type : LinearTransType,
                                    base_minctracc_conf : LinearMinctraccConf = default_linear_minctracc_conf) \
        -> MultilevelMinctraccConf:
    parsers = {"blur"               : float,
               "objective"          : lambda x: LinearObjectiveFn[x],
               "step"               : float,
               "gradient"           : parse_bool,
               "simplex"            : float,
               "transform_type"     : lambda x: LinearTransType[x],  # TODO: add readable exception
               "tolerance"          : float,
               "w_rotations"        : thrice_result(float),
               "w_translations"     : thrice_result(float),
               "w_scales"           : thrice_result(float),
               "w_shear"            : thrice_result(float)}

    is_ignored_key = lambda k: k in ("memory_required", "iterations", "optimization")

    return parse_minctracc_protocol(f=f, names={}, parsers=parsers, is_ignored_key=is_ignored_key,
                                    modify=lambda b, c: b.replace(linear_conf=c),
                                    base_minctracc_conf=base_minctracc_conf)


MinctraccMemCfg = NamedTuple("MinctraccMemCfg",
                             [('base_mem', float),
                              ('mem_per_voxel', float)])

default_minctracc_mem_cfg = MinctraccMemCfg(base_mem=3e-1, mem_per_voxel=6e-7)


# TODO: add memory estimation hook
def minctracc(source: MincAtom,
              target: MincAtom,
              conf: MinctraccConf,
              transform: Optional[XfmAtom] = None,
              transform_name_wo_ext: Optional[str] = None,
              transform_info: Optional[List[str]] = None,
              generation: Optional[int] = None,
              subdir: Optional[str] = None,
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

    subdir = subdir if subdir is not None else 'transforms'

    if lin_conf is None and nlin_conf is None:
        raise ValueError("minctracc: no linear or nonlinear configuration specified")

    if lin_conf is not None and lin_conf.transform_type not in LinearTransType.__members__.values():
        raise ValueError("minctracc: invalid transform type %s" % lin_conf.transform_type)
    # TODO: probably not needed since we're using an enum
    if transform_name_wo_ext:
        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, subdir,
                                            "%s.xfm" % (transform_name_wo_ext)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)
    # the generation provided can be 0 (zero), but that's a proper generation,
    # so we should explicitly test for "is not None" here.
    elif generation is not None:
        if lin_conf:
            trans_type = lin_conf.transform_type.name
        else:
            trans_type = "nlin"

        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, subdir,
                                            "%s_mt_to_%s_%s_%s.xfm" %
                                            (source.filename_wo_ext,
                                             target.filename_wo_ext,
                                             trans_type,
                                             generation)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)
    else:
        out_xfm = XfmAtom(name=os.path.join(source.pipeline_sub_dir, source.output_sub_dir, subdir,
                                            "%s_mt_to_%s.xfm" % (
                                            source.filename_wo_ext, target.filename_wo_ext)),
                          pipeline_sub_dir=source.pipeline_sub_dir,
                          output_sub_dir=source.output_sub_dir)

    source_for_minctracc = source
    target_for_minctracc = target
    if conf.blur_resolution is not None:
        img_or_grad = lambda result: result.gradient if conf.use_gradient else result.img
        source_for_minctracc = img_or_grad(s.defer(mincblur(source, conf.blur_resolution)))
        target_for_minctracc = img_or_grad(s.defer(mincblur(target, conf.blur_resolution)))

    # NOTE: this is broken in the presence of unanticipated (vs., e.g., nlin_conf.objective) null fields;
    # if we're not going to allow these, we should probably wrap this in a try/catch to give a better error
    # and/or start with a 'default' linear/nonlinear/both minctracc conf as appropriate and `replace` all
    # fields with the user-supplied ones.
    stage = CmdStage(cmd=['minctracc', '-clobber', '-debug']
                         + (['-%s' % lin_conf.objective.name] if lin_conf and lin_conf.objective else [])
                         + (['-transformation', transform.path] if transform
                            else (transform_info if transform_info else ["-identity"]))
                         + (['-' + lin_conf.transform_type.name]
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
                             + ['-weight', str(nlin_conf.weight)]
                             + ['-stiffness', str(nlin_conf.stiffness)]
                             + ['-sub_lattice', str(nlin_conf.sub_lattice)]
                             + (['-lattice_diameter']
                                + (space_sep(nlin_conf.lattice_diameter)
                                   if nlin_conf.lattice_diameter is not None
                                   else space_sep((3 * s for s in conf.step_sizes)))))
                            if nlin_conf is not None else [])
                         + (['-nonlinear %s' % (nlin_conf.objective.name if nlin_conf.objective else '')]
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

    if nlin_conf is not None:  # TODO at the moment basically ignore resource requirements for linear stages ...
        def set_memory(st, cfg):
            voxels = reduce(mul, volumeFromFile(source.path).getSizes())
            st.setMem(voxels * cfg.mem_per_voxel + cfg.base_mem)   # FIXME hard-coded 7 is a hack ...
            # TODO make a wrapper to generate these set_memory functions?

        stage.when_runnable_hooks.append(lambda st: set_memory(st, default_minctracc_mem_cfg))

    s.add(stage)

    # note accessing a None resampled field from an XfmHandler is an error
    # (by property magic, but would be similar using getters),
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
                                   ("radius_or_bins", int),
                                   ("use_gradient_image", bool)])


default_similarity_metric_conf = SimilarityMetricConf(
    metric="CC",
    weight=1.0,
    radius_or_bins=3,
    use_gradient_image=False)

MincANTSConf = NamedTuple("MincANTSConf",
                          [("file_resolution", float),
                           ("iterations", str),
                           ("transformation_model", str), # TODO make an enumeration so, e.g., users don't type "Syn"
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
    transformation_model="'SyN[0.1]'",
    regularization="'Gauss[2,1]'",
    use_mask=True,
    file_resolution=None,
    sim_metric_confs=[default_similarity_metric_conf,
                      default_similarity_metric_conf.replace(use_gradient_image=True)])  # type: MincANTSConf


def get_default_multi_level_mincANTS(file_resolution: float) -> MultilevelMincANTSConf:
    """
    Create a multilevel ANTS configuration based on the provided file resolution.
    The iterations are:
    100x100x100x0
    100x100x100x20
    100x100x100x100
    """
    conf1 = mincANTS_default_conf.replace(file_resolution=file_resolution,
                                          iterations="100x100x100x0")
    conf2 = mincANTS_default_conf.replace(file_resolution=file_resolution,
                                          iterations="100x100x100x20")
    conf3 = mincANTS_default_conf.replace(file_resolution=file_resolution,
                                          iterations="100x100x100x100")
    return MultilevelMincANTSConf([conf1,conf2,conf3])


def get_linear_configuration_from_options(conf, transform_type : LinearTransType, file_resolution : float) \
        -> MultilevelMinctraccConf:

    if conf.protocol:
        minctracc_conf = parse_minctracc_linear_protocol_file(
                           filename=conf.protocol,
                           transform_type=transform_type,
                           # we could allow overriding the default_linear_minctracc_conf even further up the call chain
                           minctracc_conf=default_linear_minctracc_conf(transform_type)
                                          # FIXME replace anything here based on file_resolution?
                                          # minctracc config uses factors, so should be OK.
                         )
    else:
        warnings.warn("No %s protocol specified -- using the defaults, which might not be what you want"
                      % transform_type)
        minctracc_conf = default_lsq12_multilevel_minctracc #.replace(...)
        # FIXME `replace` the resolution (<-> step factors, etc.), transform_type, etc.!!  Also print a warning?
        raise NotImplementedError("You need to supply a protocol at the moment.  Sorry!")

    return minctracc_conf


def get_nonlinear_configuration_from_options(nlin_protocol : Union[MincANTSConf, MinctraccConf],
                                             reg_method : str,
                                             file_resolution : float):
    """
    :param nlin_protocol: path to the protocol on the system (can be None)
    :param reg_method: the registration method (currently ANTS or minctracc)
    :param file_resolution: resolution at which registrations are performed
    :return:  MultilevelMincANTSConf or MultilevelMinctraccConf
    """

    # TODO maybe just take the whole nlin_conf as a param to save typing at the call site?

    # determine what configuration to use for the non linear registration
    if nlin_protocol:
        # actually parse it:
        if reg_method == "ANTS":
            non_linear_configuration = parse_mincANTS_protocol_file(nlin_protocol, file_resolution)
        elif reg_method == "minctracc":
            non_linear_configuration = parse_minctracc_nonlinear_protocol_file(nlin_protocol)
        else:
            raise ValueError("?!")
    else:
        # get one of the default configurations
        if reg_method == "ANTS":
            non_linear_configuration = get_default_multi_level_mincANTS(file_resolution=file_resolution)
        elif reg_method == "minctracc":
            #TODO: this. Still TODO.
            raise ValueError("Error.. we do not have proper minctracc nonlinear defaults yet. ")
        else:
            raise ValueError("?!")

    return non_linear_configuration


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

def parse_mincANTS_protocol_file(config_file,
                                 file_resolution,
                                 mincANTS_conf=mincANTS_default_conf) -> MultilevelMincANTSConf:
    """
    config_file -- path to file on the system that contains the ANTS configuration

    Use the resulting list to `.replace` the default values.
    """

    # parsers to use for each row of the protocol file
    parsers = {"blur"               : parse_many(parse_nullable(float)),
               "gradient"           : parse_many(parse_bool),
               "similarity_metric"  : parse_many(str),
               "weight"             : parse_many(float),
               "radius_or_histo"    : parse_many(int),
               "transformation"     : str,
               "regularization"     : str,
               "iterations"         : str,
               "useMask"            : bool,
               "memoryRequired"     : float}

    # mapping from protocol file names to Python field names of the ANTS and similarity metric configurations
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

    with open(config_file, 'r') as f:
        reader = csv.reader(f, delimiter=";")
        # build a mapping from (Python, not file) field names to a list of values (one for each generation)
        d = {}
        for l in reader:
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
        raise ParseError("Invalid ANTS configuration: all params must have the same number of generations.")
    if len(d) == 0:
        raise ParseError("Empty file ...")   # TODO should this really be an error?
    if "blur" in d:
        print("Warning: no longer using `blur` even though it's specified in the protocol ...")
        # TODO should be a warnings/logger.warning, not a print
        # TODO: why is this not being used anymore? It allows you to specify what you want
        # TODO: to do with the similarity metrics. Not sure whether we should have hard coded
        # TODO: defaults for this?
        del d["blur"]
    if "memory_required" in d:
        print("Warning: don't currently use the memory ...")  # doesn't have to be same length -> can crash code below
        del d["memory_required"]

    vs = list(d.values())
    l = len(vs[0])

    # convert a mapping of options to _single_ values to a single-generation ANTS configuration object:
    def convert_single_gen(single_gen_params, file_resolution) -> MincANTSConf:  # TODO name this better ...
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
                                             file_resolution=file_resolution,
                                             **other_attrs)

    full_configuration = MultilevelMincANTSConf([convert_single_gen({ key : vs[j] for key, vs in d.items() },
                                                                    file_resolution) for j in range(l)])

    return full_configuration


def all_equal(xs, by=lambda x: x):
    return len(set((by(x) for x in xs))) == 1


MincANTSMemCfg = NamedTuple("MincANTSMemCfg",
                            [('base_mem', float),
                             ('mem_per_voxel_coarse', float),
                             ('mem_per_voxel_fine', bool)])

default_mincANTS_mem_cfg = MincANTSMemCfg(base_mem=0.177, mem_per_voxel_coarse=1.385e-7, mem_per_voxel_fine=2.1e-7)


def ANTS(source: MincAtom,
         target: MincAtom,
         conf: MincANTSConf,
         transform_name_wo_ext: str = None,
         generation: int = None,
         resample_source: bool = False,
         subdir_for_resample: str = "resampled") -> Result[XfmHandler]:
    """
    ...
    transform_name_wo_ext -- to use for the output transformation (without the extension)
    generation            -- if provided, the transformation name will be:
                             source.filename_wo_ext + "_ANTS_nlin-" + generation
    resample_source       -- whether or not to resample the source file   
    
    Construct a single call to ANTS.
    Also does blurring according to the specified options
    since the cost function might use these.
    """
    s = Stages()

    # if we resample the source, and place it in the "tmp" directory, we should do
    # the same with the transformation that is created:
    trans_output_dir = "transforms"
    if resample_source and subdir_for_resample == "tmp":
        trans_output_dir = "tmp"

    if transform_name_wo_ext:
        name = os.path.join(source.pipeline_sub_dir, source.output_sub_dir, trans_output_dir,
                            "%s.xfm" % (transform_name_wo_ext))
    elif generation is not None:
        name = os.path.join(source.pipeline_sub_dir, source.output_sub_dir, trans_output_dir,
                            "%s_ANTS_nlin-%s.xfm" % (source.filename_wo_ext, generation))
    else:
        name = os.path.join(source.pipeline_sub_dir, source.output_sub_dir, trans_output_dir,
                            "%s_ANTS_to_%s.xfm" % (source.filename_wo_ext, target.filename_wo_ext))
    out_xfm = XfmAtom(name=name, pipeline_sub_dir=source.pipeline_sub_dir, output_sub_dir=source.output_sub_dir)

    similarity_cmds = []       # type: List[str]
    similarity_inputs = set()  # type: Set[MincAtom]
    # TODO: similarity_inputs should be a set, but `MincAtom`s aren't hashable
    for sim_metric_conf in conf.sim_metric_confs:
        if conf.file_resolution is not None and sim_metric_conf.use_gradient_image:
            src = s.defer(mincblur(source, fwhm=conf.file_resolution)).gradient
            dest = s.defer(mincblur(target, fwhm=conf.file_resolution)).gradient
        elif conf.file_resolution is None and sim_metric_conf.use_gradient_image:
            # the file resolution is not set, however we want to use the gradients
            # for this similarity metric...
            raise ValueError("A similarity metric in the ANTS configuration "
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
        inputs=(source, target) + tuple(similarity_inputs) + cast(tuple, ((source.mask,) if source.mask else ())),
        # need to cast to tuple due to mypy bug; see mypy/issues/622
        outputs=(out_xfm,),
        cmd=['ANTS', '3',
             '--number-of-affine-iterations', '0']
            + similarity_cmds
            + ['-t', conf.transformation_model,
               '-r', conf.regularization,
               '-i', conf.iterations,
               '-o', out_xfm.path]
            + (['-x', source.mask.path] if conf.use_mask and source.mask else []))


    def set_memory(st, mem_cfg):
        # see comments re: mincblur memory configuration
        voxels = reduce(mul, volumeFromFile(source.path).getSizes())
        mem_per_voxel = (mem_cfg.mem_per_voxel_coarse
                         if int(conf.iterations.split('x')[-1]) == 0  # yikes ... this parsing should be done earlier
                         else mem_cfg.mem_per_voxel_fine)
        st.setMem(mem_cfg.base_mem + voxels * mem_per_voxel)

    # see comments re: mincblur memory configuration
    stage.when_runnable_hooks.append(lambda st: set_memory(st, default_mincANTS_mem_cfg))

    s.add(stage)
    resampled = (s.defer(mincresample(img=source, xfm=out_xfm, like=target,
                                      interpolation=Interpolation.sinc,
                                      subdir=subdir_for_resample))
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


# TODO expand parameter list to be similar to mincANTS_NLIN_build_model, possible add resolution parameter?
def minctracc_NLIN_build_model(imgs: List[MincAtom],
                               initial_target: MincAtom,
                               conf: MultilevelMinctraccConf,
                               nlin_dir: str,
                               mincaverage = mincbigaverage) -> Result[WithAvgImgs[List[XfmHandler]]]:
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
                              nlin_prefix : str = "",
                              mincaverage = mincbigaverage) -> Result[WithAvgImgs[List[XfmHandler]]]:
    """
    This functions runs a hierarchical ANTS registration on the input
    images (imgs) creating an unbiased average.
    The ANTS configuration `confs` that is passed in should be
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
    # changed start=1 to start=0 (and i -> i+1 in average creation) to match old code:
    for i, conf_inst in enumerate(conf.confs, start=0):
        # in the following command we resample the output of the ANTS command. This is because
        # we create an average during each iteration which is used as the target for the next iteration.
        # However, we should not save all resampled files in the resampled/ directory (default for
        # the ANTS() call). Do this only for the last iteration:
        resampled_subdir = "resampled" if i == len(conf.confs) else "tmp"
        xfms = [s.defer(ANTS(source=img, target=avg, conf=conf_inst, generation=i,
                             resample_source=True, subdir_for_resample=resampled_subdir))
                for img in imgs]
        #  TODO make resampled name 'final-nlin' ?? need another option to ANTS for that, I guess ...
        # if no nlin_prefix is provided, we should remove the leading dash
        avg = s.defer(mincaverage([xfm.resampled for xfm in xfms],
                                  name_wo_ext='%s-nlin-%d' % (nlin_prefix, i+1) if nlin_prefix != "" else 'nlin-%d' % (i+1),
                                  output_dir=nlin_dir))
        avg_imgs.append(avg)
    return Result(stages=s, output=WithAvgImgs(output=xfms, avg_img=avg, avg_imgs=avg_imgs))

# nothing requires the linear conf to specify 12 params, so perhaps this should be called 'lin_nlin' ...
def lsq12_nlin(source: MincAtom,
               target: MincAtom,
               lsq12_conf: MinctraccConf,
               nlin_conf: Union[MultilevelMinctraccConf, MultilevelMincANTSConf, MincANTSConf],  # sigh ... ...
               resample_source: bool = True):
    """
    Runs a 12 parameter (or really any) linear registration followed by a nonlinear registration
    (minctracc or ANTS, depending on the supplied nonlinear configuration)
    from the source to the target. (I.e., this is *not* a model-building function.)
    It can be used, for instance, for intra-subject registrations in the registration_chain or in MAGeT
    """
    s = Stages()

    # This is strange.  The minctracc case requires a multilevel conf, while the ANTS case requires a single conf.
    # Of course, it's because a 'single' ANTS run may work at several resolutions iteratively; if we're not
    # constructing intermediate models, there's no need for separate calls.  However, this is annoying since
    # one would hope this function would similarly to the model-building version in being able to dispatch
    # on the configuration passed in (and in the same way, to avoid extra logic).

    if isinstance(nlin_conf, MultilevelMincANTSConf):
        # if only a sinle level is specified inside this multilevel configuration, all is fine:
        if len(nlin_conf.confs) == 1:
            # and we can just extract that single level
            nlin_conf = nlin_conf.confs[0]
        else:
            raise ValueError("The function lsq12_nlin was provided with a MultilevelMincANTSConf with more than 1 "
                             "level. This is a function performing a source to target registration, and thus should "
                             "only run a single level of ANTS (with its internal iterations option).")


    if isinstance(nlin_conf, MincANTSConf):
        lsq12_transform_handler = s.defer(multilevel_minctracc(source=source,
                                                               target=target,
                                                               conf=lsq12_conf,
                                                               resample_source=True))
        nlin_transform_handler = s.defer(ANTS(source=lsq12_transform_handler.resampled,
                                              target=target,
                                              conf=nlin_conf,
                                              resample_source=resample_source))
        full_transform = s.defer(concat_xfmhandlers(xfms=[lsq12_transform_handler, nlin_transform_handler],
                                                    name=source.filename_wo_ext + "_to_" +
                                                      target.filename_wo_ext + "_lsq12_mincANTS_nlin",
                                                    resample_source=resample_source))
    elif isinstance(nlin_conf, MultilevelMinctraccConf):
        lsq12_transform_handler = s.defer(multilevel_minctracc(source=source,
                                                               target=target,
                                                               conf=lsq12_conf,
                                                               # TODO allow a transform here?
                                                               resample_source=resample_source))
        nlin_transform_handler = s.defer(multilevel_minctracc(source=source,
                                                              target=target,
                                                              conf=nlin_conf,
                                                              transform=lsq12_transform_handler.xfm,
                                                              resample_source=resample_source))
        full_transform = nlin_transform_handler
    else:
        raise ValueError("Expected one of the following 3 options: 1) a MultilevelMincANTSConf with a single "
                         "level inside of it, 2) a MincANTSConf, or 3) a MultilevelMinctraccConf. The nlin_conf "
                         "that we got has type: " + type(nlin_conf))

    return Result(stages=s, output=full_transform)


# TODO obsolete...remove uses and definition?
def LSQ12_mincANTS_nlin(source: MincAtom,
                        target: MincAtom,
                        linear_conf: MinctraccConf,
                        nlin_conf: MincANTSConf):
    """
    Runs a 12 parameter registration followed by a non linear ANTS registration
    from the source to the target. This is a *non* model building function, which
    can be used for intra-subject registrations in the registration_chain, or in
    MAGeT for instance
    """
    # TODO: currently this only works for ANTS for the non linear part

    s = Stages()
    # we need to resample the source file in this case, because that will
    # be the input for the non linear stage
    lsq12_transform_handler = s.defer(multilevel_minctracc(source,
                                                           target,
                                                           conf=linear_conf,
                                                           resample_source=True))

    nlin_transform_handler = s.defer(ANTS(source=lsq12_transform_handler.resampled,
                                          target=target,
                                          conf=nlin_conf,
                                          resample_source=True))

    full_transform = s.defer(concat_xfmhandlers([lsq12_transform_handler,
                                                 nlin_transform_handler],
                                                name=source.filename_wo_ext + "_to_" +
                                                target.filename_wo_ext + "_lsq12_mincANTS_nlin"))
    return Result(stages=s, output=full_transform)


# some stuff for the registration chain.
# The Subject class moved here since `intrasubject_registrations` was also here.

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
        return ("Subject(intersubject_registration_time_pt: %s, time_pt_dict keys: %s ... (values not shown))"
                % (self.intersubject_registration_time_pt, self.time_pt_dict.keys()))


def intrasubject_registrations(subj: Subject,
                               linear_conf: MinctraccConf,
                               nlin_conf: MincANTSConf) \
        -> Result[Tuple[List[Tuple[int, int, XfmHandler]], int]]:
    """
    
    subj -- Subject (has a intersubject_registration_time_pt and a time_pt_dict 
            that maps timepoints to individual subjects

    Return:
    ([ (source_time_pt, target_time_pt, XfmHandler),
       (...,...,...),(...,...,...)],
     index_of_common_time_pt)
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
                        timepts_indices[source_index + 1],
                        s.defer(lsq12_nlin(source=src[1],
                                           target=dest[1],
                                           lsq12_conf=linear_conf,
                                           nlin_conf=nlin_conf,
                                           resample_source=True)))
                       for source_index, (src, dest) in enumerate(pairs(timepts))]
    return Result(stages=s, output=(time_pt_to_xfms, index_of_common_time_pt))


# def multilevel_registration(source, target, registration_function, conf, curr_dir, transform=None):
#    ...


# TODO: this is very static right now, but I just want to get things running

_lin_conf_1 = LinearMinctraccConf(objective=LinearObjectiveFn.xcorr,
                                  simplex=2.8,
                                  transform_type=LinearTransType.lsq12,
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
                         resample_source: bool = False) -> Result[XfmHandler]:
    if len(conf.confs) == 0:  # not a "registration" at all; also, src_blur/target_blur will be undefined ...
        raise ValueError("No configurations supplied")
    s = Stages()
    last_resampled = None
    for idx, conf_ in enumerate(conf.confs):
        transform_handler = s.defer(minctracc(source, target, conf=conf_,
                                              transform=transform,
                                              transform_info=transform_info if idx == 0 else None,
                                              generation=idx,
                                              subdir='tmp' if idx < len(conf.confs) - 1 else None,
                                              resample_source=resample_source))
        transform = transform_handler.xfm
        # why potentially throw away the resampled image?
        last_resampled = transform_handler.resampled if resample_source else None
    return Result(stages=s,
                  output=XfmHandler(xfm=transform,
                                    source=source,
                                    target=target,
                                    resampled=last_resampled))


def multilevel_pairwise_minctracc(imgs: List[MincAtom],
                                  conf: MultilevelMinctraccConf,
                                  # transforms : List[] = None,
                                  max_pairs : Optional[int],
                                  # max_pairs doesn't even make sense for a non-pairwise MinctraccConf,
                                  # suggesting that the pairwise conf being a list of confs is inappropriate
                                  like: MincAtom = None,
                                  output_dir_for_avg: str = ".",
                                  mincaverage = mincbigaverage,
                                  output_name_for_avg: str = None) -> Result[WithAvgImgs[List[XfmHandler]]]:
    """Pairwise registration of all images.
    max_pairs - number of images to register each image against. (Currently we might register against one fewer.)

    If no output_name_for_avg is supplied, the output name will be:

    avg_{transform_type}                          -- in case all the configurations have the same transformation type
    ave_{transform_type_1}_..._{transform_type_n} -- otherwise
    """
    s = Stages()

    if max_pairs is not None and max_pairs < 2:
        raise ValueError("must register at least two pairs")

    if len(imgs) < 2:
        raise ValueError("currently need at least two images")
        # otherwise 0 imgs passed to mincavg (could special-case)

    confs = conf.confs
    # the name of the average file that is produced by this function:
    if not output_name_for_avg:
        all_same_transform_type = True
        first_transform_type = confs[0].linear_conf.transform_type.name if confs[0].linear_conf else "nlin"
        alternate_name = "avg"
        for stage in confs:
            current_transform_type = stage.linear_conf.transform_type.name if stage.linear_conf else "nlin"
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
        # FIXME: do something about the configuration, currently it all seems a bit broken...
        # Interestingly it seems like it's actually quite important to include the registration
        # between the input image and itself. Imagine this toy example:
        # file 1: volume 1
        # file 2: volume 8
        # file 3: volume 64
        # if you only align file 1 to 2 and 3, it will end up having volume size 22.4
        # after averaging the scaling vectors (2,2,2) and (4,4,4) to be (2.82843,2.82843,2.82843)
        # but if we include the registration to itself, i.e. (1,1,1) we end up with a (2,2,2)
        # average transform and its volume will be 8. The same holds for all the other files.
        # --> volumes after alignment and averaging using only the other targets:
        # file 1: volume 22.4
        # file 2: volume 8
        # file 3: volume 2.7
        # --> volumes after alignment using other targets and itself:
        # file 1: volume 8
        # file 2: volume 8
        # file 3: volume 8
        xfms = [s.defer(multilevel_minctracc(src_img, target_img,
                                             conf=conf))
                for target_img in target_imgs]

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
        avg_xfms = [avg_xfm_from(img, target_imgs=gen.sample(imgs, max_pairs)) for img in imgs]
                      # FIXME might use one fewer image than `max_pairs`...

    final_avg = s.defer(mincaverage([xfm.resampled for xfm in avg_xfms], avg_file=final_avg))

    return Result(stages=s, output=WithAvgImgs(avg_imgs=[final_avg], avg_img=final_avg, output=avg_xfms))


""" Pairwise lsq12 registration, returning array of transforms and an average image
    Assumption: given that this is a pairwise registration, we assume that all the input
                files to this function have the same shape (i.e., resolution and dimension sizes).
                This allows us to use any of the input files as the likefile for resampling if
                none is provided. """
# TODO actually check the assumption in the above comment ...
# TODO all this does is call multilevel_pairwise_minctracc and then return an average; fold into that procedure?
# TODO eliminate/provide default val for resolutions, move resolutions into conf, finish conf ...
def lsq12_pairwise(imgs: List[MincAtom],
                   resolution: float,
                   #conf: MultilevelMinctraccConf,  # TODO: override transform_type field?
                   # FIXME the multilevel minctracc conf should still be programatically settable, but
                   # what to do if both it and a protocol file are supplied?
                   lsq12_conf: LSQ12Conf,
                   lsq12_dir: str,
                   create_qc_images: bool = True,
                   like: MincAtom = None,
                   mincaverage = mincbigaverage) -> Result[WithAvgImgs[List[XfmHandler]]]:

    minctracc_conf = get_linear_configuration_from_options(conf=lsq12_conf,
                                                           transform_type=LinearTransType.lsq12,
                                                           file_resolution=resolution)

    s = Stages()
    avgs_and_xfms = s.defer(multilevel_pairwise_minctracc(imgs=imgs, conf=minctracc_conf, like=like,
                                                          output_dir_for_avg=lsq12_dir, mincaverage=mincaverage,
                                                          max_pairs=lsq12_conf.max_pairs))

    if create_qc_images:
        s.defer(create_quality_control_images(imgs=[avgs_and_xfms.avg_img]
                                                   + [xfm.resampled for xfm in avgs_and_xfms.output],
                                              montage_output=os.path.join(lsq12_dir, "LSQ12_montage"),
                                              message="lsq12"))

    return Result(stages=s, output=avgs_and_xfms)

def pairwise_antsRegistration(imgs: List[MincAtom],
                              output_dir_for_avg: str = None,
                              output_name_wo_ext: str = None):
    s = Stages()

    if len(imgs) < 2:
        raise ValueError("currently need at least two images")

    if not output_name_wo_ext:
        output_name_wo_ext = "full_pairwise_nlin_antsRegistration"

    if not output_dir_for_avg:
        raise ValueError("Please specify an output directory for the final average in pairwise_antsRegistration.")

    final_avg = MincAtom(name=os.path.join(output_dir_for_avg, output_name_wo_ext + ".mnc"),
                         pipeline_sub_dir=output_dir_for_avg)

    def avg_nlin_xfm_from(src_img: MincAtom,
                          target_imgs: List[MincAtom]):
        # antsRegistration returns stages and two XfmHandlers
        # 1) source -> target, 2) target -> source
        # we only care about the first one here
        xfmHs = [s.defer(antsRegistration(source=src_img,
                                         target=target_img))[0]
                        for target_img in target_imgs]

        # xfmavg is not well defined on transformations that have
        # non linear grids associated with them. Turn all transformations
        # into grid file, average those and make a transformation for
        # the final average grid

        #xfmGrids = [s.defer(minc_displacement(single_xfmH)) for single_xfmH in xfmHs]
        xfms_from_xfmHs = [xfmH.xfm for xfmH in xfmHs]
        avg_xfmGrid = MincAtom(name=os.path.join(src_img.pipeline_sub_dir,
                                                 src_img.output_sub_dir,
                                                 "transforms",
                                                 "%s_avg_nlin_antsReg_grid.mnc" % src_img.filename_wo_ext),
                               pipeline_sub_dir=src_img.pipeline_sub_dir)

        #list_of_grids = [ "%s/%s_grid_0.mnc" % (xfmH.xfm.dir, xfmH.xfm.filename_wo_ext) for xfmH in xfmHs]

        #custom_minc_avg = CmdStage(inputs=tuple(xfms_from_xfmHs),
        #                           outputs=(avg_xfmGrid,),
        #                           cmd=['mincaverage', '-clobber'] +
        #                               sorted(list_of_grids) +
        #                               [avg_xfmGrid.path])
        # s.add(custom_minc_avg)

        # generate grid files from antsRegistration calls:
        minc_displacement_grids = [s.defer(minc_displacement(xfmH)) for xfmH in xfmHs]
        minc_avg_on_minc_displ = CmdStage(inputs=tuple(xfms_from_xfmHs) + tuple(minc_displacement_grids),
                                          outputs=(avg_xfmGrid,),
                                          cmd=['mincaverage', '-clobber'] +
                                              sorted([m_displ_grid.path for m_displ_grid in minc_displacement_grids]) +
                                              [avg_xfmGrid.path])
        s.add(minc_avg_on_minc_displ)


        #avg_xfmGrid = s.defer(mincaverage(imgs=xfmGrids,
        #                                  output_dir=os.path.join(src_img.pipeline_sub_dir,
        #                                                          src_img.output_sub_dir,
        #                                                          "transforms"),
        #                                  name_wo_ext="%s_avg_nlin_antsReg_grid" % src_img.filename_wo_ext))

        avg_xfm = XfmAtom(name=os.path.join(src_img.pipeline_sub_dir,
                                            src_img.output_sub_dir,
                                            "transforms",
                                            "%s_avg_nlin_antsReg.xfm" % src_img.filename_wo_ext),
                          pipeline_sub_dir=src_img.pipeline_sub_dir,
                          output_sub_dir=src_img.output_sub_dir)

        create_xfm_for_avgGrid = CmdStage(inputs=(avg_xfmGrid,),
                                          outputs=(avg_xfm,),
                                          cmd=(["make_xfm_for_grid.pl",
                                                "-clobber",
                                                avg_xfmGrid.path,
                                                avg_xfm.path]))
        s.add(create_xfm_for_avgGrid)

        res = s.defer(mincresample(img=src_img,
                                   xfm=avg_xfm,
                                   like=src_img,
                                   interpolation=Interpolation.sinc))
        return XfmHandler(xfm=avg_xfm,
                          source=src_img,
                          target=final_avg,
                          resampled=res)

    avg_xfmHs = [avg_nlin_xfm_from(src_img=img, target_imgs=imgs) for img in imgs]

    final_avg = s.defer(mincaverage(imgs=[xfmH.resampled for xfmH in avg_xfmHs],
                                    avg_file=final_avg))

    return Result(stages=s, output=WithAvgImgs(avg_imgs=[final_avg], avg_img=final_avg, output=avg_xfmHs))



def nlin_pairwise(imgs: List[MincAtom],
                  nlin_dir: str,
                  create_qc_images: bool = True) -> Result[WithAvgImgs[List[XfmHandler]]]:
    """

    :param img: a list of minc atoms for which a full pairwise non linear registration will be run
    :param nlin_dir: the directory that will hold the average non linear file: for each input file,
                     we compute the average non linear transformation towards all the other input
                     files. That average transformation is applied to the input, and we average
                     those files.
    :param create_qc_images:
    :return: and array of transformations (avg nlin towards all other files), and the average MincAtom
    """
    s = Stages()

    avgs_and_xfms = s.defer(pairwise_antsRegistration(imgs=imgs,
                                                      output_dir_for_avg=nlin_dir))

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
    res = s.defer(lsq12_pairwise(imgs=vs, lsq12_conf=conf, lsq12_dir=lsq12_dir, like=like))
    return Result(stages=s, output=WithAvgImgs(output=dict(zip(ks, res.output)),
                                               avg_imgs=res.avg_imgs,
                                               avg_img=res.avg_img))


def nlin_build_model(imgs           : List[MincAtom],
                     initial_target : MincAtom,
                     conf           : Union[MultilevelMinctraccConf, MultilevelMincANTSConf],
                     nlin_dir       : str,
                     nlin_prefix    : str = ""):
    s = Stages()

    if isinstance(conf, MultilevelMinctraccConf):
        nlin_result = s.defer(minctracc_NLIN_build_model(imgs=imgs,  # TODO add nlin_prefix to this one!!!
                                                         initial_target=initial_target,
                                                         conf=conf,
                                                         nlin_dir=nlin_dir))
        return Result(stages=s, output=nlin_result)
    elif isinstance(conf, MultilevelMincANTSConf):
        nlin_result = s.defer(mincANTS_NLIN_build_model(imgs=imgs,
                                                        initial_target=initial_target,
                                                        conf=conf,
                                                        nlin_prefix=nlin_prefix,
                                                        nlin_dir=nlin_dir))
        return Result(stages=s, output=nlin_result)
    else:
        # this should never happen
        raise ValueError("The nonlinear configuration passed to nlin_build_model is neither for minctracc nor for ANTS.")



def generate_halfway_transform(input_xfm_handler: XfmHandler,
                               like_file:         MincAtom) -> Result[XfmAtom]:
    """
    This function produces a transformation that moves halfway along
    the provided transform. It works as follows:
    1) combine all transforms into a single grid
    2) multiply the grid by 0.5
    3) create a .xfm file to point at the created grid file
    """
    s = Stages()

    # combine all transformations into a single grid:
    full_grid = s.defer(minc_displacement(input_xfm_handler))

    # multiply the vectors by 0.5
    halfway_grid = s.defer(mincmath(op='mult',
                                    const=0.5,
                                    vols=[full_grid],
                                    subdir="tmp",
                                    new_name=full_grid.filename_wo_ext + "_halfway"))

    # create an XfmAtom for this grid file, and
    # write its information to disk
    halfway_xfm = input_xfm_handler.xfm.newname_with_suffix(suffix="_halfway")
    # write the xfm file:
    create_xfm_text_file_cmd = CmdStage(inputs=(halfway_grid,),
                                        outputs=(halfway_xfm,),
                                        cmd=(["make_xfm_for_grid.pl",
                                              halfway_grid.path,
                                              halfway_xfm.path]))
    s.add(create_xfm_text_file_cmd)
    return Result(stages=s, output=halfway_xfm)



def nonlinear_midpoint_average(img_A: MincAtom,
                               img_B: MincAtom,
                               out_name_wo_ext: str,
                               out_dir: str) -> Result[MincAtom]:
    """
    :param img_A:
    :param img_B:
    :return: the midway point between img_A and img_B --> img_AB
    """
    s = Stages()

    # start with an antRegistration call between the two files
    xfm_handlers_antsReg = s.defer(antsRegistration(source=img_A,
                                                    target=img_B,
                                                    subdir='tmp'))

    # generate halfway transformations
    xfm_handler_A_to_B = xfm_handlers_antsReg[0]
    xfm_handler_B_to_A = xfm_handlers_antsReg[1]

    transform_A_to_B_halfway = s.defer(generate_halfway_transform(input_xfm_handler=xfm_handler_A_to_B,
                                                                  like_file=img_B))
    transform_B_to_A_halfway = s.defer(generate_halfway_transform(input_xfm_handler=xfm_handler_B_to_A,
                                                                  like_file=img_A))
    # resample A and B halfway:
    A_halfway_to_B = s.defer(mincresample(img=img_A,
                                          xfm=transform_A_to_B_halfway,
                                          like=img_B,
                                          subdir='tmp',
                                          interpolation=Interpolation.sinc))
    B_halfway_to_A = s.defer(mincresample(img=img_B,
                                          xfm=transform_B_to_A_halfway,
                                          like=img_A,
                                          subdir='tmp',
                                          interpolation=Interpolation.sinc))

    # the output file (avg of both files resampled to the midway point)
    avg_mid_point = s.defer(mincaverage(imgs=[A_halfway_to_B, B_halfway_to_A],
                                        output_dir=out_dir,
                                        name_wo_ext=out_name_wo_ext))

    return Result(stages=s, output=avg_mid_point)


def tournament_style_model(imgs                  : List[MincAtom],
                           tournament_dir        : str,
                           tournament_name_wo_ext: Optional[str] = "tournament") -> Result[MincAtom]:
    """
    generate an average of the input files based on a tournament style bracket,
    for example given 6 input images (1,2,3,4,5,6):

    1 ---|
         |---|
    2 ---|   |
             |---|
    3 ---|   |   |
         |---|   |
    4 ---|       |--- final average
                 |
        5 ---|   |
             |---|
        6 ---|


    :param imgs:
    :param tournament_dir:
    :return:
    """
    s = Stages()

    if len(imgs) == 1:
        # simply return this image:
        return Result(stages=s, output=imgs[0])
    else:
        # split up the list, calculate averages for the sublists and average
        midpoint_avg = s.defer(nonlinear_midpoint_average(img_A = s.defer(tournament_style_model(imgs=imgs[: math.floor(len(imgs)/2)],
                                                                                                 tournament_dir=tournament_dir,
                                                                                                 tournament_name_wo_ext=tournament_name_wo_ext + "_L")),
                                                          img_B = s.defer(tournament_style_model(imgs=imgs[math.floor(len(imgs)/2) :],
                                                                                                 tournament_dir=tournament_dir,
                                                                                                 tournament_name_wo_ext=tournament_name_wo_ext + "_R")),
                                                          out_name_wo_ext=tournament_name_wo_ext,
                                                          out_dir=tournament_dir))
        return Result(stages=s, output=midpoint_avg)


def antsRegistration(source: MincAtom,
                     target: MincAtom,
                     #conf: antsRegistrationConf,
                     transform_source: Optional[XfmAtom] = None,
                     transform_target: Optional[XfmAtom] = None,
                     transform_source_to_target_wo_ext: Optional[XfmAtom] = None,
                     generation: Optional[int] = None,
                     subdir: Optional[str] = None,
                     resample_source_and_target: Optional[bool] = False,
                     resample_subdir: Optional[str] = 'tmp'):
    """

    :param source: fixedImage
    :param target: movingImage
    :param conf:
    :param transform_source: for --initial-fixed-transform
    :param transform_target: for --initial-moving-transform
    :param transform_source_to_target_wo_ext: name of the target_to_source transform will be based on this
    :param generation:
    :param subdir:
    :param resample_source:
    :return:
    """
    s = Stages()

    source_subdir = subdir if subdir is not None else 'transforms'

    # Deal with the transformations first. This function will return two XfmHandlers. One
    # from source to target, and one in the other direction.
    if transform_source_to_target_wo_ext:
        xfm_source_to_target = XfmAtom(name=os.path.join(source.pipeline_sub_dir,
                                                         source.output_sub_dir,
                                                         source_subdir,
                                                         "%s.xfm" % (transform_source_to_target_wo_ext)),
                                       pipeline_sub_dir=source.pipeline_sub_dir,
                                       output_sub_dir=source.output_sub_dir)
    elif generation is not None:
        xfm_source_to_target = XfmAtom(name=os.path.join(source.pipeline_sub_dir,
                                                         source.output_sub_dir,
                                                         source_subdir,
                                                         "%s_antsR_to_%s_nlin_%s.xfm" %
                                                         (source.filename_wo_ext,
                                                          target.filename_wo_ext,
                                                          generation)),
                                       pipeline_sub_dir=source.pipeline_sub_dir,
                                       output_sub_dir=source.output_sub_dir)
    else:
        xfm_source_to_target = XfmAtom(name=os.path.join(source.pipeline_sub_dir,
                                                         source.output_sub_dir,
                                                         source_subdir,
                                                         "%s_antsR_to_%s.xfm" %
                                                         (source.filename_wo_ext,
                                                          target.filename_wo_ext)),
                                       pipeline_sub_dir=source.pipeline_sub_dir,
                                       output_sub_dir=source.output_sub_dir)
    # model the target_to_source in a similar manner. Given that
    # antsRegistration will be passed the "output prefix" for the transform,
    # being the whole filename with .xfm, this transform will live in a
    # directory belonging to the source image.
    # TODO: is this what we want? perhaps we actually want to move this transformation
    # over to a subdirectory of the target file...
    xfm_target_to_source = xfm_source_to_target.newname_with_suffix("_inverse", subdir='tmp')

    # run full command

    # outputs from antRegistration are:
    #   {output_prefix}_grid_0.mnc
    #   {output_prefix}.xfm
    #   {output_prefix}_inverse_grid_0.mnc
    #   {output_prefix}_inverse.xfm

    # Outputs:
    # 1) transform from source_to_target
    # 2) transform from target_to_source --> need to create an XfmHandler for this one

    # TODO: use a proper configuration to set the parameters
    # TODO: add a second metric for the gradients (and get gradient files)
    cmd_ants_reg = CmdStage(
        inputs=(source, target, source.mask, target.mask) if source.mask and target.mask else (source, target),
        outputs=(xfm_source_to_target, xfm_target_to_source),
        cmd=['antsRegistration'] \
            + ['--dimensionality', '3'] \
            + ['--verbose'] \
            + ['--minc'] \
            + ['--collapse-output-transforms', '1'] \
            + ['--write-composite-transform'] \
            + ['--winsorize-image-intensities', '[0.01,0.99]'] \
            + ['--use-histogram-matching', '1'] \
            + ['--float', '0'] \
            + ['--output', '[' + xfm_source_to_target.dir + '/' + xfm_source_to_target.filename_wo_ext + ']'] \
            + ['--transform', 'SyN[0.5,3,0]'] \
            + ['--convergence', '[100x100x100x100x50x10,1e-6,10]'] \
            + ['--metric', 'CC[' + source.path + ',' + target.path + ',1,6]'] \
            + (['--masks', '[' + source.mask.path + ',' +
                target.mask.path + ']'] if source.mask.path and target.mask.path else []) \
            + ['--shrink-factors', '16x8x6x4x2x1'] \
            + ['--smoothing-sigmas', '8x4x3x2x1x0vox'])
    #TODO: memory estimation
    s.add(cmd_ants_reg)

    # create file names for the two output files. It's better to use our standard
    # mincresample command for this, because that also deals with any associated
    # masks, whereas antsRegistration would only resample the input files.
    resampled_source = (s.defer(mincresample(img=source,
                                             xfm=xfm_source_to_target,
                                             like=target,
                                             interpolation=Interpolation.sinc,
                                             subdir=resample_subdir))
                        if resample_source_and_target else None)
    resampled_target = (s.defer(mincresample(img=target,
                                             xfm=xfm_target_to_source,
                                             like=source,
                                             interpolation=Interpolation.sinc,
                                             subdir=resample_subdir))
                        if resample_source_and_target else None)

    # return an XfmHandler for both the forward and the inverse transformations
    return Result(stages=s,
                  output=(XfmHandler(source=source,
                                     target=target,
                                     xfm=xfm_source_to_target,
                                     resampled=resampled_source),
                          XfmHandler(source=target,
                                     target=source,
                                     xfm=xfm_target_to_source,
                                     resampled=resampled_target)))


def lsq12_nlin_build_model(imgs       : List[MincAtom],
                           lsq12_conf : LSQ12Conf,
                           lsq12_dir  : str,
                           nlin_dir   : str,
                           nlin_conf  : Union[MultilevelMinctraccConf, MultilevelMincANTSConf],
                           resolution : float,
                           nlin_prefix : str = "") -> Result[WithAvgImgs[List[XfmHandler]]]:
    """
    Runs both a pairwise lsq12 registration followed by a non linear
    registration procedure on the input files.
    """
    s = Stages()

    # TODO: make sure that we pass on a correct configuration for lsq12_pairwise
    # it should be able to get this passed in....
    lsq12_result = s.defer(lsq12_pairwise(imgs=imgs, like=None,
                                          resolution=resolution,
                                          lsq12_conf=lsq12_conf,
                                          lsq12_dir=lsq12_dir))

    # extract the resampled lsq12 images
    lsq12_resampled_imgs = [xfm_handler.resampled for xfm_handler in lsq12_result.output]

    if lsq12_conf.generate_tournament_style_lsq12_avg:
        target_for_nlin_stages = s.defer(tournament_style_model(imgs=lsq12_resampled_imgs,
                                                                tournament_dir=lsq12_dir[0:len(lsq12_dir)-6] + "_tournament"))
    else:
        target_for_nlin_stages = lsq12_result.avg_img

    nlin_result = s.defer(nlin_build_model(imgs=lsq12_resampled_imgs,
                                           initial_target=target_for_nlin_stages,
                                           conf=nlin_conf,
                                           nlin_dir=nlin_dir,
                                           nlin_prefix=nlin_prefix))

    # concatenate the transformations from lsq12 and nlin before returning them
    from_imgs_to_nlin_xfms = [s.defer(concat_xfmhandlers(xfms=[lsq12_xfmh, nlin_xfmh],
                                                         name=img.filename_wo_ext + "_lsq12_and_nlin")) for
                              lsq12_xfmh, nlin_xfmh, img in zip(lsq12_result.output,
                                                                nlin_result.output,
                                                                imgs)]

    return Result(stages=s, output=WithAvgImgs(output=from_imgs_to_nlin_xfms,
                                               avg_img=nlin_result.avg_img,
                                               avg_imgs=nlin_result.avg_imgs))

def lsq12_nlin_pairwise(imgs       : List[MincAtom],
                        lsq12_conf : LSQ12Conf,
                        lsq12_dir  : str,
                        nlin_dir   : str,
                        nlin_conf  : Union[MultilevelMinctraccConf, MultilevelMincANTSConf],
                        resolution : float,
                        nlin_prefix : str = "") -> Result[WithAvgImgs[List[XfmHandler]]]:
    """
    Runs both a pairwise lsq12 registration followed by a non linear
    registration procedure on the input files.
    """
    s = Stages()

    # TODO: make sure that we pass on a correct configuration for lsq12_pairwise
    # it should be able to get this passed in....
    lsq12_result = s.defer(lsq12_pairwise(imgs=imgs, like=None,
                                          resolution=resolution,
                                          lsq12_conf=lsq12_conf,
                                          lsq12_dir=lsq12_dir))

    # extract the resampled lsq12 images
    lsq12_resampled_imgs = [xfm_handler.resampled for xfm_handler in lsq12_result.output]

    if lsq12_conf.generate_tournament_style_lsq12_avg:
        raise ValueError("You specified both --generate-tournament-style-lsq12-avg and --nlin-pairwise. "
                         "When the non linear alignment is performed using full pairwise registrations, "
                         "the lsq12 average is never used as a target. As such it does not make any "
                         "sense to spend a lot of computation time in building a crisp 12 parameter "
                         "average. Use either --generate-tournament-style-lsq12-avg together with "
                         "--no-nlin-pairwise, or if you want to run the full pairwise non linear "
                         "registrations use --no-generate-tournament-style-lsq12-avg together with "
                         "--nlin-pairwise")

    nlin_result = s.defer(nlin_pairwise(imgs=lsq12_resampled_imgs,
                                            nlin_dir=nlin_dir))

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
    # FIXME: opening and closing this file many times might be quite slow on some platforms ...
    # better to change to can_read_MINC_files.
    with open(os.devnull, 'w') as dev_null:
        ok = subprocess.call(["mincinfo", filename], stdout=dev_null, stderr=dev_null) == 0
    return ok


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
        issuesWithInputs = False
        for inputF in args:
            if not can_read_MINC_file(inputF):  # FIXME this will be quite slow on SciNet, etc.
                print("\nError: can not read input file: " + str(inputF) + "\n", file=sys.stderr)
                issuesWithInputs = True
        if issuesWithInputs:
            raise ValueError("\nIssues reading input files.\n")
    # lastly we should check that the actual filenames are distinct, because
    # directories are made based on the basename

    duplicates_exist = False
    seen = set()  # type: Set[str]
    for inputF in args:
        fileBase = os.path.splitext(os.path.basename(inputF))[0]
        if fileBase in seen:
            warnings.warn("\nThe following name occurs at least twice in the "
                          "input file list:\n" + str(fileBase) + ".mnc")
            duplicates_exist = True
        seen.add(fileBase)
    if duplicates_exist:
        raise ValueError("Please provide unique names for all input files")


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
         conf: LSQ6Conf,
         resample_subdir: str = "resampled",
         resample_images: bool = False,
         post_alignment_xfm: XfmAtom = None,
         propagate_masks: bool = True,
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
                           linear_conf=(default_linear_minctracc_conf(LinearTransType.lsq6)
                                        .replace(w_translations=[defaults["translations"][i]] * 3,
                                                 simplex=defaults["simplex_factors"][i] * resolution)),
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
            mt_conf = parse_minctracc_linear_protocol_file(filename=conf.protocol_file,
                                                           transform_type=LinearTransType.lsq6,
                                                           minctracc_conf=default_lsq6_minctracc_conf)
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
            mt_conf = parse_minctracc_linear_protocol_file(filename=conf.protocol_file,
                                                           transform_type=LinearTransType.lsq6,
                                                           minctracc_conf=default_lsq6_minctracc_conf)
        else:
            mt_conf = conf_from_defaults(defaults)  # FIXME print a warning?!

        xfms_to_target = [s.defer(multilevel_minctracc(source=img,
                                                       target=target,
                                                       conf=mt_conf))
                          for img in imgs]
    else:
        raise ValueError("bad lsq6 method: %s" % conf.lsq6_method)

    if post_alignment_xfm:
        composed_xfms = [s.defer(xfmconcat([xfm.xfm, post_alignment_xfm],
                                           name=xfm.xfm.output_sub_dir + "_lsq6"))
                         for xfm in xfms_to_target]
        resampled_imgs = ([s.defer(mincresample(img=native_img,
                                                xfm=overall_xfm,
                                                like=post_alignment_target,
                                                interpolation=Interpolation.sinc,
                                                new_name_wo_ext=native_img.filename_wo_ext + "_lsq6",
                                                subdir=resample_subdir))
                           for native_img, overall_xfm in zip(imgs, composed_xfms)]
                          if resample_images else [None] * len(imgs))
        final_xfmhs = [XfmHandler(xfm=overall_xfm,
                                  target=post_alignment_target,
                                  source=native_img,
                                  # resample the input to the final lsq6 space
                                  # we should go back to basics in terms of the file name that we create here.
                                  # It should be fairly basic. Something along the lines of:
                                  # {orig_file_base}_resampled_lsq6.mnc
                                  # if we are still going to perform either non uniformity correction, or
                                  # intensity normalization, the following file is a temp file:
                                  resampled=resampled_img)
                       for native_img, resampled_img, overall_xfm in zip(imgs, resampled_imgs, composed_xfms)]
    else:
        final_xfmhs = xfms_to_target
        if resample_images:
            for native_img, xfm in zip(imgs, final_xfmhs):  # is xfm.resampled even mutable ?!
                xfm.resampled = s.defer(mincresample(img=native_img,
                                                     xfm=xfm.xfm,
                                                     like=xfm.target,
                                                     interpolation=Interpolation.sinc,
                                                     new_name_wo_ext=native_img.filename_wo_ext + "_lsq6",
                                                     subdir=resample_subdir))

    # we've just performed a 6 parameter alignment between a bunch of input files
    # and a target. The input files could have been the very initial input files to the
    # pipeline, and have no masks associated with them. In that case, and if the target does
    # have a mask, we should add masks to the resampled files now.
    if propagate_masks and resample_images:
        mask_to_add = post_alignment_target.mask if post_alignment_target else target.mask
        for xfm in final_xfmhs:
            if not xfm.resampled.mask:
                xfm.resampled.mask = mask_to_add

    # could have stuff for creating an average and/or QC images here, but seemingly we use this either
    # as part of lsq6_nuc_inorm or in more unusual situations such as the registration chain where
    # we first aggregate the results of multiple lsq6 calls and create a common montage, so haven't bothered ...

    # TODO return average, etc.?
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
                   lsq6_dir: str,
                   create_qc_images: bool = True,
                   create_average: bool = True,
                   subject_matter: Optional[str] = None):
    s = Stages()

    # run the actual 6 parameter registration
    init_target = registration_targets.registration_native or registration_targets.registration_standard

    source_imgs_to_lsq6_target_xfms = s.defer(lsq6(imgs=imgs, target=init_target,
                                                   resolution=resolution,
                                                   conf=lsq6_options,
                                                   resample_images=not (lsq6_options.nuc or lsq6_options.inormalize),
                                                   post_alignment_xfm=registration_targets.xfm_to_standard,
                                                   post_alignment_target=registration_targets.registration_standard))

    xfms_to_final_target_space = [xfm_handler.xfm for xfm_handler in source_imgs_to_lsq6_target_xfms]

    # resample the mask from the initial model to native space
    # we can use it for either the non uniformity correction or
    # for intensity normalization later on
    masks_in_native_space = None  # type: List[MincAtom]
    if registration_targets.registration_standard.mask and \
        (lsq6_options.nuc or lsq6_options.inormalize):
        # we should apply the non uniformity correction in
        # native space. Given that there is a mask, we should
        # resample it to that space using the inverse of the
        # lsq6 transformation we have so far
        masks_in_native_space = [s.defer(mincresample(img=registration_targets.registration_standard.mask,
                                                      xfm=xfm_to_lsq6,
                                                      like=native_img,
                                                      interpolation=Interpolation.nearest_neighbour,
                                                      invert=True))
                                 for native_img, xfm_to_lsq6 in zip(imgs, xfms_to_final_target_space)]

    # NUC
    nuc_imgs_in_native_space = None  # type: List[MincAtom]
    if lsq6_options.nuc:
        # if masks are around, they will be passed along to nu_correct,
        # if not we create a list with the same length as the number
        # of images with None values
        # what we get back here is a list of MincAtoms with NUC files
        # we will always apply a final resampling to these files,
        # so these will always be temp files
        nuc_imgs_in_native_space = [s.defer(nu_correct(src=native_img,
                                                       resolution=resolution,
                                                       mask=native_img_mask,
                                                       subject_matter=subject_matter,
                                                       subdir="tmp"))
                                    for native_img, native_img_mask
                                    in zip(imgs,
                                           masks_in_native_space if masks_in_native_space
                                                                 else [None] * len(imgs))]

    inorm_imgs_in_native_space = None  # type: List[MincAtom]
    if lsq6_options.inormalize:
        # TODO: this is still static
        inorm_conf = default_inormalize_conf
        input_imgs_for_inorm = nuc_imgs_in_native_space if nuc_imgs_in_native_space else imgs
        # same as with the NUC files, these intensity normalized files will be resampled
        # using the lsq6 transform no matter what, so these ones are temp files
        inorm_imgs_in_native_space = (
            [s.defer(inormalize(src=nuc_img,
                                conf=inorm_conf,
                                mask=native_img_mask,
                                subdir="tmp"))
             for nuc_img, native_img_mask in zip(input_imgs_for_inorm,
                                                 masks_in_native_space or [None] * len(input_imgs_for_inorm))])

    # the only thing left to check is whether we have to resample the NUC/inorm images to LSQ6 space:
    if lsq6_options.inormalize:
        # the final resampled files should be the normalized files resampled with the 
        # lsq6 transformation
        final_resampled_lsq6_files = [s.defer(mincresample(
                                                img=inorm_img,
                                                xfm=xfm_to_lsq6,
                                                like=registration_targets.registration_standard,
                                                interpolation=Interpolation.sinc,
                                                new_name_wo_ext=inorm_img.filename_wo_ext + "_lsq6",
                                                subdir="resampled"))
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
                                                           new_name_wo_ext=nuc_filename_wo_ext,
                                                           subdir="resampled"))
                                      for nuc_img, xfm_to_lsq6, nuc_filename_wo_ext
                                      in zip(nuc_imgs_in_native_space,
                                             xfms_to_final_target_space,
                                             nuc_filenames_wo_ext_lsq6)]
    else:
        # in this case neither non uniformity correction nor intensity normalization was applied,
        # so we must have passed `resample_source=True` to the actual lsq6 calls above, and thus:
        final_resampled_lsq6_files = [xfm.resampled for xfm in source_imgs_to_lsq6_target_xfms]

    # we've just performed a 6 parameter alignment between a bunch of input files
    # and a target. The input files could have been the very initial input files to the
    # pipeline, and have no masks associated with them. In that case, and if the target does
    # have a mask, we should add masks to the resampled files now.
    # TODO potentially done twice if neither nuc nor inorm corrections applied since `lsq6` also does this
    mask_to_add = registration_targets.registration_standard.mask
    for resampled_input in final_resampled_lsq6_files:
        if not resampled_input.mask:
            resampled_input.mask = mask_to_add

    if create_average:
        if lsq6_options.copy_header_info:
            s.defer(mincaverage(imgs=final_resampled_lsq6_files,
                                output_dir=lsq6_dir,
                                copy_header_from_first_input=True))
        else:
            s.defer(mincbigaverage(imgs=final_resampled_lsq6_files,
                                   output_dir=lsq6_dir))

    if create_qc_images:
        s.defer(create_quality_control_images(imgs=final_resampled_lsq6_files,
                                              #montage_dir=lsq6_dir,  # FIXME
                                              montage_output=os.path.join(lsq6_dir, "LSQ6_montage")))

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

    # check how many options have been specified that can be used as the initial target
    number_of_target_options = sum((bootstrap is not False,
                                    init_model is not None,
                                    lsq6_target is not None,
                                    pride_of_models is not None))

    no_target_msg = ("Error: please specify a target for the 6 parameter alignment. "
                     "Options are: --lsq6-target, --init-model, --bootstrap, and for some pipelines --pride-of-models.")
    too_many_target_msg = ("Error: please specify only one of the following options: "
                           "--lsq6-target, --init-model, --bootstrap, --pride-of-models. "
                           "Don't know which target to use...")

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
        return get_registration_targets_from_init_model(target_file,
                                                        output_dir=output_dir,
                                                        pipeline_name=pipeline_name)
    else:
        raise ValueError("Invalid target type: %s" % lsq6_conf.target_type)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_pride_of_models_mapping(pride_csv: str,
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
    mapping_file = open(pride_csv, 'r')
    mapping_file_reader = csv.reader(mapping_file)
    headers = next(mapping_file_reader)
    # there are two column names that need to be present in the csv
    # file, and those are: "model_file" and "time_point"
    if not "model_file" in headers or not "time_point" in headers:
        raise ValueError("Error: the csv file that indicates the mapping "
                         "between time points and model files (" +
                         str(pride_csv) + ") does not contain (all) the "
                         "required column names: model_file and time_point.")

    model_file_index = headers.index("model_file")
    time_point_index = headers.index("time_point")

    common_resolution = None

    pride_of_models_dict = {}
    for row in mapping_file_reader:
        # ensure that the timepoints are given in integers/floats
        time_point = row[time_point_index]
        if not is_number(time_point):
            raise ValueError("Error: the specified time point is not an "
                             "integer nor a float: " + str(time_point) +
                             " In the following row:\n" + str(row))

        model_in_standard_space = row[model_file_index]
        if not common_resolution:
            common_resolution = get_resolution_from_file(model_in_standard_space)
        else:
            if not common_resolution == get_resolution_from_file(model_in_standard_space):
                raise ValueError("Error: we currently require all the initial models in the "
                                 "pride of models to have the same resolution. The resolution of "
                                 "this file: " + str(model_in_standard_space) + " (" +
                                 str(get_resolution_from_file(model_in_standard_space)) + ") is different "
                                 "from the resolution we found so far: " + str(common_resolution))

        pride_of_models_dict[float(time_point)] = get_registration_targets_from_init_model(init_model_standard_file=model_in_standard_space,
                                                                                           output_dir=output_dir,
                                                                                           pipeline_name=pipeline_name,
                                                                                           timepoint=time_point)
    mapping_file.close()
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
            memory=1,
            procs=1)
        s.add(mincpik_stage)
        individualImages.append(img_verification)


        # add a label to each of the individual images
        # so it will be easier for the user to identify
        # which images potentially fail
        img_verification_convert = img.newname_with_suffix("_QC_image_labeled",
                                                           subdir="tmp",
                                                           ext=".png")
        # FIXME: no other stages depend on
        # these, but we do want them to finish as soon as possible
        # -- perhaps we could instead return the montage stage (or, if no montage is to be created,
        # an empty stage) from this whole procedure and add it as in input (or better, a non-input dependency,
        # which isn't currently supported) to succeeding stages as desired?
        convert_stage = CmdStage(
            inputs=(img_verification,),
            outputs=(img_verification_convert,),
            cmd=["convert",
                 "-label", img.output_sub_dir,
                 img_verification.path,
                 img_verification_convert.path],
            memory=1,
            procs=1)
        s.add(convert_stage)
        individualImagesLabeled.append(img_verification_convert)


    # if montageOutput is specified, create the overview image
    if create_montage:
        montage_output_fileatom = FileAtom("%s.png" % montage_output)

        montage_stage = CmdStage(
            inputs=tuple(individualImagesLabeled),
            outputs=(montage_output_fileatom,),
            cmd=["montage", "-geometry", "+2+2"] +
                [labeled_img.path for labeled_img in individualImagesLabeled] +
                [montage_output_fileatom.path],
            memory=1,
            procs=1)
        montage_stage.set_log_file(os.path.join(os.path.dirname(montage_output_fileatom.path),
                                                "log",
                                                montage_output_fileatom.filename_wo_ext + ".log"))
        message_to_print = ("\n\n*\n* * *\n* * * * *\n* * * * * * *\nPlease consider the following verification "
                            "image, showing %s. \n%s\n* * * * * * *\n* * * * *\n* * *\n*\n" % (message, montage_output_fileatom.path))

        montage_stage.when_finished_hooks.append(
            lambda _: print(message_to_print))

        s.add(montage_stage)

    # TODO return some output images ?
    return Result(stages=s, output=None)


def optional(truthy, val):
    if truthy:
        return [val]
    else:
        return []


def param2xfm(out_xfm, center=None, translation=None, rotations=None, scales=None, shears=None):
    s = CmdStage(inputs=(), outputs=(out_xfm,),
                 cmd=["param2xfm", "-clobber"]
                     + flatten(*[(optional(x, [s, x]))
                        for s, x in
                        [("-center", center),
                         ("-translation", translation),
                         ("-rotations", rotations),
                         ("-scales", scales),
                         ("-shears", shears)]])
                     + [out_xfm.path])
    return Result(stages=Stages([s]), output=out_xfm)


class FlipAxis(AutoEnum):
    # named so that <...>.name returns the appropriate string to pass to volflip
    x = y = z = ()


def volflip(img : MincAtom, axis : FlipAxis = None):
    # N.B.: it's currently the users of a `volflip`ped volume's responsibility to add that volume's mask/labels
    # to their inputs if used (as with mincresample itself; minctracc/ANTS have to do this ...)
    s = Stages()
    flipped = img.newname_with_suffix(suffix="_flipped")
    if img.mask is not None:
        flipped.mask = s.defer(volflip(img.mask, axis=axis))
    if img.labels is not None:
        flipped.labels = s.defer(volflip(img.labels, axis=axis))
    stage = CmdStage(inputs=(img,), outputs=(flipped,),
                     cmd=["volflip", "-clobber", img.path, flipped.path] + (["-%s" % axis] if axis else []))
    s.add(stage)
    return Result(stages=s, output=flipped)


class LabelOp(AutoEnum):
    convert = merge = remap = select = remove = mask = binarize = ()


def minc_label_ops(in_labels : MincAtom, op : LabelOp, op_arg : Optional[Union[MincAtom, str]] = None):
    out_labels = NotImplementedError

    if op in { LabelOp.convert, LabelOp.binarize }:
        if op_arg is not None:
            raise ValueError("extraneous argument provided with %s op" % op)
    else:
        if op_arg is None:
            raise ValueError("op %s needs a value" % op)

    s = CmdStage(inputs=(in_labels,), outputs=(out_labels,),
                 cmd=["minc_label_ops"]
                     + ["--%s" % op]
                     + ([op_arg] if op_arg is not None else [])
                     + [in_labels.path, out_labels.path])
    return Result(stages=Stages([s]), output=out_labels)
