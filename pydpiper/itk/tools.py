import copy
import math
import os
import warnings
from typing import Optional, Sequence, List, Union, Tuple
from configargparse import Namespace
from dataclasses import dataclass

from pydpiper.core.util import AutoEnum
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.conversion import generic_converter
from pydpiper.minc.files import XfmAtom
from pydpiper.minc.nlin import Algorithms, NLIN
from pydpiper.core.stages import Result, CmdStage, Stages, identity_result
from pydpiper.core.files  import ImgAtom, FileAtom

import SimpleITK as sitk


# TODO delete ITK prefix?
class ITKXfmAtom(FileAtom):
    pass

class ITKImgAtom(ImgAtom):
    pass



def c3d_cmd(files, op, out_file):

    c = CmdStage(cmd = ["c3d"] + [f.path for f in files] + [op, '-o', out_file.path],
                 inputs = files, outputs=(out_file,))
    return Result(stages=Stages([c]), output=out_file)


def convert(infile : ImgAtom, out_ext : str) -> Result[ImgAtom]:
    s = Stages()
    outfile = infile.newext(ext=out_ext)
    if infile.mask is not None:
        outfile.mask = s.defer(convert(infile.mask, out_ext=out_ext))
    if infile.labels is not None:
        outfile.mask = s.defer(convert(infile.labels, out_ext=out_ext))
    s.add(CmdStage(inputs=(infile,), outputs=(outfile,),
                   cmd = ['c3d', infile.path, '-o', outfile.path]))
    return Result(stages=s, output=outfile)


def itk_convert_xfm(xfm : ITKXfmAtom, out_ext : str) -> Result[ITKXfmAtom]:
    if xfm.ext == out_ext:
        return identity_result(xfm)
    else:
        out_xfm = xfm.newext(out_ext)
        cmd = CmdStage(inputs=(xfm,), outputs=(out_xfm,),
                       cmd=["itk_convert_xfm", "--clobber", xfm.path, out_xfm.path])
        return Result(stages=Stages((cmd,)), output=out_xfm)


# TODO 'ITK' seems like a weird place for these; probably belong in minc;
# also, 'generic_converter' is - did I mention? - generic
mnc2nii = generic_converter(renamer = lambda img: img.newext(".nii"),
                            mk_cmd = lambda i, o: ["bash", "-c", "'rm %s.path; mnc2nii %s %s'" % (o, i, o)])

nii2mnc = generic_converter(renamer = lambda img: img.newext(".mnc"),
                            mk_cmd = lambda i, o: "nii2mnc -clobber {i} {o}".split())


class Interpolation(object):
    def render(self):
        return self.__class__.__name__

class Linear(Interpolation): pass
class NearestNeighbor(Interpolation): pass
class MultiLabel(Interpolation):  """TODO: add options"""
class Gaussian(Interpolation): """TODO: add options"""
class BSpline(Interpolation):
    def __init__(self, order=None):
        self.order = order
    def render(self):
        return self.__class__.__name__ + (("[order=%d]" % self.order) if self.order is not None else "")
class CosineWindowedSinc(Interpolation): pass
class WelchWindowedSinc(Interpolation): pass
class HammingWindowedSinc(Interpolation): pass
class LanczosWindowedSinc(Interpolation): pass


def as_deformation(transform, reference_image, interpolation: Interpolation = None,
                   invert: bool = None, dimensionality: int = None,
                   default_voxel_value: float = None, new_name_wo_ext: str = None,
                   subdir: str = None, ext: str = None) -> Result[ITKImgAtom]:
    """Convert an arbitrary ITK transformation to a deformation field representation.
    Consider this an image rather than a transformation since, e.g., AverageImages can be used."""

    if not subdir:
        subdir = 'tmp'

    ext = ext or ".nii.gz"

    if not new_name_wo_ext:
        new_name_wo_ext = os.path.basename(out_path(transform)) + "_def"
   #     ...
    #else:

    if not new_name_wo_ext and hasattr(transform, 'newname'):
        out_xfm = xfmToImage(transform.newname(name=new_name_wo_ext, subdir=subdir, ext=ext))
    else:
        out_xfm = xfmToImage(reference_image.newname(name=new_name_wo_ext, subdir=subdir, ext=ext))

    # TODO add rest of --output options
    cmd = (["antsApplyTransforms",
            "--reference-image", reference_image.path,
            "--output", "[%s,1]" % out_xfm.path]
           + [serialize_transform(transform if invert is None else InverseTransform(transform))]
           + (["--dimensionality", dimensionality] if dimensionality is not None else [])
           + (["--interpolation", interpolation.render()] if interpolation is not None else [])
           + (["--default-voxel-value", str(default_voxel_value)] if default_voxel_value is not None else []))
    s = CmdStage(cmd=cmd,
                 inputs=inputs(transform) + (reference_image,),
                 outputs=(out_xfm,))
    return Result(stages=Stages([s]), output=out_xfm)


# TODO: generalize to multiple transforms (see program name)
def antsApplyTransforms(img,
                        transform,
                        reference_image,
                        #outfile: str = None,
                        interpolation: Interpolation = None,
                        invert: bool = None,
                        dimensionality: int = None,
                        input_image_type = None,
                        output_warped_file: bool = None,
                        default_voxel_value: float = None,
                        #static_case_for_R: bool = None,
                        #float: bool = None
                        new_name_wo_ext: str = None,
                        subdir: str = None):

    if not subdir:
        subdir = 'resampled'

    if new_name_wo_ext:
        out_img = img.newname(name=new_name_wo_ext + ("-invert" if invert else ""), subdir=subdir)
    else:
        out_img = img.newname(name=transform.filename_wo_ext + ('-invert' if invert else "") + '-resampled',
                              subdir=subdir)

    transform = InverseTransform(transform) if invert else transform

    # TODO add rest of --output options
    cmd = (["antsApplyTransforms",
            "--verbose",
            "--input", img.path,
            "--reference-image", reference_image.path,
            "--output", out_img.path]
           #+ (["--transform", transform.path] if invert is None
           #   else ["-t", "[%s,%d]" % (transform.path, invert)])
           + [serialize_transform(transform)]
           + (["--dimensionality", dimensionality] if dimensionality is not None else [])
           + (["--interpolation", interpolation.render()] if interpolation is not None else [])
           + (["--default-voxel-value", str(default_voxel_value)] if default_voxel_value is not None else []))
    s = CmdStage(cmd=cmd,
                 inputs=(img, reference_image) + inputs(transform),
                 outputs=(out_img,))
    return Result(stages=Stages([s]), output=out_img)


def resample_simple(img, xfm, like,
                    invert = False,
                    use_nn_interpolation = None,
                    new_name_wo_ext = None,
                    subdir = None,
                    postfix = None):
    return antsApplyTransforms(img=img, transform=xfm, reference_image=like,
                               interpolation=MultiLabel() if use_nn_interpolation else None,
                               invert=invert, new_name_wo_ext=new_name_wo_ext, subdir=subdir)


def resample(img,
             xfm,  # TODO: update to handler?
             like,
             invert = False,
             use_nn_interpolation = None,
             new_name_wo_ext: str = None,
             subdir: str = None,
             resample_labels: bool = True,
             postfix: str = None):

    s = Stages()

    if not subdir:
        subdir = 'resampled'

    # we need to get the filename without extension here in case we have
    # masks/labels associated with the input file. When that's the case,
    # we supply its name with "_mask" and "_labels" for which we need
    # to know what the main file will be resampled as
    if not new_name_wo_ext:
        # FIXME this is wrong when invert=True
        if isinstance(xfm, ImgAtom):
            new_name_wo_ext = os.path.basename(out_path(xfm)) + "-resampled-" + postfix
        else:
            new_name_wo_ext = f"{os.path.basename(img.path)}_{os.path.basename(like.path)}-resampled-{postfix or ''}"
    new_img = s.defer(resample_simple(img=img, xfm=xfm, like=like,
                                      invert=invert,
                                      use_nn_interpolation=use_nn_interpolation,
                                      new_name_wo_ext=new_name_wo_ext,
                                      subdir=subdir))
    new_img.mask = s.defer(resample_simple(img=img.mask, xfm=xfm, like=like,
                                           use_nn_interpolation=True,
                                           invert=invert,
                                           new_name_wo_ext=new_name_wo_ext + "_mask",
                                           subdir=subdir)) if img.mask is not None else None
    new_img.labels = s.defer(resample_simple(img=img.labels, xfm=xfm, like=like,
                                             use_nn_interpolation=True,
                                             invert=invert,
                                             new_name_wo_ext=new_name_wo_ext + "_labels",
                                             subdir=subdir)) if resample_labels and img.labels is not None else None

    # Note that new_img can't be used for anything until the mask/label files are also resampled.
    # This shouldn't create a problem with stage dependencies as long as masks/labels appear in inputs/outputs of CmdStages.
    # (If this isn't automatic, a relevant helper function would be trivial.)
    # TODO: can/should this be done semi-automatically? probably ...
    return Result(stages=s, output=new_img)


# is c3d or ImageMath better for this (memory)?
def max(imgs : Sequence[ImgAtom], out_img : ImgAtom):
    if len(imgs) > 1:
        cmd = CmdStage(inputs = imgs, outputs = (out_img,),
                       cmd = (['c3d'] + [img.path for img in imgs]
                              + ['-accum', '-max', '-endaccum', '-o', out_img.path]))
        return Result(stages=Stages((cmd,)), output=out_img)
    elif len(imgs) == 1:
        # c3d needs at least two images with -accum
        return identity_result(imgs[0])
    else:
        raise ValueError("need at least one image")


def average_images(imgs        : Sequence[ImgAtom],
                   dimensions  : int = 3,
                   normalize   : bool = False,
                   output_dir  : str = '.',
                   name_wo_ext : str = "average",
                   out_ext     : Optional[str] = None,
                   robust      : Optional[bool] = None,
                   avg_file    : Optional[ITKImgAtom] = None) -> Result[ITKImgAtom]:

    s = Stages()

    if len(imgs) == 0:
        raise ValueError("`AverageImages` arg `imgs` is empty (can't average zero files)")

    if robust:
        warnings.warn("robust averaging not implemented")

    ext = out_ext or imgs[0].ext

    # the output_dir basically gives us the equivalent of the pipeline_sub_dir for
    # regular input files to a pipeline, so use that here
    avg = avg_file or ImgAtom(name=os.path.join(output_dir, '%s.todo' % name_wo_ext),
                              orig_name=None,
                              pipeline_sub_dir=output_dir)
    avg.ext = ext

    # if all input files have masks associated with them, add the combined mask to
    # the average:
    # TODO what if avg_file has a mask ... should that be used instead? (then rename avg -> avg_file above)
    all_inputs_have_masks = all((img.mask for img in imgs))
    if all_inputs_have_masks:
        combined_mask = (ImgAtom(name=os.path.join(avg_file.dir, '%s_mask.todo' % avg_file.filename_wo_ext),
                                 orig_name=None,
                                 pipeline_sub_dir=avg_file.pipeline_sub_dir)
                         if avg_file is not None else
                         ImgAtom(name=os.path.join(output_dir, '%s_mask.todo' % name_wo_ext),
                                 orig_name=None,
                                 pipeline_sub_dir=output_dir))
        combined_mask.ext = ext
        s.defer(max(imgs=sorted({img_inst.mask for img_inst in imgs}),
                    out_img=combined_mask))
        avg.mask = combined_mask
    s.add(CmdStage(inputs = imgs,
                   outputs = (avg,),
                   cmd = ["AverageImages", str(dimensions), avg.path, "%d" % normalize]
                         + [img.path for img in imgs]))
    return Result(stages=s, output=avg)

@dataclass
class Rigid:
    gradientStep: float
    def render(self):
        return f"Rigid[{self.gradientStep}]"
@dataclass
class Affine:
    gradientStep: float
    def render(self):
        return f"Affine[{self.gradientStep}]"
@dataclass
class Similarity:
    gradientStep: float
    def render(self):
        return f"Similarity[{self.gradientStep}]"
@dataclass
class AlignGeometricCentres:
    def render(self):
        return "AlignGeometricCenters"
@dataclass
class AlignCentresOfMass:
    def render(self):
        return "AlignCentersOfMass"

TransformType = Union[Rigid, Affine, Similarity, AlignGeometricCentres, AlignCentresOfMass]


@dataclass
class MI:
    fixed : ImgAtom
    moving : ImgAtom
    numberOfBins : Optional[int] = None
    # TODO add sampling
    def render(self):
        return (f"MI[{self.fixed.path},{self.moving.path}"
                + (f",{self.numberOfBins}" if self.numberOfBins else "") + "]")
class Mattes:
    fixed : ImgAtom
    moving : ImgAtom
    numberOfBins : Optional[int] = None
    def render(self):
        return (f"Mattes[{self.fixed.path},{self.moving.path}"
                + (f",{self.numberOfBins}" if self.numberOfBins else "") + "]")
class GC:
    """global correlation"""
    fixed : ImgAtom
    moving : ImgAtom
    radius : Optional[float] = None
    def render(self):
        return (f"GC[{self.fixed.path},{self.moving.path}"
                + (f",{self.radius}" if self.radius else "") + "]")

class ConvergenceParams:
    threshold : float
    window : int

class Convergence:
    iterations : int
    params : Optional[ConvergenceParams]
    def render(self):
        if self.params is not None:
            return f"[{self.iterations},{self.params.threshold},{self.params.window}]"
        else:
            return f"[{self.iterations}]"


#class MetricType(AutoEnum):
#    mi = mattes = gc = ()
def parse_metric(name):
    d = { 'MI' : MI, 'Mattes' : Mattes, 'GC' : GC }
    try:
        return d[name]
    except KeyError:
        return ValueError(f'unknown metric type {name}; need one of {d.keys()}')

AntsAIMetric = Union[MI, Mattes, GC]


def antsAI(metrics : List[AntsAIMetric],
           transform : TransformType,
           convergence : Convergence,
           fixedImageMask : Optional[ImgAtom] = None,
           movingImageMask : Optional[ImgAtom] = None,
           outputFileName : Optional[str] = None,
           out_xfm : Optional[XfmAtom] = None,
           transform_name_wo_ext : Optional[str] = None,
           dimensionality : Optional[int] = 3):   # antsAI claims not to need -d but fails otherwise
    # TODO add -p/-b (principal axis, blobs)!!!
    out_file = outputFileName or NotImplemented
    ex = metrics[0].moving

    trans_output_dir = "transforms"
    #if resample_source and resample_subdir == "tmp":
    #    trans_output_dir = "tmp"

    # antsAI claims to be able to generate only ITK .mat files
    if transform_name_wo_ext:
        name = os.path.join(ex.pipeline_sub_dir, ex.output_sub_dir, trans_output_dir,
                            f"{transform_name_wo_ext}.mat")
    else:
        name = os.path.join(
            ex.pipeline_sub_dir,
            ex.output_sub_dir, trans_output_dir,
            f"{metrics[0].moving.filename_wo_ext}_antsAI_to_{metrics[0].fixed.filename_wo_ext}.mat")
    out_xfm = out_xfm or ITKXfmAtom(name=name, pipeline_sub_dir=ex.pipeline_sub_dir, output_sub_dir=ex.output_sub_dir)

    if fixedImageMask:
        if movingImageMask:
            mask_str = f"[{fixedImageMask.path},{movingImageMask.path}]"
        else:
            mask_str = fixedImageMask.path
    elif movingImageMask:
        warnings.warn("antsAI seemingly can't use a movingImageMask without a fixedImageMask, ignoring")

    s = CmdStage(cmd = ["antsAI"] + ([f"-d {dimensionality}"] if dimensionality is not None else [])
                   + [f"-m {m.render()}" for m in metrics]
                   + [f"-t {transform.render()}"]
                   + ([f"-c {convergence.render()}"] if convergence is not None else [])
                   + ([f"-x {mask_str}"] if fixedImageMask is not None else [])
                   + [f"-o {out_xfm.path}"],
                 inputs = tuple(m.fixed for m in metrics)
                          + tuple(m.moving for m in metrics)
                          + ((fixedImageMask,) if fixedImageMask else ())
                          + ((movingImageMask,) if movingImageMask else ()),
                 outputs = (out_xfm,))
    return Result(stages=Stages([s]),
                  output=XfmHandler(fixed=metrics[0].fixed, moving=metrics[0].moving, xfm=out_xfm))
                  # TODO using metrics[0].fixed here is a hack -- seemingly we should refactor antsAI(...)
                  # to take fixed, moving images, but clearly there may not be a single fixed, moving due to multiple
                  # metrics, so generalization should be pursued (e.g. allowing fixed, moving to be lists of images)

@dataclass
class AntsAIConf:
    transform : TransformType
    metric : str
    convergence : Optional[Convergence] = None

# FIXME move up ?
from omegaconf import OmegaConf


def imageToXfm(i : ITKImgAtom) -> ITKXfmAtom:
    x = copy.deepcopy(i)
    x.__class__ = ITKXfmAtom
    del x.mask
    del x.labels
    return x

def xfmToImage(x : ITKXfmAtom):
    i = copy.deepcopy(x)
    i.__class__ = ITKImgAtom
    i.mask = None
    i.labels = None
    return i

@dataclass
class ConcatTransform:
    """Since ITK doesn't seem to have a file format for composite transforms,
     we simply store a list of them wrapped in this thin wrapper"""
    transforms: List['Transform']
    name: str

@dataclass
class IdentityTransform: pass

@dataclass
class InverseTransform:
    transform: 'Transform'

# the algebra of transformations:
Transform = Union[ITKXfmAtom, ConcatTransform, InverseTransform, IdentityTransform]


def canonicalize(t : Transform):
    """Push all inversions as far into the leaves as possible"""
    if isinstance(t, IdentityTransform):
        return t
    elif isinstance(t, InverseTransform):
        if type(t.transform) == InverseTransform:
            return canonicalize(t.transform)
        elif type(t.transform) == IdentityTransform:
            return IdentityTransform()
        elif type(t.transform) == ConcatTransform:
            return ConcatTransform(tuple(reversed([canonicalize(InverseTransform(s)) for s in t.transform.transforms])),
                                   name=t.transform.name + "-canon")
        else:
            return t
    elif isinstance(t, ConcatTransform):
        return ConcatTransform([canonicalize(t) for t in t.transforms], name = t.name)
    elif isinstance(t, FileAtom):
        return t
    else:
        raise TypeError(f"don't know what to do with this 'transform': {t}")



def serialize_transform(t : Transform):
    """Generate a string suitable for antsApplyTransforms"""
    t = canonicalize(t)
    if isinstance(t, IdentityTransform):
        return ""  # raise error / return None ?
    elif isinstance(t, InverseTransform):
        return f"-t [{t.transform.path},1]"
    elif isinstance(t, ConcatTransform):
        return " ".join([serialize_transform(u) for u in reversed(t.transforms)])
    elif isinstance(t, FileAtom):
        return f"-t {t.path}"
    else:
        raise TypeError(f"don't know what to do with this 'transform': {t}")

def inputs(t : Transform) -> Tuple[ITKXfmAtom]:
    if isinstance(t, FileAtom): return (t,)
    elif isinstance(t, IdentityTransform): return ()
    elif isinstance(t, InverseTransform): return inputs(t.transform)
    elif isinstance(t, ConcatTransform): return tuple(s for u in t.transforms for s in inputs(u))
    else: raise TypeError("?!")

def out_path(t : Transform) -> ITKXfmAtom:
    if hasattr(t, "path"):
        return t.path
    elif isinstance(t, IdentityTransform):
        return "id"
        #raise ValueError("shouldn't need to be here")
    elif isinstance(t, ConcatTransform):
        return t.name
    elif isinstance(t, InverseTransform):
        return t.transform.newname_with_suffix("_inv").path
    else:
        raise TypeError(f"{t}?")


# TODO move this
class Algorithms(Algorithms):

    @staticmethod
    def concat(ts, name):
        return identity_result(ConcatTransform(ts, name))

    @staticmethod
    def open(f):
        return sitk.ReadImage(f)

    @staticmethod
    def get_resolution_from_file(f):
        return min(sitk.ReadImage(f).GetSpacing())

    @staticmethod
    def average(*args, **kwargs): return average_images(*args, **kwargs)

    #@staticmethod
    #def compose_transforms(xfms): raise NotImplementedError
    #  CLI tool doesn't seem to exist in the ITK world, just keep list of transforms around?

    @staticmethod
    def nu_correct(img,
                   resolution,
                   mask,
                   # TODO add weights (-w)
                   subject_matter,
                   subdir="tmp"):

        out_img = img.newname_with_suffix("_N")

        if mask is None:
            warnings.warn("running N4 without a mask")

        cmd = CmdStage(cmd = ["N4BiasFieldCorrection", "-d 3", "-i", img.path, "-o", out_img.path] +
                             (["-x", mask.path] if mask is not None else []),
                       inputs = (img, mask) if mask is not None else (img,),
                       outputs = (out_img,))
        return Result(stages=Stages((cmd,)), output=out_img)

    @staticmethod
    def intensity_normalize(src, mask, subdir : str = "tmp"):
        warnings.warn("intensity normalization not implemented for ITK command-line tools; doing nothing")
        return identity_result(src)

    @staticmethod
    def hard_mask(img, *, mask, subdir = "tmp"):
        out_file = img.newname_with_suffix("_hard_masked", subdir=subdir)
        c = CmdStage(cmd = ["c3d", img, mask, "-mul", out_file.path], inputs=(img, mask), outputs=(out_file,))
        return Result(stages=Stages([c]), output=out_file)

    @staticmethod
    def dilate_mask(mask, voxels, subdir = "tmp"):
        out_file = mask.newname_with_suffix(f"_dilated_{voxels}", subdir=subdir)
        c = CmdStage(cmd = ["ImageMath", "3", out_file.path, "MD", str(voxels), mask.path])
        return Result(stages=Stages([c]), output=out_file)

    @staticmethod
    def label_vote(label_files, output_dir, name):
        # TODO change API to relative output dir
        if len(label_files) == 0:
            raise ValueError("can't vote with 0 files")
        elif len(label_files) == 1:
            return identity_result(label_files[0])  # ImageMath MajorityVoting silently fails with 1 image (as of 2.3.4)
        else:
            out_file = ImgAtom(os.path.join(output_dir, name))  # TODO get proper dir!!!
            c = CmdStage(cmd = ["ImageMath", "3", out_file.path, "MajorityVoting"] + [f.path for f in sorted(label_files)],
                         inputs = label_files, outputs = (out_file,))
            return Result(stages=Stages([c]), output=out_file)

    @staticmethod
    def union_mask(imgs, new_name, subdir):
        return c3d_cmd(imgs, op="-max", out_file = ImgAtom(new_name))

    @staticmethod
    def blur(img, fwhm, gradient=True, subdir='tmp'):
        # note c3d can take voxel rather than fwhm specification, but the Algorithms interface
        # currently doesn't allow this to be used ... maybe an argument from switching from mincblur
        if fwhm in (-1, 0, None):
            if gradient:
                raise ValueError("can't compute gradient without a positive FWHM")
            return identity_result(Namespace(img=img))

        if gradient:
            out_gradient = img.newname_with_suffix("_blur%s_grad" % fwhm)
        else:
            out_gradient = None

        # c3d -smooth takes stdev, not fwhm
        stdev = "%.3g" % (fwhm / (2 * math.sqrt(2 * math.log(2))))

        out_img = img.newname_with_suffix("_blurred%s" % stdev)

        cmd = CmdStage(cmd=['c3d', img.path, '-smooth', "%smm" % stdev, '-o', out_img.path]
                           + (['-gradient', '-o', out_gradient.path] if gradient else []),
                       inputs=(img,), outputs=(out_img, out_gradient) if gradient else (out_img,))
        return Result(stages=Stages((cmd,)),
                      output=Namespace(img=out_img, gradient=out_gradient)
                               if gradient else Namespace(img=out_img))

    @staticmethod
    def resample(*args, **kwargs): return resample(*args, **kwargs)

    @staticmethod
    def identity_transform(output_sub_dir=None): return identity_result(IdentityTransform())
    # TODO: does this ever need to be materialized?

    @staticmethod
    def scale_transform(xfm, scale, newname_wo_ext):
        """scale a nonlinear transformation.
           Note that a transformation, even a linear one, to which this is applied is converted to a deformation."""
        s = Stages()
        defs = s.defer(as_deformation(transform=xfm.xfm, reference=xfm.moving))
        scaled_defs = (defs.newname(newname_wo_ext) if newname_wo_ext else
                        defs.newname_with_suffix("_scaled_%s" % scale))
        s.defer(CmdStage(cmd=['c3d', '-scale', str(scale), defs.path, "-o", scaled_defs.path],
                         inputs=(defs,), outputs=(scaled_defs,)))
        return Result(stages=s, output=scaled_defs)

    @staticmethod
    def xfminvert(t, subdir = None):
        return Result(stages = Stages(),
                      output = InverseTransform(t) if not isinstance(t, InverseTransform) else t.transform)

    @staticmethod
    def average_affine_transforms(xfms, avg_xfm):
        #if not output_filename_wo_ext:
        #    output_filename_wo_ext = "average_xfm"
        #if all_from_same_sub:
        #    outf = xfms[0].newname(name=output_filename_wo_ext, subdir="transforms", ext=".xfm")
        # TODO why not use ANTs' AverageAffineTransform instead of this presumably similar script?
        #s = CmdStage(cmd=["AverageAffineTransforms"] + [x.xfm.path for x in xfms] + [avg_xfm.path],
        #             inputs = [x.xfm for x in xfms], outputs = (avg_xfm,))
        def render(t):
            t = canonicalize(t)
            if isinstance(t, FileAtom):
                return t.path
            elif isinstance(t, InverseTransform) and isinstance(t.transform, FileAtom):
                return f"-i {t.path}"
            else:
                raise ValueError("only a .mat or inverse of a .mat are supported")
        s = CmdStage(cmd=["AverageAffineTransform", "3", avg_xfm.path] + [render(t.xfm) for t in xfms],
                     inputs = tuple(x.xfm for x in xfms), outputs = (avg_xfm,))
        return Result(stages=Stages([s]), output=avg_xfm)

    @staticmethod
    def average_transforms(xfms, avg_xfm):
        s = Stages()
        defs = [s.defer(as_deformation(transform=xfm.xfm, reference_image=xfm.moving)) for xfm in xfms]
        #avg_img = NotImplemented
        avg = imageToXfm(s.defer(average_images(defs,
                                                avg_file=xfmToImage(avg_xfm),
                                                #output_dir=os.path.join(defs[0].pipeline_sub_dir,
                                                #                        defs[0].output_sub_dir,
                                                #                        "transforms")
                                                )))
        return Result(stages=s, output=avg)

    @staticmethod
    def log_determinant(xfm):
        s = Stages()
        displacement_field = s.defer(as_deformation(xfm.xfm, reference_image = xfm.moving))  # xfm.target?
        if hasattr(xfm.xfm, 'newname_with_suffix'):
            out_file = xfm.xfm.newname_with_suffix("_log_det")
        elif hasattr(xfm.xfm, 'name'):
            out_file = xfm.moving.newname_with_suffix(f"_to_{xfm.fixed}_log_det")
        s.add(CmdStage(cmd = ["CreateJacobianDeterminantImage", "3", displacement_field.path, out_file.path, "1", "0"],
                       inputs = (displacement_field,), outputs = (out_file,)))
        return Result(stages=s, output=out_file)

class AntsAI(NLIN):

    img_ext = ".nii.gz"
    xfm_ext = ".mat"

    Conf = AntsAIConf
    MultilevelConf = List[AntsAIConf]

    Algorithms = Algorithms

    @staticmethod
    def get_default_conf(resolution): return AntsAIConf(transform=Rigid(0.1), metric={'name' : 'MI'})

    @staticmethod
    def get_default_multilevel_conf(resolution):
        return [AntsAIConf(transform=Rigid(0.1), metric={'name' : "MI"})]

    def hierarchical_to_single(m): return m[-1]

    @staticmethod
    def accepts_initial_transform(): return False

    @classmethod
    def parse_protocol_file(cls, filename: str, resolution: float): return OmegaConf.load(filename)

    @classmethod
    def parse_multilevel_protocol_file(cls, filename: str, resolution: float): return OmegaConf.load(filename)

    @staticmethod
    def register(fixed,
                 moving, *,
                 conf,
                 resample_moving,
                 resample_subdir,
                 generation=None,
                 transform_name_wo_ext=None,
                 initial_moving_transform=None):
        metric_type = parse_metric(conf.metric['name'])
        params = conf.metric.copy()
        del params['name']
        metric = metric_type(fixed=fixed, moving=moving, **params)
        return antsAI(metrics=[metric],
                      transform=conf.transform,
                      convergence=conf.convergence,
                      fixedImageMask=fixed.mask,
                      movingImageMask=moving.mask,
                      outputFileName=None,
                      transform_name_wo_ext=transform_name_wo_ext)
