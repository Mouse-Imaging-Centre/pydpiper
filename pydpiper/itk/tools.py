import os
from typing import Optional, Sequence
from configargparse import Namespace

from pydpiper.minc.files import ToMinc
from pydpiper.minc.nlin import Algorithms
from pydpiper.core.stages import Result, CmdStage, Stages
from pydpiper.core.files  import ImgAtom, FileAtom


# TODO delete ITK prefix?
class ITKXfmAtom(FileAtom):
    pass


def convert(infile : ImgAtom, out_ext : str) -> Result[ImgAtom]:
    s = Stages()
    outfile = infile.newext(ext=out_ext)
    if infile.mask is not None:
        outfile.mask = s.defer(convert(infile.mask, out_ext=out_ext))
    if infile.labels is not None:
        outfile.mask = s.defer(convert(infile.labels, out_ext=out_ext))
    s.add(CmdStage(inputs=(infile,), outputs=(outfile,),
                   cmd = ['c3d', '-o', outfile.path, infile.path]))
    return Result(stages=s, output=outfile)


def itk_convert_xfm(xfm : ITKXfmAtom, out_ext : str) -> Result[ITKXfmAtom]:
    if xfm.ext == out_ext:
        return Result(stages=Stages(), output=xfm)
    else:
        out_xfm = xfm.newext(out_ext)
        cmd = CmdStage(inputs=(xfm,), outputs=(out_xfm,),
                       cmd=["itk_convert_xfm", xfm.path, out_xfm.path])
        return Result(stages=Stages((cmd,)), output=out_xfm)


# is c3d or ImageMath better for this (memory)?
def max(imgs : Sequence[ImgAtom], out_img : ImgAtom):
    cmd = CmdStage(inputs = imgs, outputs = (out_img,),
                   cmd = ['c3d'] + [img.path for img in imgs] + ['-accum', '-max', '-o', out_img.path])
    return Result(stages=Stages((cmd,)), output=out_img)


def average_images(imgs        : Sequence[ImgAtom],
                   dimensions  : int = 3,
                   normalize   : bool = False,
                   output_dir  : str = '.',
                   name_wo_ext : str = "average",
                   out_ext     : Optional[str] = None,
                   avg_file    : Optional[ImgAtom] = None):

    s = Stages()

    if len(imgs) == 0:
        raise ValueError("`AverageImages` arg `imgs` is empty (can't average zero files)")

    ext = out_ext or imgs[0].ext

    # the output_dir basically gives us the equivalent of the pipeline_sub_dir for
    # regular input files to a pipeline, so use that here
    avg = avg_file or ImgAtom(name=os.path.join(output_dir, '%s%s' % (name_wo_ext, ext)),
                              orig_name=None,
                              pipeline_sub_dir=output_dir)

    # if all input files have masks associated with them, add the combined mask to
    # the average:
    # TODO what if avg_file has a mask ... should that be used instead? (then rename avg -> avg_file above)
    all_inputs_have_masks = all((img.mask for img in imgs))
    if all_inputs_have_masks:
        combined_mask = (ImgAtom(name=os.path.join(avg_file.dir, '%s_mask%s' % (avg_file.filename_wo_ext, ext)),
                                 orig_name=None,
                                 pipeline_sub_dir=avg_file.pipeline_sub_dir)
                         if avg_file is not None else
                         ImgAtom(name=os.path.join(output_dir, '%s_mask%s' % (name_wo_ext, ext)),
                                 orig_name=None,
                                 pipeline_sub_dir=output_dir))
        s.defer(max(imgs=sorted({img_inst.mask for img_inst in imgs}),
                    out_img=combined_mask))
        avg.mask = combined_mask
    s.add(CmdStage(inputs = imgs,
                   outputs = (avg,),
                   cmd = ["AverageImages", str(dimensions), avg.path, "%d" % normalize]
                         + [img.path for img in imgs]))
    return Result(stages=s, output=avg)


class ToMinc(ToMinc):
    @staticmethod
    def to_mnc(img): return convert(img, out_ext=".mnc")
    @staticmethod
    def from_mnc(img): return convert(img, out_ext=".nii.gz")
    @staticmethod
    def to_mni_xfm(xfm): return itk_convert_xfm(xfm, out_ext=".mnc")
    @staticmethod
    def from_mni_xfm(xfm): return itk_convert_xfm(xfm, out_ext=".nii.gz")

# TODO move this
class Algorithms(Algorithms):
    average   = average_images
    @staticmethod
    def blur(img, fwhm, gradient=True, subdir='tmp'):
        # note c3d can take voxel rather than fwhm specification, but the Algorithms interface
        # currently doesn't allow this to be used ... maybe an argument from switching from mincblur
        if fwhm in (-1, 0, None):
            if gradient:
                raise ValueError("can't compute gradient without a positive FWHM")
            return Result(stages=Stages(), output=Namespace(img=img))

        if gradient:
            out_gradient = img.newname_with("_blur%s_grad" % fwhm)
        else:
            out_gradient = None

        out_img = img.newname_with("_blurred%s" % fwhm)

        cmd = CmdStage(cmd=['c3d', '-smooth', "%smm" % fwhm, '-o', out_img.path, img.path]
                           + (['-gradient', '-o', out_gradient.path] if gradient else []),
                       inputs=(img), outputs=(out_img, out_gradient) if gradient else (out_img,))
        return Result(stages=Stages((cmd,)),
                      output=Namespace(img=out_img, gradient=out_gradient)
                               if gradient else Namespace(img=out_img))
    scale_transform = NotImplemented

    average_transforms = NotImplemented