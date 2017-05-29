
import os
import warnings
from typing import Optional, Tuple, Sequence

from pydpiper.core.stages import Result, CmdStage, Stages
from pydpiper.core.util import flatten
import pydpiper.itk.tools  as itk #lgorithms, ToMinc
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.files import MincAtom, XfmAtom, NiiAtom
from pydpiper.minc.nlin import NLIN


#def transformix(img : ImgAtom, xfm : XfmAtom, out_dir: str): pass


class ITKToMinc(object):
    pass


def to_itk_xfm(xfm): raise NotImplementedError

def from_itk_xfm(xfm): raise NotImplementedError

class Algorithms(itk.Algorithms):
    @staticmethod
    def scale_transform(xfm): raise NotImplementedError

    def resample(img: I,
                 xfm: X,  # TODO: update to handler?
                 like: I,
                 invert: bool = False,
                 interpolation = None,   #interpolation: Interpolation = None,
                 #  TODO fix type for non-minc resampling programs; also, can't import Interpolation here
                 extra_flags: Sequence[str] = (),
                 new_name_wo_ext: str = None,
                 subdir: str = None,
                 postfix: str = None) -> Result[I]:
        raise NotImplementedError("use antsApplyTransform !")


    #@staticmethod
    #def


class Elastix(NLIN):

  Conf = str  # the filename

  MultilevelConf = Sequence[str]  # the filenames
  
  Algorithms = itk.Algorithms
  
  ToMinc = itk.ToMinc

  @staticmethod
  def hierarchical_to_single(c): return c

  @staticmethod
  def get_default_conf(resolution): return None

  @staticmethod
  def get_default_multilevel_conf(resolution): return None

  @classmethod
  def parse_protocol_file(cls, filename, resolution):
      p = cls.parse_multilevel_protocol_file(filename, resolution)
      if p.length() > 1:
          warnings.warn("too many confs")
      return p[-1]
      # the protocol is a list of elastix parameter files to pass (via -p <f1> -p <f2> ... -p <fn>)
      #with open(filename, 'r') as f:
      #    return f.readlines()# return filename
  # just pass the whole protocol to elastix for now
  # (might also want/need to read via simpleelastix in order to determine/set/override resolution, etc.)

  @classmethod
  def parse_multilevel_protocol_file(cls, filename, resolution):
      # a list of single level protocol files (each containing a list of elastix parameter files)
      with open(filename, 'r') as f:
          return [l.split(',') for l in f]

  @staticmethod
  def accepts_initial_transform(): return True

  @staticmethod
  def register(source: MincAtom,
               target: MincAtom,
               conf: Conf,
               initial_source_transform: Optional[XfmAtom] = None,
               transform_name_wo_ext: str = None,
               generation: int = None,  # not used; remove from API (fix ANTS)
               resample_source: bool = False,
               resample_subdir: str = "resampled") -> Result[XfmHandler]:
      out_dir = os.path.join(source.pipeline_sub_dir, source.output_sub_dir,
                             "%s_elastix_to_%s" % (source.filename_wo_ext, target.filename_wo_ext))

      # elastix chooses this for us:
      out_img = NiiAtom(name=os.path.join(out_dir, "result.%d.mnc" % 0), # TODO number of param files ?!?!
                        pipeline_sub_dir=source.pipeline_sub_dir,
                        output_sub_dir=source.output_sub_dir)
      #out_xfm = XfmAtom(name = "%s_elastix_to_%s.xfm" % (source.filename_wo_ext, target.filename_wo_ext),
      #                  pipeline_sub_dir=source.pipeline_sub_dir, output_sub_dir=source.output_sub_dir)
      out_xfm = XfmAtom(name=os.path.join(out_dir, "TransformParameters.%d.txt" % 0),  # TODO number of param files ?!?!
                        pipeline_sub_dir=source.pipeline_sub_dir,
                        output_sub_dir=source.output_sub_dir)
      cmd = (['elastix', '-f', source.path, '-m', target.path] + (flatten(*[["-p", x] for x in conf]))
              + (["-fMask", source.mask.path] if source.mask else [])
              + (["-mMask", target.mask.path] if target.mask else [])
              + (["-t0", initial_source_transform.path] if initial_source_transform else [])
              + (["-out", out_dir]))
      s = CmdStage(cmd=cmd,
                   inputs=(source, target)
                          + ((source.mask,) if source.mask else ())
                          + ((target.mask,) if target.mask else ()),
                   outputs = (out_xfm, out_img))

      #s2 = CmdStage(cmd=['transformix', '-out', os.path.join(resample_subdir, "%s" % c),
      #                   "-tp", out_xfm, "-in", out_name],
      #              inputs=(), outputs=())

      xfm = XfmHandler(source=source, target=target, xfm=out_xfm,
                       resampled=out_img)
      return Result(stages=Stages([s]), output=xfm)

# one question is whether we should have separate NLIN/LSQ12/LSQ6 interfaces or not, given that these differences seem
# like they should be rather irrelevant to general registration procedures ... at present minctracc
# is the main difficulty, since it uses different