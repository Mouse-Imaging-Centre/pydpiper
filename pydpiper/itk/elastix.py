
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


#def as_deformation(xfm):
#    c = CmdStage(cmd=["transformix", "-def", "all", "-out", dirname, "-tp", xfm.path, "-xfm", out_path],
#                 inputs=(xfm,), outputs=NotImplemented)
#    return Result(stages=Stages([c]), output=NotImplemented)


def to_itk_xfm(xfm): raise NotImplementedError

def from_itk_xfm(xfm): raise NotImplementedError

class Algorithms(itk.Algorithms):
    @staticmethod
    def scale_transform(xfm, scale):
        scaled_xfm = xfm.newname_with_suffix("_scaled_%s" % scale)
        # don't actually evaluate the resulting vector field:
        s = CmdStage(cmd=["echo", ('(Transform "WeightedCombinationTransform")\n'
                                   '(SubTransforms %s)\n'
                                   '(Scales %s)\n') % (xfm.path, scale)],
                     inputs=(xfm,), outputs=(scaled_xfm,))
        return Result(stages=Stages([s]), output=scaled_xfm)
    def resample(img,
                 xfm,  # TODO: update to handler?
                 like,
                 invert = False,
                 use_nn_interpolation = None,
                 new_name_wo_ext: str = None,
                 subdir: str = None,
                 postfix: str = None):
        raise NotImplementedError("use transformix ?")

    @staticmethod
    def average_transforms(xfms, avg_xfm):
        intermediate_xfm = avg_xfm.newname_with_suffix("_inter", subdir="tmp")
        s = Stages()
        s.add(CmdStage(cmd=["echo", ('(Transform "WeightedCombinationTransform")\n'
                                     '(SubTransforms %s)\n'
                                     '(NormalizeCombinationsWeights "true")\n') %
                                       ' '.join(sorted(xfm.path for xfm in xfms))],
                       inputs=xfms, outputs=(intermediate_xfm,)))
        s.add(CmdStage(cmd=["transformix", "-def", "all",
                            "-out", os.path.dirname(avg_xfm.path),
                            "-tp", intermediate_xfm.path,
                            "-xfm", avg_xfm.path],
                       inputs=(intermediate_xfm,), outputs=(avg_xfm,)))

class ToMinc(itk.ToMinc):
    @staticmethod
    def to_mni_xfm(xfm):
        s = Stages()
        defs = xfm.newname_with_suffix("_defs", subdir="tmp")
        s.add(CmdStage(cmd=["transformix", "-def", "all",
                            "-out", defs.dir,
                            "-tp", xfm.path,
                            "-xfm", os.path.join(defs.filename_wo_ext, defs.ext)],
                       inputs=(xfm,), outputs=(defs,)))
        out_xfm = s.defer(itk.itk_convert_xfm(defs, out_ext=".mnc"))
        return Result(stages=s, output=out_xfm)

    @staticmethod
    def from_mni_xfm(xfm):
        raise NotImplemented("write a transformix parameter file")

class Elastix(NLIN):

  img_ext = xfm_ext = ".nii.gz"

  Conf = str  # the filename

  MultilevelConf = Sequence[str]  # the filenames
  
  Algorithms = itk.Algorithms
  
  ToMinc = ToMinc

  @staticmethod
  def hierarchical_to_single(c): return c

  @staticmethod
  def get_default_conf(resolution): return None

  @staticmethod
  def get_default_multilevel_conf(resolution): return None

  @classmethod
  def parse_protocol_file(cls, filename, resolution):
      p = cls.parse_multilevel_protocol_file(filename, resolution)
      # # TODO should just hierarchical_to_single(p) instead of p[-1] since using multiple confs
      # # makes sense for a single registration ??
      if p is None:   # silly hack
          return None
      if len(p) > 1:
          warnings.warn("too many confs; using the last one")
      return p[-1]
      #return p
      # the protocol is a list of elastix parameter files to pass (via -p <f1> -p <f2> ... -p <fn>)
      #with open(filename, 'r') as f:
      #    return f.readlines()# return filename
  # just pass the whole protocol to elastix for now
  # (might also want/need to read via simpleelastix in order to determine/set/override resolution, etc.)

  @classmethod
  def parse_multilevel_protocol_file(cls, filename, resolution):
      # a list of single level protocol files (each containing a list of elastix parameter files)
      with open(filename, 'r') as f:
          return [l.strip().split(',') for l in f if len(l.strip()) > 0]

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
      if conf is None:
          raise ValueError("no configuration supplied")

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
