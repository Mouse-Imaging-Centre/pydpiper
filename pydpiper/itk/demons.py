import os
import warnings
from typing import Sequence, Optional

#from pydpiper import itk
from pydpiper.core.files import ImgAtom
from pydpiper.core.stages import Result, CmdStage, Stages
import pydpiper.itk.tools as itk
from pydpiper.itk.tools import ITKXfmAtom
from pydpiper.minc.containers import GenericXfmHandler
from pydpiper.minc.nlin import NLIN


class DEMONS(NLIN):

  img_ext = ".nii.gz"
  xfm_ext = ".txt"  # also could be .xfm

  Conf = str

  MultilevelConf = Sequence[str]

  Algorithms = itk.Algorithms

  @staticmethod
  def hierarchical_to_single(c): return c

  @staticmethod
  def get_default_conf(resolution): return None

  @staticmethod
  def get_default_multilevel_conf(resolution): return [(), (), ()]

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

  @classmethod
  def register(cls,
               fixed: ImgAtom,
               moving: ImgAtom,
               conf: Conf,
               initial_moving_transform: Optional[ITKXfmAtom] = None,
               transform_name_wo_ext: str = None,
               generation: int = None,  # not used; remove from API (fix ANTS)
               resample_moving: bool = False,
               resample_subdir: str = "resampled") -> Result[GenericXfmHandler]:
      if conf is None:
      #    raise ValueError("no configuration supplied")
          warnings.warn("demons: no configuration supplied")

      # TODO this stuff is basically stolen from ANTS ... we should make some utility wrappers
      # instead of pasting these boring lines everywhere
      if resample_moving and resample_subdir == "tmp":
          trans_output_dir = "tmp"
      else:
          trans_output_dir = "transforms"

      # TODO instead of setting ext here manually, add to Algorithms/Types ... ?
      xfm_ext = "h5"
      if transform_name_wo_ext:
          name = os.path.join(moving.pipeline_sub_dir, moving.output_sub_dir, trans_output_dir,
                              f"{transform_name_wo_ext}.{xfm_ext}")
      elif generation is not None:
          name = os.path.join(moving.pipeline_sub_dir, moving.output_sub_dir, trans_output_dir,
                            f"%s_demons_nlin-%s.{xfm_ext}" % (moving.filename_wo_ext, generation))
      else:
          name = os.path.join(moving.pipeline_sub_dir, moving.output_sub_dir, trans_output_dir,
                              f"%s_demons_to_%s.{xfm_ext}" % (moving.filename_wo_ext, fixed.filename_wo_ext))
      out_xfm = ITKXfmAtom(name=name, pipeline_sub_dir=moving.pipeline_sub_dir, output_sub_dir=moving.output_sub_dir)

      out_img = moving.newname_with_suffix("_to_%s" % fixed.filename_wo_ext)

      cmd = (['DemonsRegistration', '--use-histogram-matching', '--verbose', '-f', fixed.path, '-m', moving.path]
              + (["--fixed-mask", fixed.mask.path] if fixed.mask else [])
              + (["--moving-mask", moving.mask.path] if moving.mask else [])
              + (["-p", initial_moving_transform.path] if initial_moving_transform else [])
              + ["-o", out_img.path, "-O", out_xfm.path])  # TODO more configuration
      s = CmdStage(cmd=cmd,
                   inputs=(fixed, moving)
                          + ((moving.mask,) if moving.mask else ())
                          + ((fixed.mask,) if fixed.mask else ()),
                   outputs = (out_xfm, out_img))

      xfm = GenericXfmHandler(source=moving, target=fixed, xfm=out_xfm,
                              resampled=out_img)
      return Result(stages=Stages([s]), output=xfm)