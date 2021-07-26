import os
import warnings
from dataclasses import dataclass
from typing import Sequence, Optional, Tuple

from omegaconf import OmegaConf

from pydpiper.core.files import ImgAtom
from pydpiper.core.stages import Result, CmdStage, Stages
import pydpiper.itk.tools as itk
from pydpiper.itk.tools import ITKXfmAtom
from pydpiper.minc.containers import GenericXfmHandler
from pydpiper.minc.nlin import NLIN
from pydpiper.minc.registration import MincAlgorithms


@dataclass
class DemonsConfig:
    iterations: Optional[Tuple[int]] = None
    deformation_field_sigma: Optional[float] = None
    update_field_sigma: Optional[float] = None
    max_step_length: Optional[float] = None
    update_rule: Optional[int] = None
    gradient_type: Optional[int] = None
    verbosity: Optional[int] = None
    use_histogram_matching: Optional[bool] = None
    # TODO replace ints with enums here when possible


# empirically, seems important for succeess
default_demons_config = OmegaConf.structured(DemonsConfig(use_histogram_matching=True))


class DEMONS(NLIN):

  img_ext = ".nii.gz"
  xfm_ext = "nii.gz"  # also could be .xfm for a MINC-only registration suite

  Conf = DemonsConfig

  MultilevelConf = Sequence[DemonsConfig]

  Algorithms = itk.Algorithms

  @staticmethod
  def hierarchical_to_single(c):
      return c

  @staticmethod
  def get_default_conf(resolution): return default_demons_config

  @staticmethod
  def get_default_multilevel_conf(resolution): return [default_demons_config] * 3

  @classmethod
  def parse_protocol_file(cls, filename, resolution):
      return OmegaConf.load(filename)
      # p = cls.parse_multilevel_protocol_file(filename, resolution)
      # # # TODO should just hierarchical_to_single(p) instead of p[-1] since using multiple confs
      # # # makes sense for a single registration ??
      # if p is None:   # silly hack
      #     return None
      # if len(p) > 1:
      #     warnings.warn("too many confs; using the last one")
      # return p[-1]
      #return p
      # the protocol is a list of elastix parameter files to pass (via -p <f1> -p <f2> ... -p <fn>)
      #with open(filename, 'r') as f:
      #    return f.readlines()# return filename
  # (might also want/need to read via simpleelastix in order to determine/set/override resolution, etc.)

  @classmethod
  def parse_multilevel_protocol_file(cls, filename, resolution):
      return OmegaConf.load(filename).confs

  @staticmethod
  def accepts_initial_transform(): return False  # might work for some formats e.g. .h5 but not .xfm

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
          warnings.warn("demons: no configuration supplied")

      # TODO this stuff is basically stolen from ANTS ... we should make some utility wrappers
      # instead of pasting these boring lines everywhere
      if resample_moving and resample_subdir == "tmp":
          trans_output_dir = "tmp"
      else:
          trans_output_dir = "transforms"

      if transform_name_wo_ext:
          name = os.path.join(moving.pipeline_sub_dir, moving.output_sub_dir, trans_output_dir,
                              f"{transform_name_wo_ext}.{cls.xfm_ext}")
      elif generation is not None:
          name = os.path.join(moving.pipeline_sub_dir, moving.output_sub_dir, trans_output_dir,
                            f"%s_demons_nlin-%s.{cls.xfm_ext}" % (moving.filename_wo_ext, generation))
      else:
          name = os.path.join(moving.pipeline_sub_dir, moving.output_sub_dir, trans_output_dir,
                              f"%s_demons_to_%s.{cls.xfm_ext}" % (moving.filename_wo_ext, fixed.filename_wo_ext))
      out_xfm = ITKXfmAtom(name=name, pipeline_sub_dir=moving.pipeline_sub_dir, output_sub_dir=moving.output_sub_dir)

      # the Demons program can output a resampled image via `-o <resampled>` but this has histogram matching
      # applied to it if enabled and we seemingly need to enable it for Demons to work well, so resampled ourselves:
      #out_img = moving.newname_with_suffix("_to_%s" % fixed.filename_wo_ext, subdir="resampled")

      s = Stages()

      def option(flag, val):
          if val is not None:
              return [flag, str(val)]
          else:
              return []
      def option_flag(flag, b):
          if b:
              return [flag]
          else:
              return []

      cmd = (['DemonsRegistration', '-f', fixed.path, '-m', moving.path]
              + (["--fixed-mask", fixed.mask.path] if fixed.mask else [])
              + (["--moving-mask", moving.mask.path] if moving.mask else [])
              + (["-p", initial_moving_transform.path] if initial_moving_transform else [])
              #+ ["-o", out_img.path]
              + ["-O", out_xfm.path]
              + option_flag('--use-histogram-matching', conf.use_histogram_matching)
              + option('--def-field-sigma', conf.deformation_field_sigma)
              + option('--up-field-sigma', conf.update_field_sigma)
              + option('--max-step-length', conf.max_step_length)
              + option('--update-rule', conf.update_rule)
              + option('--gradient-type', conf.gradient_type)
              + option('--verbose', conf.verbosity))
      s.add(CmdStage(cmd=cmd,
                   inputs=(fixed, moving)
                          + ((moving.mask,) if moving.mask else ())
                          + ((fixed.mask,) if fixed.mask else ()),
                   outputs = (out_xfm,)))

      out_img = s.defer(cls.Algorithms.resample(img = moving, xfm = out_xfm, like = fixed, subdir='resampled'))

      xfm = GenericXfmHandler(moving=moving, fixed=fixed, xfm=out_xfm, resampled=out_img)

      return Result(stages=s, output=xfm)


class DEMONS_MINC(DEMONS):
  Algorithms = MincAlgorithms
  img_ext = ".mnc"
  xfm_ext = ".xfm"