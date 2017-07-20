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


class Demons(NLIN):

  Conf = str

  MultilevelConf = Sequence[str]

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
  def register(source: ImgAtom,
               target: ImgAtom,
               conf: Conf,
               initial_source_transform: Optional[ITKXfmAtom] = None,
               transform_name_wo_ext: str = None,
               generation: int = None,  # not used; remove from API (fix ANTS)
               resample_source: bool = False,
               resample_subdir: str = "resampled") -> Result[GenericXfmHandler]:
      if conf is None:
          raise ValueError("no configuration supplied")

      # TODO this stuff is basically stolen from ANTS ... we should make some utility wrappers
      # instead of pasting these boring lines everywhere
      if resample_source and resample_subdir == "tmp":
          trans_output_dir = "tmp"
      else:
          trans_output_dir = "transforms"

      # TODO instead of setting ext here manually, add to Algorithms/Types ... ?
      if transform_name_wo_ext:
          name = os.path.join(source.pipeline_sub_dir, source.output_sub_dir, trans_output_dir,
                              "%s.nii.gz" % (transform_name_wo_ext))
      else:
          name = os.path.join(source.pipeline_sub_dir, source.output_sub_dir, trans_output_dir,
                              "%s_demons_to_%s.nii.gz" % (source.filename_wo_ext, target.filename_wo_ext))
      out_xfm = ITKXfmAtom(name=name, pipeline_sub_dir=source.pipeline_sub_dir, output_sub_dir=source.output_sub_dir)

      out_img = source.newname_with_suffix("_to_%s" % target.filename_wo_ext)

      cmd = (['DemonsRegistration', '-f', source.path, '-m', target.path]
              + (["--fixed-mask", source.mask.path] if source.mask else [])
              + (["--moving-mask", target.mask.path] if target.mask else [])
              + (["-p", initial_source_transform.path] if initial_source_transform else [])
              + conf + ["-o", out_img.path, "-O", out_xfm.path])
      s = CmdStage(cmd=cmd,
                   inputs=(source, target)
                          + ((source.mask,) if source.mask else ())
                          + ((target.mask,) if target.mask else ()),
                   outputs = (out_xfm, out_img))

      xfm = GenericXfmHandler(source=source, target=target, xfm=out_xfm,
                              resampled=out_img)
      return Result(stages=Stages([s]), output=xfm)