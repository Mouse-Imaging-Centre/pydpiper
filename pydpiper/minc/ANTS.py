

# TODO use a class instead of a file(=module)?
import csv
import os
import warnings
from functools import reduce
from operator import mul
from typing import cast, List, Optional

from pyminc.volumes.factory import volumeFromFile

from pydpiper.core.stages import Result, CmdStage, Stages, identity_result
from pydpiper.core.util import NamedTuple
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.files import XfmAtom, MincAtom, IdMinc
from pydpiper.minc.nlin import NLIN
# TODO in order to remove circularity from the module import (which gives an exception at import time)
# TODO we need to move some stuff around ...
from pydpiper.minc.registration import (mincblur, mincresample, Interpolation,
                                        parse_many, parse_nullable, parse_bool, ParseError,
                                        all_equal, mincbigaverage, WithAvgImgs, MincAlgorithms, parse_n)
from pydpiper.itk.tools import Algorithms as ITKAlgorithms

SimilarityMetricConf = NamedTuple('SimilarityMetricConf',
                                  [("metric", str),
                                   ("weight", float),
                                   ("radius_or_bins", int),
                                   # TODO rename to blur_resolution?
                                   ("blur", float),  # TODO switch to factors instead?  allow vox/pc/mm ...?
                                   ("use_gradient_image", bool)])

default_similarity_metric_conf = SimilarityMetricConf(
    metric="CC",
    weight=1.0,
    radius_or_bins=3,
    blur=None,
    use_gradient_image=False)

ANTSConf = NamedTuple("ANTSConf",
                      [("file_resolution", float),
                       ("iterations", str),
                       ("transformation_model", str),  # TODO make an enumeration so, e.g., users don't type "Syn"
                       ("regularization", str),
                       ("use_mask", bool),
                       ("sim_metric_confs", List[SimilarityMetricConf])])

# we don't supply a resolution default here because it's preferable
# to take resolution from initial target instead
ANTS_default_conf = ANTSConf(
    iterations="100x100x100x150",
    transformation_model="'SyN[0.1]'",
    regularization="'Gauss[2,1]'",
    use_mask=True,
    file_resolution=None,
    sim_metric_confs=[default_similarity_metric_conf,
                      default_similarity_metric_conf.replace(use_gradient_image=True)])  # type: ANTSConf


class MultilevelANTSConf(object):
    def __init__(self, confs: List[ANTSConf]):
        self.confs = confs


def set_memory(st, source: MincAtom, conf: ANTSConf, mem_cfg):
    # see comments re: mincblur memory configuration
    voxels = reduce(mul, volumeFromFile(source.path).getSizes())
    mem_per_voxel = (mem_cfg.mem_per_voxel_coarse
                     if int(conf.iterations.split('x')[-1]) == 0
                     # yikes ... this parsing should be done earlier
                     else mem_cfg.mem_per_voxel_fine)
    st.setMem(mem_cfg.base_mem + voxels * mem_per_voxel)


# doesn't seem to be used anywhere
def get_default_multilevel_ANTS_conf(file_resolution: float) -> MultilevelANTSConf:
    """
    Create a multilevel ANTS configuration based on the provided file resolution.
    The iterations are:
    100x100x100x0
    100x100x100x20
    100x100x100x100
    """
    conf1 = ANTS_default_conf.replace(file_resolution=file_resolution,
                                      iterations="100x100x100x0")
    conf2 = ANTS_default_conf.replace(file_resolution=file_resolution,
                                      iterations="100x100x100x20")
    conf3 = ANTS_default_conf.replace(file_resolution=file_resolution,
                                      iterations="100x100x100x100")
    return MultilevelANTSConf([conf1, conf2, conf3])


ANTSMemCfg = NamedTuple("ANTSMemCfg",
                        [('base_mem', float),
                         ('mem_per_voxel_coarse', float),
                         ('mem_per_voxel_fine', bool)])

default_ANTS_mem_cfg = ANTSMemCfg(base_mem=0.177, mem_per_voxel_coarse=1.385e-7, mem_per_voxel_fine=2.1e-7)


class ANTS(NLIN):

  img_ext = ".mnc"
  xfm_ext = ".xfm"

  Conf = ANTSConf

  MultilevelConf = MultilevelANTSConf

  ToMinc = IdMinc

  Algorithms = MincAlgorithms

  # TODO I don't like all this weird class stuff -- complicated and seems unnecessary.
  # We should probably use generic NamedTuples instead, with the Python 3.6 syntax.
  @staticmethod
  def get_default_conf(resolution):
      return ANTS_default_conf.replace(file_resolution=resolution)

  # TODO parametrize over resolution as above?
  # TODO since this is not a property, things are not type-correct.
  # After declassifying things this shouldn't be a problem.
  @staticmethod
  def get_default_multilevel_conf(resolution):
      return get_default_multilevel_ANTS_conf(file_resolution=resolution)

  @staticmethod
  def accepts_initial_transform(): return False

  @staticmethod
  def register(source: MincAtom,
               target: MincAtom,
               conf: ANTSConf,
               initial_source_transform: Optional[XfmAtom] = None,
               transform_name_wo_ext: str = None,
               generation: int = None,
               resample_source: bool = False,
               #resample_name_wo_ext: Optional[str] = None,
               resample_subdir: str = "resampled") -> Result[XfmHandler]:
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

    if initial_source_transform is not None:
        raise ValueError("ANTs doesn't accept an initial transform")

    # if we resample the source, and place it in the "tmp" directory, we should do
    # the same with the transformation that is created:
    trans_output_dir = "transforms"
    if resample_source and resample_subdir == "tmp":
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
        if sim_metric_conf.use_gradient_image:
            if sim_metric_conf.blur is not None:
                gradient_blur_resolution = sim_metric_conf.blur
            elif conf.file_resolution is not None:
                gradient_blur_resolution = conf.file_resolution
            else:
                gradient_blur_resolution = None
                raise ValueError("A similarity metric in the ANTS configuration "
                                 "wants to use the gradients, but I know neither the file resolution nor "
                                 "an intended nonnegative blur fwhm.")
            if gradient_blur_resolution <= 0:
                warnings.warn("Not blurring the gradients as this was explicitly disabled")
            src = s.defer(mincblur(source, fwhm=gradient_blur_resolution)).gradient
            dest = s.defer(mincblur(target, fwhm=gradient_blur_resolution)).gradient
        else:
            # these are not gradient image terms; only blur if explicitly specified:
            if sim_metric_conf.blur is not None and sim_metric_conf.blur > 0:
                src  = s.defer(mincblur(source, fwhm=sim_metric_conf.blur)).img
                dest = s.defer(mincblur(source, fwhm=sim_metric_conf.blur)).img
            else:
                src  = source
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
        category = f"ANTS-{conf.iterations}",
        cmd=['ANTS', '3',
             '--number-of-affine-iterations', '0']
            + similarity_cmds
            + ['-t', conf.transformation_model,
               '-r', conf.regularization,
               '-i', conf.iterations,
               '-o', out_xfm.path]
            + (['-x', source.mask.path] if conf.use_mask and source.mask else []))

    # see comments re: mincblur memory configuration
    stage.when_runnable_hooks.append(lambda st: set_memory(st, source=source, conf=conf,
                                                           mem_cfg=default_ANTS_mem_cfg))

    s.add(stage)
    resampled = (s.defer(mincresample(img=source, xfm=out_xfm, like=target,
                                      interpolation=Interpolation.sinc,
                                      #new_name_wo_ext=resample_name_wo_ext,
                                      subdir=resample_subdir))
                 if resample_source else None)  # type: Optional[MincAtom]
    return Result(stages=s,
                  output=XfmHandler(source=source,
                                    target=target,
                                    xfm=out_xfm,
                                    resampled=resampled))


  # @classmethod
  # def build_model(cls,
  #                 imgs: List[MincAtom],
  #                 initial_target: MincAtom,
  #                 conf: MultilevelANTSConf,
  #                 nlin_dir: str,
  #                 nlin_prefix : str = "",
  #                 mincaverage = mincbigaverage) -> Result[WithAvgImgs[List[XfmHandler]]]:
  #   """
  #   This functions runs a hierarchical ANTS registration on the input
  #   images (imgs) creating an unbiased average.
  #   The ANTS configuration `confs` that is passed in should be
  #   a list of configurations for each of the levels/generations.
  #   After each round of registrations, an average is created out of the
  #   resampled input files, which is then used as the target for the next
  #   round of registrations.
  #   """
  #   if len(conf.confs) == 0:
  #       raise ValueError("No configurations supplied ...")
  #   s = Stages()
  #   avg = initial_target
  #   avg_imgs = []  # type: List[MincAtom]
  #   # changed start=1 to start=0 (and i -> i+1 in average creation) to match old code:
  #   for i, conf_inst in enumerate(conf.confs, start=0):
  #       # in the following command we resample the output of the ANTS command. This is because
  #       # we create an average during each iteration which is used as the target for the next iteration.
  #       # However, we should not save all resampled files in the resampled/ directory (default for
  #       # the ANTS() call). Do this only for the last iteration:
  #       resampled_subdir = "resampled" if i == len(conf.confs) else "tmp"
  #       xfms = [s.defer(cls.source_to_target(source=img, target=avg, conf=conf_inst, generation=i,
  #                                            resample_source=True, resample_subdir=resampled_subdir))
  #               for img in imgs]
  #       #  TODO make resampled name 'final-nlin' ?? need another option to ANTS for that, I guess ...
  #       # if no nlin_prefix is provided, we should remove the leading dash
  #       avg = s.defer(mincaverage([xfm.resampled for xfm in xfms],
  #                                 name_wo_ext='%s-nlin-%d' % (nlin_prefix, i+1) if nlin_prefix != "" else 'nlin-%d' % (i+1),
  #                                 output_dir=nlin_dir))
  #       avg_imgs.append(avg)
  #   return Result(stages=s, output=WithAvgImgs(output=xfms, avg_img=avg, avg_imgs=avg_imgs))

  @staticmethod
  def hierarchical_to_single(c): return c.confs

  @classmethod
  def parse_protocol_file(cls, filename, resolution):
      confs = cls.parse_multilevel_protocol_file(config_file=filename,
                                                 resolution=resolution).confs
      if len(confs) > 1:
          print("warning, too many confs")  # TODO get correct warning

      return confs[0]


  #@classmethod
  @classmethod
  def parse_multilevel_protocol_file(cls,
                                     config_file,
                                     resolution,
                                     ANTS_conf=ANTS_default_conf) -> MultilevelANTSConf:
    """
    config_file -- path to file on the system that contains the ANTS configuration

    Use the resulting list to `.replace` the default values.
    """

    # parsers to use for each row of the protocol file
    parsers = {"blur"               : parse_many(float),
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
    names = {"blur" : "blur",
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
    #if "blur" in d:
        #warnings.warn("You've specified your own blur level")
        # TODO should be a warnings/logger.warning, not a print
        # TODO: why is this not being used anymore? It allows you to specify what you want
        # TODO: to do with the similarity metrics. Not sure whether we should have hard coded
        # TODO: defaults for this?
    if "memory_required" in d:
        print("Warning: don't currently use the memory ...")  # doesn't have to be same length -> can crash code below
        del d["memory_required"]

    vs = list(d.values())
    l = len(vs[0])

    # convert a mapping of options to _single_ values to a single-generation ANTS configuration object:
    def convert_single_gen(single_gen_params, file_resolution) -> ANTSConf:  # TODO name this better ...
        # TODO check for/catch IndexError ... a bit hard to use zip since some params may not be defined ...
        sim_metric_names = {"blur", "use_gradient_image", "metric", "weight", "radius_or_bins"}
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

        return ANTS_default_conf.replace(sim_metric_confs=sim_metric_confs,
                                         file_resolution=file_resolution,
                                         **other_attrs)

    full_configuration = MultilevelANTSConf([convert_single_gen({key : vs[j] for key, vs in d.items()},
                                                                resolution) for j in range(l)])

    return full_configuration

class ANTS_ITK(ANTS):
    Algorithms = ITKAlgorithms