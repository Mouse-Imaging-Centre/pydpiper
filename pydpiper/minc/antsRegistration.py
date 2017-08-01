
import os
import warnings
from functools import reduce
from operator import mul
from typing import Optional, Tuple, Sequence

from pyminc.volumes.factory import volumeFromFile

from pydpiper.minc.ANTS import ANTSMemCfg
from pydpiper.core.util import AutoEnum, NamedTuple, flatten
from pydpiper.minc.nlin import NLIN
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.registration import mincresample, Interpolation, mincblur, MincAlgorithms
from pydpiper.core.stages import Stages, CmdStage, Result, identity_result
from pydpiper.minc.files import MincAtom, XfmAtom, IdMinc

ConvergenceCriteria = NamedTuple("ConvergenceCriteria",
                                 [("convergence_threshold", float),
                                  ("convergence_window_size", Optional[int])])

ConvergenceConf = NamedTuple("ConvergenceConf",
                             [('iterations', Tuple[int]),
                              ('convergence_criteria', Optional[ConvergenceCriteria])])


def render_convergence_conf(cc : ConvergenceConf):
    iter_str = 'x'.join(str(i) for i in cc.iterations)
    if cc.convergence_criteria is None:
        return iter_str
    else:
        threshold = cc.convergence_criteria.convergence_threshold
        window_size = cc.convergence_criteria.convergence_window_size
        if window_size is None:
            return "[%s,%s]" % (iter_str, threshold)
        else:
            return "[%s,%s,%s]" % (iter_str, threshold, window_size)

#class Interpolation(object):
#    pass

#class Affine(Interpolation):
#    pass


#Linear = Interpolation("Linear", [])
#NearestNeighbor = Interpolation("NearestNeighbour", [])
#MultiLabel = Interpolation("MultiLabel", [])
# FIXME this is not really general and doesn't handle most of the metrics; see some work below
Metric = NamedTuple('SimilarityMetricConf',
                    [("metric", str),
                     ("weight", float),
                     ("radius_or_bins", int),
                     ("use_gradient_image", bool)])

default_metric = Metric(metric="CC", weight=1, radius_or_bins=6, use_gradient_image=False)
default_metrics = (default_metric, default_metric.replace(use_gradient_image=True))


#class Metric(object):
#    pass

#CC = NamedTuple("CC",
#                [("fixedImage", )
#                 ("movingImage", )
#                 ("metricWeight", float)
#                 ("radius", )
#                 ("sampling", Optional[Sampling])])

#def render_metric(metric : Metric):
#    return ""

   # for sim_metric_conf in conf.sim_metric_confs:
   #      if conf.file_resolution is not None and sim_metric_conf.use_gradient_image:
   #          src = s.defer(mincblur(source, fwhm=conf.file_resolution)).gradient
   #          dest = s.defer(mincblur(target, fwhm=conf.file_resolution)).gradient
   #      elif conf.file_resolution is None and sim_metric_conf.use_gradient_image:
   #          # the file resolution is not set, however we want to use the gradients
   #          # for this similarity metric...

   #      else:
   #          src = source
   #          dest = target
   #      similarity_inputs.add(src)
   #      similarity_inputs.add(dest)
   #      inner = ','.join([src.path, dest.path,
   #                        str(sim_metric_conf.weight), str(sim_metric_conf.radius_or_bins)])
   #      subcmd = "'" + "".join([sim_metric_conf.metric, '[', inner, ']']) + "'"
   #      similarity_cmds.extend(["-m", subcmd])


class SamplingStrategy(AutoEnum):
    none = regular = random = ()

    def render(self): return self.name.title()

Sampling = NamedTuple("Sampling",
                      [('sampling_strategy', SamplingStrategy),
                       ('sampling_fraction', Optional[float])])


class Transform(object): pass
def mk_gradient_step_transform_class(name : str):
    class T(Transform):
        def __init__(self, gradientStep):
            self.gradientStep = gradientStep
            self.name = name
        def render(self):
            return "%s[%s]" % (self.name, self.gradientStep)
    return T

Rigid, Affine, CompositeAffine, Similarity, Translation = [mk_gradient_step_transform_class(s)
                                                           for s in ("Rigid", "Affine", "CompositeAffine",
                                                                     "Similarity", "Translation")]

#class SyN(Transform):
#    def __init__(self, *,
#                 gradientStep,
#                 updateFieldVarianceInVoxelSpace,
#                 totalFieldVarianceInVoxelSpace):
#        self.args = **kwargs #?!

# FIXME inheriting from NamedTuple doesn't seem to work ...
#class TransformationModel(NamedTuple):
#    pass

#Rigid = TransformationModel("Rigid", [('gradientStep', float)])
#Affine = TransformationModel("Affine", [('gradientStep', float)])


SimilarityMetricConf = NamedTuple('SimilarityMetricConf',
                                  [("metric", str),
                                   ("weight", float),
                                   ("radius_or_bins", int),
                                   ("use_gradient_image", bool)])


default_ANTSRegistration_mem_cfg = ANTSMemCfg(base_mem=0.177,
                                              mem_per_voxel_coarse=1.385e-7,
                                              mem_per_voxel_fine=2.1e-7)

# TODO ANTSRegistration can actually conduct a series of different registrations with different
# parameters; i.e., transformation_model (--transform) and convergence (--convergence) should be lists,
# and most other things should (optionally?) be lists of lists
# TODO this definition of conf allows the ...x...x... sequences to be of unequal lengths,
# which will surely cause either a crash or unexpected behaviour... best to 'transpose'.
ANTSRegistrationConf = NamedTuple("ANTSRegistrationConf",
    [("file_resolution", float),
     # file_resolution is magic added by us; all other fields are from antsRegistration
     #  (except use_masks, which is sort of a weird way to get the input file ...)
     # we drop file-specific fields (--output, etc.) but pass these to antsRegistration separately
     # TODO save/restore-state, write-composite-transform, ...
     ("dimensionality", Optional[int]),
     ("convergence", ConvergenceConf),
     #("interpolation", Optional[Interpolation]),
     ("transformation_model", str),  # TODO make an enumeration so, e.g., users don't type "Syn"
     ("use_masks", bool),
     ("use_histogram_matching", bool),
     ("smoothing_sigmas", Sequence[int]),
     ("shrink_factors", Sequence[int]),
     ("metrics", Sequence[Metric])
     #("sim_metric_confs", List[SimilarityMetricConf])
     ])

ANTSRegistrationDefaultConf = ANTSRegistrationConf(
    file_resolution = None,
    dimensionality = 3,
    convergence = ConvergenceConf(iterations=(100, 100, 100, 100, 100, 100),
                                  convergence_criteria=None),
    #interpolation = None,
    transformation_model = "SyN[0.5,3,0]",
    use_masks = True,
    use_histogram_matching = True,
    smoothing_sigmas = (8, 4, 3, 2, 1, 0),
    shrink_factors = (16, 8, 6, 4, 2, 1),
    metrics = default_metrics
)


def antsRegistration(source: MincAtom,
                     target: MincAtom,
                     conf: ANTSRegistrationConf,
                     initial_source_transform: Optional[XfmAtom] = None,
                     initial_target_transform: Optional[XfmAtom] = None,
                     # TODO create source_to_target_transform_name_wo_ext, target_to_source_...
                     transform_name_wo_ext: Optional[str] = None,
                     generation: Optional[int] = None,
                     subdir: Optional[str] = None,
                     resample_source: bool = False,
                     resample_target: bool = False,
                     resample_subdir: str  = 'tmp'):
    """
    :param source: fixedImage
    :param target: movingImage
    :param conf:
    :param initial_source_transform: for --initial-fixed-transform
    :param initial_target_transform: for --initial-moving-transform
    :param transform_name_wo_ext: name of the target_to_source transform will be based on this
    :param generation:
    :param subdir:
    :param resample_source:
    :return:
    """
    s = Stages()

    source_subdir = subdir if subdir is not None else 'transforms'

    # Deal with the transformations first. This function will return two XfmHandlers. One
    # from source to target, and one in the other direction.
    if transform_name_wo_ext:
        xfm_source_to_target = XfmAtom(name=os.path.join(source.pipeline_sub_dir,
                                                         source.output_sub_dir,
                                                         source_subdir,
                                                         "%s.xfm" % transform_name_wo_ext),
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

    def optional(x, f, default=[]):
        return f(x) if x is not None else []
    # TODO: use a proper configuration to set the parameters
    # TODO: add a second metric for the gradients (and get gradient files)

    if any(m.use_gradient_image for m in conf.metrics):
        if conf.file_resolution is None:
            raise ValueError("A similarity metric in the ANTS configuration "
                             "wants to use the gradients, but the file resolution for the "
                             "configuration has not been set.")
        blurred_source, blurred_target = [s.defer(mincblur(img, fwhm=conf.file_resolution)).gradient
                                          for img in (source, target)]
    else:
        blurred_source = blurred_target = None

    def render_metric(m : Metric):
        if m.use_gradient_image:
            if conf.file_resolution is None:
                raise ValueError("A similarity metric in the ANTS configuration "
                                 "wants to use the gradients, but the file resolution for the "
                                 "configuration has not been set.")
            fixed = blurred_source
            moving = blurred_target
        else:
            fixed = source
            moving = target
        return "'%s[%s,%s,%s,%s]'" % (m.metric, fixed.path, moving.path, m.weight, m.radius_or_bins)

    cmd = CmdStage(
        inputs=tuple(img for img in
                     (source, target,
                      source.mask, target.mask,
                      blurred_source, blurred_target,
                      initial_source_transform, initial_target_transform)
                     if img is not None),
        outputs=(xfm_source_to_target, xfm_target_to_source),
        cmd=['antsRegistration']
            + optional(conf.dimensionality, lambda d: ['--dimensionality', "%d" % d])
            + ['--convergence', render_convergence_conf(conf.convergence)]
            + ['--verbose']
            + ['--minc']
            + ['--collapse-output-transforms', '1']
            + ['--write-composite-transform']
            + ['--winsorize-image-intensities', '[0.01,0.99]']
            + optional(conf.use_histogram_matching, lambda _: ['--use-histogram-matching', '1'])
            + ['--float', '0']
            + ['--output', '[' + xfm_source_to_target.dir + '/' + xfm_source_to_target.filename_wo_ext + ']']
            + ['--transform', conf.transformation_model]
            + optional(initial_source_transform, lambda xfm: ['--initial-fixed-transform', xfm.path])
            + optional(initial_target_transform, lambda xfm: ['--initial-moving-transform', xfm.path])
            + flatten(*[['--metric', render_metric(m)] for m in conf.metrics])
            + (['--masks', '[' + source.mask.path + ',' +
                target.mask.path + ']'] if source.mask.path and target.mask.path and conf.use_masks else [])
            + ['--shrink-factors', 'x'.join(str(s) for s in conf.shrink_factors)]
            + ['--smoothing-sigmas', 'x'.join(str(s) for s in conf.smoothing_sigmas)]
    )

    # shamelessly stolen from ANTS, probably inaccurate
    # see comments re: mincblur memory configuration
    def set_memory(st, mem_cfg):
        # see comments re: mincblur memory configuration
        voxels = reduce(mul, volumeFromFile(source.path).getSizes())
        mem_per_voxel = (mem_cfg.mem_per_voxel_coarse
                         if 0 in conf.convergence.iterations[-1:]  #-2?
                         # yikes ... this parsing should be done earlier
                         else mem_cfg.mem_per_voxel_fine)
        st.setMem(mem_cfg.base_mem + voxels * mem_per_voxel)

    cmd.when_runnable_hooks.append(lambda st: set_memory(st, mem_cfg=default_ANTSRegistration_mem_cfg))

    s.add(cmd)

    # create file names for the two output files. It's better to use our standard
    # mincresample command for this, because that also deals with any associated
    # masks, whereas antsRegistration would only resample the input files.
    resampled_source = (s.defer(mincresample(img=source,
                                             xfm=xfm_source_to_target,
                                             like=target,
                                             interpolation=Interpolation.sinc,
                                             subdir=resample_subdir))
                        if resample_source else None)
    resampled_target = (s.defer(mincresample(img=target,
                                             xfm=xfm_target_to_source,
                                             like=source,
                                             interpolation=Interpolation.sinc,
                                             subdir=resample_subdir))
                        if resample_target else None)

    # return an XfmHandler for both the forward and the inverse transformations
    return Result(stages=s,
                  output=XfmHandler(source=source,
                                    target=target,
                                    xfm=xfm_source_to_target,
                                    resampled=resampled_source,
                                    inverse=XfmHandler(source=target,
                                                       target=source,
                                                       xfm=xfm_target_to_source,
                                                       resampled=resampled_target)))
                  #output=(XfmHandler(source=source,
                  #                   target=target,
                  #                   xfm=xfm_source_to_target,
                  #                   resampled=resampled_source),
                  #        XfmHandler(source=target,
                  #                   target=source,
                  #                   xfm=xfm_target_to_source,
                  #                   resampled=resampled_target)))

class ANTSRegistration(NLIN):

    img_ext = ".mnc"
    xfm_ext = ".xfm"

    Conf = ANTSRegistrationConf

    MultilevelConf = Sequence[ANTSRegistrationConf]

    ToMinc = IdMinc

    Algorithms = MincAlgorithms

    @staticmethod
    def get_default_conf(resolution):
        return ANTSRegistrationDefaultConf.replace(
                   file_resolution=resolution,
                   convergence=ANTSRegistrationDefaultConf.convergence.replace(
                       iterations = (100, 100, 100, 100, 100, 150)
                   ))

    @staticmethod
    def get_default_multilevel_conf(resolution):
        return tuple(ANTSRegistrationDefaultConf.replace(
                         file_resolution = resolution,
                         convergence =
                             ANTSRegistrationDefaultConf.convergence.replace(
                                 iterations = iters,
                             ))
                     for iters in ((100, 100, 100, 100, 0,   0),
                                   (100, 100, 100, 100, 100, 0),
                                   (100, 100, 100, 100, 100, 20),
                                   (100, 100, 100, 100, 100, 150)))

    @staticmethod
    def hierarchical_to_single(c): return c

    @staticmethod
    def accepts_initial_transform(): return True  # TODO implement this since antsRegistration supports it!

    @staticmethod
    def register(*args, **kwargs): return antsRegistration(*args, **kwargs)

    @classmethod
    def parse_protocol_file(cls, filename : str, resolution : float):
        warnings.warn("not implemented")
        return None

    @classmethod
    def parse_multilevel_protocol_file(cls, filename : str, resolution : float):
        warnings.warn("not implemented")
        return None
