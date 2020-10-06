
import os
import warnings
from typing import (List, Type, Sequence, Optional)  #, NamedTuple, Tuple
import random

from pydpiper.core.stages import (Result, Stages)  #, CmdStage, identity_result
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.registration import (WithAvgImgs, mincbigaverage, #Interpolation,
                                        invert_xfmhandler, #minc_displacement, mincmath,
                                        xfmconcat, param2xfm)
from pydpiper.minc.files import MincAtom, XfmAtom, ImgAtom, IdMinc
from pydpiper.minc.nlin import NLIN, NLIN_BUILD_MODEL   #, Algorithms

gen = random.Random(42)

# TODO expand parameter list to be similar to ANTS_NLIN_build_model, possibly add resolution parameter?
# use/pass generation parameter for naming ?!
def build_model(reg_module : Type[NLIN]) -> Type[NLIN_BUILD_MODEL]:
    def f(imgs: List[MincAtom],
          initial_target: MincAtom,
          conf: reg_module.MultilevelConf,
          use_robust_averaging: bool,
          nlin_dir: str,
          nlin_prefix: str,
          #algorithms : Type[Algorithms]
          ) -> Result[WithAvgImgs[List[XfmHandler]]]:
        confs = reg_module.hierarchical_to_single(conf) if conf is not None else None
        if confs is None or len(confs) == 0:
            raise ValueError("No configurations supplied ...")
        s = Stages()

        avg = initial_target
        avg_imgs = []
        xfms = [None] * len(imgs)
        for i, conf in enumerate(confs, start=1):
            xfms = s.defer_all([reg_module.register(source=img,
                                                # in the case the registration algorithm doesn't accept
                                                # an initial transform,
                                                # we could use the resampled output of the previous
                                                # step for a more efficient registration process,
                                                # although this would require more careful bookkeeping
                                                # of transforms and incur additional resampling error
                                                target=avg,
                                                conf=conf,
                                                initial_source_transform=xfm.xfm
                                                              if reg_module.accepts_initial_transform()
                                                                   and (xfm is not None)
                                                              else None,
                                                ##generation=i,
                                                # TODO reduce unneeded resamplings if accepts_initial_transform?
                                                resample_source=True)
                    for img, xfm in zip(imgs, xfms)])
            avg = s.defer(reg_module.Algorithms.average([xfm.resampled for xfm in xfms],
                                                        robust=use_robust_averaging,
                                                        name_wo_ext='%s-nlin-%d' % (nlin_prefix, i),
                                                        output_dir=nlin_dir))
            avg_imgs.append(avg)
        return Result(stages=s, output=WithAvgImgs(output=xfms, avg_img=avg, avg_imgs=avg_imgs))
    return mk_build_model_class(nlin=reg_module,
                                build_model=f,
                                build_model_conf_type=reg_module.MultilevelConf,
                                get_default_build_model_conf=reg_module.get_default_multilevel_conf,
                                parse_build_model_protocol=reg_module.parse_multilevel_protocol_file)


def nonlinear_midpoint_xfm(nlin_algorithm : Type[NLIN],
                           img_A: MincAtom,
                           img_B: MincAtom,
                           conf, #: nlin_algorithm.Conf,
                           out_name_wo_ext: str,
                           out_dir: str,
                           mincaverage = mincbigaverage) -> Result[MincAtom]:
    """
    :param img_A:
    :param img_B:
    :return: the midway point between img_A and img_B --> img_AB
    """
    s = Stages()

    # # invariant: all the `xfms_A` have the same `resampled` field, and same for `xfms_B`:

    # start with an antRegistration call between the two files
    #xfm_handlers_antsReg = s.defer(antsRegistration(source=img_A,
    #                                                target=img_B,
    #                                                subdir='tmp'))
    A_to_B = s.defer(nlin_algorithm.register(source=img_A,
                                             # all the xfms have the same `resampled` field
                                             target=img_B,
                                             conf=conf,
                                             transform_name_wo_ext="%s_A_to_B" % out_name_wo_ext,
                                             ####resample_name_wo_ext="%s_A_to_B" % out_name_wo_ext,
                                             resample_subdir=out_dir)) # TODO 'tmp' ?

    # N.B.: this uses `invert_xfmhandler` magic in either extracting the inverse from A_to_B
    # if it's present or inverting the transform if not ... slightly uncomfortable with this for some reason.
    B_to_A = s.defer(invert_xfmhandler(A_to_B))

    algorithms = nlin_algorithm.Algorithms
    # generate halfway transformations
    transform_A_to_B_halfway = s.defer(algorithms.scale_transform(xfm=A_to_B,
                                                                  scale=0.5, newname_wo_ext="%s_AtoB_half" % out_name_wo_ext))
    transform_B_to_A_halfway = s.defer(algorithms.scale_transform(xfm=B_to_A,
                                                                  scale=0.5, newname_wo_ext="%s_BtoA_half" % out_name_wo_ext))

    # resample A and B halfway:
    A_halfway_to_B = s.defer(algorithms.resample(img=img_A,
                                                 xfm=transform_A_to_B_halfway,
                                                 like=img_B,
                                                 subdir='tmp',
                                                 new_name_wo_ext=out_name_wo_ext + "_AtoB"))
    B_halfway_to_A = s.defer(algorithms.resample(img=img_B,
                                                 xfm=transform_B_to_A_halfway,
                                                 like=img_A,
                                                 subdir='tmp',
                                                 new_name_wo_ext=out_name_wo_ext + "_BtoA"))

    # the output file (avg of both files resampled to the midway point)
    avg_mid_point = s.defer(algorithms.average(imgs=[A_halfway_to_B, B_halfway_to_A],
                                               output_dir=out_dir,
                                               name_wo_ext=out_name_wo_ext))

    return Result(stages=s, output=(XfmHandler(source=img_A, target=None,
                                               xfm=transform_A_to_B_halfway, resampled=A_halfway_to_B),
                                    XfmHandler(source=img_B, target=None,
                                               xfm=transform_B_to_A_halfway, resampled=B_halfway_to_A),
                                    avg_mid_point))


def tournament(reg_module : Type[NLIN]):
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
    """
    # TODO: add weighting to the transforms in the case that the number of inputs is not a power of 2
    def f(imgs: List[MincAtom],
          initial_target: MincAtom,
          conf: reg_module.Conf,
          nlin_dir: str,
          nlin_prefix: str,
          tournament_name_wo_ext: str = "tournament") -> Result[List[XfmHandler]]:
      s = Stages()
      Weight = int
      def h(xfms : List[XfmHandler], name_wo_ext : str) -> List[XfmHandler]:
          # TODO add weights to each return
          # TODO check len(...) == 0 case??
          if len(xfms) <= 1:
              return xfms
          else:
              first_half  = xfms[: len(xfms)//2]
              second_half = xfms[len(xfms)//2 :]
              first_half_result  = h(first_half,  name_wo_ext=name_wo_ext + "_L")
              second_half_result = h(second_half, name_wo_ext=name_wo_ext + "_R")
              A_halfway_to_B, B_halfway_to_A, avg_img = s.defer(
                  nonlinear_midpoint_xfm(
                      img_A = first_half_result[0].resampled,
                      img_B = second_half_result[0].resampled,
                      out_name_wo_ext=name_wo_ext,
                      nlin_algorithm=reg_module,
                      conf=conf,
                      out_dir=nlin_dir))
              xfms_to_midpoint = ([XfmHandler(source=xfm.source,
                                              target=avg_img,
                                              resampled=A_halfway_to_B.resampled,
                                              xfm=s.defer(xfmconcat([xfm.xfm, A_halfway_to_B.xfm],
                                                                    name="%s_%s" % (xfm.source.filename_wo_ext,
                                                                                    name_wo_ext))))
                                   for xfm in first_half_result]
                                  + [XfmHandler(source=xfm.source,
                                                target=avg_img,
                                                resampled=B_halfway_to_A.resampled,
                                                xfm=s.defer(xfmconcat([xfm.xfm, B_halfway_to_A.xfm],
                                                                      name="%s_%s" % (xfm.source.filename_wo_ext,
                                                                                      name_wo_ext))))
                                     for xfm in second_half_result])
              return xfms_to_midpoint
      identity_xfm = s.defer(param2xfm(out_xfm=XfmAtom(pipeline_sub_dir=imgs[0].pipeline_sub_dir,
                                                       output_sub_dir=imgs[0].output_sub_dir,
                                                       name=os.path.join(imgs[0].pipeline_sub_dir,
                                                                         imgs[0].output_sub_dir,
                                                                         "id.xfm"))))
      initial_xfms = [XfmHandler(source=img, target=img,
                                 resampled=img, xfm=identity_xfm) for img in imgs]
      xfms_to_avg = h(initial_xfms, tournament_name_wo_ext)
      avg_img = xfms_to_avg[0].target
      return Result(stages=s, output=WithAvgImgs(avg_img=avg_img, avg_imgs=[avg_img],
                                                 output=xfms_to_avg))
    return mk_build_model_class(nlin=reg_module,
                                build_model_conf_type=reg_module.Conf,
                                get_default_build_model_conf=reg_module.get_default_conf,
                                build_model=f,
                                parse_build_model_protocol=reg_module.parse_protocol_file)


def mk_build_model_class(nlin : Type[NLIN],
                         build_model_conf_type,
                         get_default_build_model_conf,  # resolution -> build_model_conf_type
                         build_model,  # TODO add type sig for this procedure!
                         parse_build_model_protocol) -> Type[NLIN_BUILD_MODEL]:
    class BUILD_MODEL_CLASS(nlin):
        BuildModelConf = build_model_conf_type
        @staticmethod
        def build_model(*args, **kwargs): return build_model(*args, **kwargs)

        @staticmethod
        def parse_build_model_protocol(*args, **kwargs) -> BuildModelConf:
            return parse_build_model_protocol(*args, **kwargs)

        @staticmethod
        def get_default_build_model_conf(*args, **kwargs) -> BuildModelConf:
            return get_default_build_model_conf(*args, **kwargs)
    return BUILD_MODEL_CLASS


def pairwise(nlin_module: NLIN, max_pairs: Optional[int] = None, max_images: Optional[int] = None):
  def f(imgs: List[ImgAtom],   # TODO: these types are quite imprecise!
        nlin_dir: str,
        conf: nlin_module.Conf,
        initial_target: ImgAtom,
        nlin_prefix: str
        #output_dir_for_avg: str = None,
        #output_name_wo_ext: str = None
        ):
    s = Stages()

    if len(imgs) < 2:
        # TODO add error checking/warnings for max_images and max_pairs as well ...
        raise ValueError("currently need at least two images")

    #if not output_name_wo_ext:
    #    output_name_wo_ext = "full_pairwise_nlin_%s" % nlin_module.__name__
    output_name_wo_ext = "%s_nlin_%s" % (nlin_prefix, nlin_module.__name__)

    final_avg = ImgAtom(name=os.path.join(nlin_dir, output_name_wo_ext + ".todo"),
                        pipeline_sub_dir=nlin_dir)
    final_avg.ext = nlin_module.img_ext  # FIXME

    def avg_nlin_xfm_from(src_img: MincAtom,
                          target_imgs: List[MincAtom]):
        # TODO: should there be another affine step here?  Two images registered in the best affine way
        # to the overall average might not be ideally affinely registered to each other ...
        xfmHs = [s.defer(nlin_module.register(source=src_img,   ## TODO: add source resampling stuff !!
                                              target=target_img,
                                              conf=conf,
                                              resample_subdir=nlin_dir))   ## TODO: is this subdir correct?
                 for target_img in target_imgs]
        xfm = XfmAtom(name=os.path.join(src_img.pipeline_sub_dir,
                                        src_img.output_sub_dir,
                                       "transforms",
                                       "%s_avg_nlin_%s.todo" %
                                         (src_img.filename_wo_ext,
                                          nlin_module.__name__)),
                      pipeline_sub_dir=src_img.pipeline_sub_dir,
                      output_sub_dir=src_img.output_sub_dir)
        xfm.ext = nlin_module.xfm_ext

        avg_xfm = s.defer(nlin_module.Algorithms.average_transforms(xfms=xfmHs,
                                                                    avg_xfm=xfm))

        res = s.defer(nlin_module.Algorithms.resample(img=src_img,
                                                      xfm=avg_xfm,
                                                      like=src_img))
        return XfmHandler(xfm=avg_xfm,
                          source=src_img,
                          target=final_avg,
                          resampled=res)

    if max_images is None or max_images >= len(imgs):
        model_imgs = imgs
    else:
        warnings.warn("nonlinear max_images is set; hopefully this is NOT generating your final consensus average")
        model_imgs = gen.sample(imgs, max_images)


    if max_pairs is None or max_pairs >= len(model_imgs):
        avg_xfms = [avg_nlin_xfm_from(#conf=conf, like=like,
                                               #output_atom=final_avg,
                                               src_img=img, target_imgs=model_imgs)
                     for img in model_imgs]
    else:
        warnings.warn("nonlinear max_pairs is set; hopefully this is NOT generating your consensus average!")
        avg_xfms = [avg_nlin_xfm_from(#conf=conf, like=like,
                                               #output_atom=final_avg,
                                               src_img=img, target_imgs=gen.sample(model_imgs, max_pairs))
                     for img in model_imgs]

    # avg_xfmHs = [avg_nlin_xfm_from(src_img=img, target_imgs=imgs) for img in imgs]

    final_avg = s.defer(nlin_module.Algorithms.average(imgs=[xfm.resampled for xfm in avg_xfms],
                                                       avg_file=final_avg))

    return Result(stages=s, output=WithAvgImgs(avg_imgs=[final_avg], avg_img=final_avg, output=avg_xfms))
  return mk_build_model_class(build_model_conf_type=nlin_module.Conf,
                              build_model=f,
                              nlin=nlin_module,
                              get_default_build_model_conf=nlin_module.get_default_conf,
                              parse_build_model_protocol=nlin_module.parse_protocol_file)


# TODO add option to cut off tournament after a couple levels and do a build model??
# you'd think this is sort of unnecessary but at the moment command-line users have no way to
# specify some combination of nlin modules, so:
def tournament_and_build_model(nlin_module : Type[NLIN]):
    def f(imgs: List[MincAtom],
          nlin_dir: str,
          conf: nlin_module.MultilevelConf,
          initial_target: MincAtom,
          nlin_prefix: str,
          #output_dir_for_avg: str = None,
          #output_name_wo_ext: str = None
          ):
        s = Stages()

        tournament_result = s.defer(tournament(nlin_module).build_model(
            imgs=imgs, nlin_dir=nlin_dir, conf=nlin_module.hierarchical_to_single(conf)[-1] if conf else None,
            initial_target=initial_target, nlin_prefix=nlin_prefix
            #, output_name_wo_ext=output_name_wo_ext  #, algorithms=nlin_module.algorithms
        ))

        build_model_result = s.defer(build_model(nlin_module).build_model(
            imgs=imgs, nlin_dir=nlin_dir, conf=conf, initial_target=tournament_result.avg_img,
            nlin_prefix=nlin_prefix
            #, output_name_wo_ext=output_name_wo_ext  #, algorithms=algorithms
        ))

        return Result(stages=s, output=build_model_result)
    # TODO we'll just use the last conf for the tournament; could modify to take 2 protocols instead
    return mk_build_model_class(build_model_conf_type=nlin_module.MultilevelConf,
                                build_model=f,
                                nlin=nlin_module,
                                get_default_build_model_conf=nlin_module.get_default_multilevel_conf,
                                parse_build_model_protocol=nlin_module.parse_multilevel_protocol_file)


def pairwise_and_build_model(nlin_module : Type[NLIN]):
    def f(imgs: List[MincAtom],
          nlin_dir: str,
          conf: nlin_module.MultilevelConf,
          initial_target: MincAtom,
          nlin_prefix: str,
          #output_dir_for_avg: str = None,
          #output_name_wo_ext: str = None
          ):
        s = Stages()

        pairwise_result = s.defer(pairwise(nlin_module, max_images=25, max_pairs=None).build_model(
            imgs=imgs, nlin_dir=nlin_dir, conf=nlin_module.hierarchical_to_single(conf)[-1] if conf else None,
            initial_target=initial_target, nlin_prefix=nlin_prefix
            #, output_name_wo_ext=output_name_wo_ext  #, algorithms=nlin_module.algorithms
        ))

        build_model_result = s.defer(build_model(nlin_module).build_model(
            imgs=imgs, nlin_dir=nlin_dir, conf=conf, initial_target=pairwise_result.avg_img,
            nlin_prefix=nlin_prefix
            #, output_name_wo_ext=output_name_wo_ext  #, algorithms=algorithms
        ))

        return Result(stages=s, output=build_model_result)
    # TODO we'll just use the last conf for the pairwise registration;
    #   could modify to take 2 protocols instead
    return mk_build_model_class(build_model_conf_type=nlin_module.MultilevelConf,
                                build_model=f,
                                nlin=nlin_module,
                                get_default_build_model_conf=nlin_module.get_default_multilevel_conf,
                                parse_build_model_protocol=nlin_module.parse_multilevel_protocol_file)


# is there any point to mincifying an NLIN module by itself ??  unsure ...
# def wrap_build_model(build_model : Type[NLIN_BUILD_MODEL],
#                      wrap_input_image,
#                      #wrap_input_xfm,
#                      wrap_output_image,
#                      wrap_output_xfm) -> Type[NLIN_BUILD_MODEL]:
#     class C(build_model):
#         @classmethod
#         def register(cls, *args, **kwargs):
#             raise NotImplementedError(
#                 "I didn't do this since going back and forth to .mnc seems dangerous")
#
#         @staticmethod
#         def build_model(imgs,
#                         conf,
#                         nlin_dir,
#                         nlin_prefix,
#                         initial_target,
#                         mincaverage,
#                         output_name_wo_ext = None):
#             s = Stages()
#             #imgs = tuple(s.defer(convert(img, out_ext=".nii.gz") for img in imgs))
#             imgs = tuple(s.defer(wrap_input_image(img) for img in imgs))
#             result = build_model.build_model(imgs=imgs, conf=conf,
#                                              nlin_dir=nlin_dir, nlin_prefix=nlin_prefix,
#                                              initial_target=initial_target, mincaverage=mincaverage,
#                                              output_name_wo_ext=output_name_wo_ext)
#
#             def wrap_output_xfmh(xfmh):
#                 return XfmHandler(source=s.defer(wrap_output_image(xfmh.source)) if xfmh.source else None,
#                                   target=s.defer(wrap_output_image(xfmh.target)) if xfmh.target else None,
#                                   resampled=s.defer(wrap_output_image(xfmh.resampled)) if xfmh.resampled else None,
#                                   xfm=s.defer(wrap_output_xfm(xfmh.xfm)),
#                                   inverse_xfm=wrap_output_xfmh(xfmh.inverse_xfm))
#
#             return Result(stages=s, output=WithAvgImgs(avg_imgs=[s.defer(wrap_output_image(img))
#                                                                  for img in result.avg_imgs],
#                                                        avg_img=s.defer(wrap_output_image(result.avg_img)),
#                                                        output=[s.defer(wrap_output_xfmh(x))
#                                                                for x in result.xfms]))
#     return C

def mincify_build_model(base_build_model : Type[NLIN_BUILD_MODEL]) -> Type[NLIN_BUILD_MODEL]:
    """Takes a model building component that inputs/outputs on a certain image format
    and returns a new component that works on MINC files.  This isn't magic: the model building component
    includes a ToMinc component."""
    class C(base_build_model):
        @classmethod
        def register(cls, *args, **kwargs):
            raise NotImplementedError(
                "I didn't do this (yet) since going back and forth to .mnc seems annoying/dangerous")

        ToMinc = IdMinc

        @staticmethod
        def build_model(imgs,
                        conf,
                        nlin_dir,
                        nlin_prefix,
                        use_robust_averaging,
                        initial_target,
                        output_name_wo_ext = None):
            s = Stages()
            mincify = base_build_model.ToMinc
            imgs = tuple(s.defer(mincify.from_mnc(img)) for img in imgs)
            result = s.defer(base_build_model.build_model(imgs=imgs, conf=conf,
                                                  nlin_dir=nlin_dir, nlin_prefix=nlin_prefix,
                                                  use_robust_averaging=use_robust_averaging,
                                                  initial_target=s.defer(mincify.from_mnc(initial_target))
                                                  #output_name_wo_ext=output_name_wo_ext
                                                  ))

            def wrap_output_xfmh(xfmh):
                return XfmHandler(source=s.defer(mincify.to_mnc(xfmh.source)) if xfmh.source else None,
                                  target=s.defer(mincify.to_mnc(xfmh.target)) if xfmh.target else None,
                                  resampled=s.defer(mincify.to_mnc(xfmh.resampled)) if xfmh.has_resampled() else None,
                                  xfm=s.defer(mincify.to_mni_xfm(xfmh.xfm)),
                                  inverse=wrap_output_xfmh(xfmh.inverse) if xfmh.has_inverse() else None)

            return Result(stages=s, output=WithAvgImgs(avg_imgs=[s.defer(mincify.to_mnc(img))
                                                                 for img in result.avg_imgs],
                                                       avg_img=s.defer(mincify.to_mnc(result.avg_img)),
                                                       output=[wrap_output_xfmh(x)
                                                               for x in result.output]))
    return C()


def get_model_building_procedure(strategy : str, reg_module : Type[NLIN]) -> Type[NLIN_BUILD_MODEL]:
    d = {
          'build_model' : build_model,
          'tournament'  : tournament,
          'pairwise'    : pairwise,
          'tournament_and_build_model' : tournament_and_build_model,
          'pairwise_and_build_model'   : pairwise_and_build_model
        }
    try:
        c = mincify_build_model(d[strategy](reg_module))
    except KeyError:
        raise ValueError("unknown strategy %s; choices are %s" % (strategy, d.keys()))
    else:
        return c


