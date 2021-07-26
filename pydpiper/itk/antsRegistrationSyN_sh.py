import os
from typing import Optional

from pydpiper.core.files import ImgAtom, imgToXfm, xfmToImg
from pydpiper.core.stages import Stages, CmdStage, Result
from pydpiper.minc.containers import XfmHandler
from pydpiper.minc.files import XfmAtom

from pydpiper.itk.tools import Algorithms as ITKAlgorithms
from pydpiper.minc.nlin import NLIN


class AntsRegistrationSyNQuick_affine(NLIN):

    img_ext = ".nii.gz"
    xfm_ext = ".mat"

    reg_type = "a"

    Algorithms = ITKAlgorithms

    Conf = type(())
    MultilevelConf = type(())

    @staticmethod
    def hierarchical_to_single(m):
        return ()

    # TODO I don't like all this weird class stuff -- complicated and seems unnecessary.
    # We should probably use generic NamedTuples instead, with the Python 3.6 syntax.
    @staticmethod
    def get_default_conf(resolution):
        return ()

    # TODO parametrize over resolution as above?
    # TODO since this is not a property, things are not type-correct.
    # After declassifying things this shouldn't be a problem.
    @staticmethod
    def get_default_multilevel_conf(resolution):
        return ()

    @classmethod
    def parse_multilevel_protocol_file(cls, filename, resolution):
        return ()

    @classmethod
    def parse_protocol_file(cls, filename, resolution):
        return ()

    @staticmethod
    def accepts_initial_transform(): return True

    @classmethod
    def register(cls,
                 moving: ImgAtom,
                 fixed: ImgAtom,
                 conf: (),
                 initial_moving_transform: Optional[XfmAtom] = None,
                 transform_name_wo_ext: str = None,
                 generation: int = None,
                 resample_moving: bool = False,
                 # resample_name_wo_ext: Optional[str] = None,
                 resample_subdir: str = "resampled") -> Result[XfmHandler]:

        if cls.reg_type not in ['a', 'r']:
            raise ValueError("only 'a'ffine and 'r'igid transforms currently supported by this code")

        trans_output_dir = "transforms"
        if resample_moving and resample_subdir == "tmp":
            trans_output_dir = "tmp"

        if transform_name_wo_ext:
            base = os.path.join(moving.pipeline_sub_dir, moving.output_sub_dir, trans_output_dir,
                                "%s" % (transform_name_wo_ext))
        elif generation is not None:
            base = os.path.join(moving.pipeline_sub_dir, moving.output_sub_dir, trans_output_dir,
                                "%s_ants-aff-%s" % (moving.filename_wo_ext, generation))
        else:
            base = os.path.join(moving.pipeline_sub_dir, moving.output_sub_dir, trans_output_dir,
                                "%s_ants_to_%s" % (moving.filename_wo_ext, fixed.filename_wo_ext))

        out_xfm = XfmAtom(name=f"{base}0GenericAffine.mat",
                          pipeline_sub_dir=moving.pipeline_sub_dir, output_sub_dir=moving.output_sub_dir)
        moving_resampled = ImgAtom(name=f"{base}Warped.nii.gz",
                                   pipeline_sub_dir=moving.pipeline_sub_dir, output_sub_dir=moving.output_sub_dir)
        # moving_resampled, out_xfm must be in same dir (since antsRegistrationSyNQuick.sh forces use of output prefix
        # could try moving_resampled = xfmToImg(out_xfm.newname_....) instead

        if initial_moving_transform and not hasattr(initial_moving_transform, "path"):
            raise ValueError("currently don't handle a non-file initial transform, sorry")

        c = CmdStage(cmd=["antsRegistrationSyNQuick.sh", "-d 3", f"-t {cls.reg_type}",
                          f"-f {fixed.path}", f"-m {moving.path}", f"-o {base}"] +
                         ([f"-i {initial_moving_transform.path}"] if initial_moving_transform else []) +
                         ([f"-x {fixed.mask.path}"] if fixed.mask else []),
                     inputs=(fixed, moving) + ((fixed.mask,) if fixed.mask else ()) +
                            ((initial_moving_transform,) if initial_moving_transform else ()),
                     outputs=(out_xfm, moving_resampled))

        return Result(stages=Stages([c]),
                      output=XfmHandler(xfm=out_xfm, resampled=moving_resampled, fixed=fixed, moving=moving))


class AntsRegistrationSyNQuick_rigid(AntsRegistrationSyNQuick_affine):
    reg_type = 'r'