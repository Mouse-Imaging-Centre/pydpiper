
from typing import List, Generic, TypeVar, Optional, Sequence

from pydpiper.core.stages import Result
from pydpiper.minc.containers import GenericXfmHandler
from pydpiper.minc.files import MincAtom, ImgAtom, ToMinc

I = TypeVar('I', bound=ImgAtom)
X = TypeVar('X')

class Algorithms(Generic[I, X]):
    @staticmethod
    def blur(img : I,
             fwhm : float,
             gradient : bool = True,
             subdir : str = "tmp"):
        raise NotImplementedError

    @staticmethod
    def average(imgs : Sequence[I],
                output_dir : str = '.',
                name_wo_ext : str = "average",
                avg_file : I = None) -> Result[I]:
        raise NotImplementedError

    @staticmethod
    def resample(img: I,
                 xfm: X,  # TODO: update to handler?
                 like: I,
                 invert: bool = False,
                 use_nn_interpolation : Optional[bool] = None,
                 #interpolation = None,   #interpolation: Interpolation = None,
                 #  TODO fix type for non-minc resampling programs; also, can't import Interpolation here
                 #extra_flags: Sequence[str] = (),
                 new_name_wo_ext: str = None,
                 subdir: str = None,
                 postfix: str = None) -> Result[I]:
        raise NotImplementedError

    # must be able to handle arbitrary (not simply pure linear or pure nonlinear) transformations.
    # we should have a separate method for averaging purely affine transformations.
    # also, we should track whether or not a transform is pure affine (either by inspecting it
    # or via additional metadata in pydpiper) in order to use this more efficient functionality when possible
    @staticmethod
    def average_transforms(xfms : Sequence[GenericXfmHandler[I, X]], avg_xfm : I) -> X:
        raise NotImplementedError
    # TODO: it seems a bit heavyweight to require XfmHandlers here simply for sampling purposes

    @staticmethod
    # a bit weird that this takes and XfmH but returns an XfmA, but the extra image data is used for sampling
    def scale_transform(xfm : Sequence[GenericXfmHandler[I, X]],
                        newname_wo_ext : str, scale : float) -> X:
        raise NotImplementedError

    #def concat_xfms(): pass
    #def invert_xfm(): pass




# TODO not *actually* generic; should take a type as a field, but this is annoying to write down
class NLIN(Generic[I, X]):

  # TODO replace with actual I and X modules which include these extensions!
  img_ext = NotImplemented   # type: str
  xfm_ext = NotImplemented   # type: str

  class Conf: raise NotImplementedError

  class MultilevelConf: raise NotImplementedError

  class ToMinc(ToMinc): raise NotImplementedError

  class Algorithms(Algorithms): raise NotImplementedError

  @staticmethod
  def hierarchical_to_single(m: 'MultiLevelConf') -> Sequence[Conf]: raise NotImplementedError

  @staticmethod
  def get_default_conf(resolution) -> Optional[Conf]: raise NotImplementedError

  @staticmethod
  def get_default_multilevel_conf(resolution) -> Optional[MultilevelConf]: raise NotImplementedError

  @classmethod
  def parse_protocol_file(cls, filename : str, resolution : float): raise NotImplementedError

  # you might think it's odd to have this here, since it's not used by `register`,
  # but we want all NLIN classes to be usable for model building, and while we
  # could just give a default implementation in NLIN_BUILD_MODEL,
  # it seems a bit easier just to make the user supply it here (particularly since it can always be done,
  # and we've already implemented for minctracc, ANTS)
  @classmethod
  def parse_multilevel_protocol_file(cls, filename : str, resolution : float): raise NotImplementedError

  @staticmethod
  def accepts_initial_transform(): raise NotImplementedError

  @classmethod
  def register(cls,
               source : I,
               target : I,
               conf : Conf,
               resample_source : bool,
               resample_subdir : str,
               transform_name_wo_ext : str = None,
               initial_source_transform : Optional[I] = None): raise NotImplementedError



class NLIN_BUILD_MODEL(NLIN):

    class BuildModelConf(): raise NotImplementedError

    @staticmethod
    def build_model(imgs     : List[MincAtom],
                    conf     : BuildModelConf,
                    nlin_dir : str,
                    nlin_prefix : str,
                    initial_target : I,
                    #mincaverage,
                    output_name_wo_ext : Optional[str] = None): raise NotImplementedError

    @staticmethod
    def parse_build_model_protocol(filename : str, resolution : float) -> BuildModelConf: raise NotImplementedError

    @staticmethod
    def get_default_build_model_conf() -> BuildModelConf: raise NotImplementedError
