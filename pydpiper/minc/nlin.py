
from abc import ABCMeta, abstractmethod
from typing import List, Generic, TypeVar, Optional, Sequence

from pydpiper.core.stages import Result
from pydpiper.minc.containers import GenericXfmHandler
from pydpiper.minc.files import MincAtom, ImgAtom, ToMinc

I = TypeVar('I', bound=ImgAtom)
X = TypeVar('X')


class Algorithms(Generic[I, X], metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def blur(img : I,
             fwhm : float,
             gradient : bool = True,
             subdir : str = "tmp"):
        pass

    @staticmethod
    @abstractmethod
    def average(imgs : Sequence[I],
                output_dir : str = '.',
                name_wo_ext : str = "average",
                robust : Optional[bool] = None,
                avg_file : I = None) -> Result[I]:
        pass

    @staticmethod
    @abstractmethod
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
        pass

    # must be able to handle arbitrary (not simply pure linear or pure nonlinear) transformations.
    # we should have a separate method for averaging purely affine transformations.
    # also, we should track whether or not a transform is pure affine (either by inspecting it
    # or via additional metadata in pydpiper) in order to use this more efficient functionality when possible
    @staticmethod
    @abstractmethod
    def average_transforms(xfms : Sequence[GenericXfmHandler[I, X]], avg_xfm : I) -> X: pass
    # TODO: it seems a bit heavyweight to require XfmHandlers here simply for sampling purposes

    @staticmethod
    @abstractmethod
    # a bit weird that this takes and XfmH but returns an XfmA, but the extra image data is used for sampling
    def scale_transform(xfm : Sequence[GenericXfmHandler[I, X]],
                        newname_wo_ext : str, scale : float) -> X: pass

    #def concat_xfms(): pass
    #def invert_xfm(): pass


# TODO not *actually* generic; should take a type as a field, but this is annoying to write down
class NLIN(Generic[I, X], metaclass=ABCMeta):
#class NLIN(metaclass=ABCMeta):

  # TODO replace with actual I and X modules which include these extensions!
  img_ext = NotImplemented   # type: str
  xfm_ext = NotImplemented   # type: str

  class Conf: pass

  class MultilevelConf: pass

  class ToMinc(ToMinc): pass  # TODO remove ?

  class Algorithms(Algorithms): pass

  @staticmethod
  @abstractmethod
  def hierarchical_to_single(m: 'MultiLevelConf') -> Sequence[Conf]: pass

  @staticmethod
  @abstractmethod
  def get_default_conf(resolution) -> Optional[Conf]: pass

  @staticmethod
  @abstractmethod
  def get_default_multilevel_conf(resolution) -> Optional[MultilevelConf]: pass

  @classmethod
  @abstractmethod
  def parse_protocol_file(cls, filename : str, resolution : float): pass

  # you might think it's odd to have this here, since it's not used by `register`,
  # but we want all NLIN classes to be usable for model building, and while we
  # could just give a default implementation in NLIN_BUILD_MODEL,
  # it seems a bit easier just to make the user supply it here (particularly since it can always be done,
  # and we've already implemented for minctracc, ANTS)
  @classmethod
  @abstractmethod
  def parse_multilevel_protocol_file(cls, filename : str, resolution : float): pass

  @staticmethod
  @abstractmethod
  def accepts_initial_transform(): pass

  @classmethod
  @abstractmethod
  def register(cls,
               source : I,
               target : I,
               conf : Conf,
               resample_source : bool,
               resample_subdir : str,
               transform_name_wo_ext : str = None,
               initial_source_transform : Optional[I] = None): pass

# TODO possibly these can be the same class, thus also allowing NLIN_BUILD_MODEL -> BUILD_MODEL etc.
# TODO everything in sight should probably use dataclasses instead of the 'class C( ... typevars A B C ...)' nonsense to simulate dependent records?
class LIN(NLIN): pass

class REG(NLIN): pass


class NLIN_BUILD_MODEL(NLIN, metaclass=ABCMeta):

    class BuildModelConf: pass

    @staticmethod
    @abstractmethod
    def build_model(imgs     : List[MincAtom],
                    conf     : BuildModelConf,
                    nlin_dir : str,
                    nlin_prefix : str,
                    initial_target : I,
                    robust_averaging: bool = None,
                    output_name_wo_ext : Optional[str] = None): pass

    @staticmethod
    @abstractmethod
    def parse_build_model_protocol(filename : str, resolution : float) -> BuildModelConf: pass

    @staticmethod
    @abstractmethod
    def get_default_build_model_conf() -> BuildModelConf: pass
