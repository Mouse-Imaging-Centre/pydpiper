
from abc import ABCMeta, abstractmethod
from typing import List, Generic, TypeVar, Optional, Sequence

from pydpiper.core.stages import Result, Stages
from pydpiper.minc.containers import GenericXfmHandler
from pydpiper.minc.files import MincAtom, ImgAtom

I = TypeVar('I', bound=ImgAtom)
X = TypeVar('X')

#from dataclasses import dataclass

#@dataclass
class Algorithms():  #(Generic[I, X], metaclass=ABCMeta):

    I : type
    X : type

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
    # [bug: some of the xfmavg and xfmavg_scipy.py scripts assume exactly zero or one of each of affine and grid!]
    @staticmethod
    @abstractmethod
    def average_transforms(xfms : Sequence[GenericXfmHandler[I, X]], avg_xfm : I) -> X: raise NotImplementedError
    # TODO: it seems a bit heavyweight to require XfmHandlers here simply for sampling purposes

    @staticmethod
    @abstractmethod
    def average_affine_transforms(xfms, avg_xfm): raise NotImplementedError

    @staticmethod
    @abstractmethod
    # a bit weird that this takes and XfmH but returns an XfmA, but the extra image data is used for sampling
    def scale_transform(xfm : Sequence[GenericXfmHandler[I, X]],
                        newname_wo_ext : str, scale : float) -> Result[X]: pass

    @staticmethod
    @abstractmethod
    def xfminvert(xfm, subdir: str = "transforms") -> Result[X]: pass  # TODO rename?

    @classmethod
    @abstractmethod
    def invert_xfmhandler(cls,
                          xfm: GenericXfmHandler[I, X],
                          subdir: str = "transforms") -> Result[GenericXfmHandler[I, X]]:
        """
        xfminvert lifted to work on XfmHandlers instead of MincAtoms
        """
        # TODO: are we potentially creating some issues here... consider the following case:
        # TODO: minctracc src.mnc final_nlin.mnc some_trans.xfm
        # TODO: the xfmhandler will have:
        # TODO: source = src.mnc
        # TODO: target = final_nlin.mnc
        # TODO: xfm    = some_trans.xfm
        #
        # TODO: however, when we generate the inverse of this transform, we might want the
        # TODO: result to be:
        # TODO: source = src_resampled_to_final_nlin.mnc
        # TODO: target = src.mnc
        # TODO: xfm    = some_trans_inverted.xfm
        #
        # TODO: instead we have "final_nlin.mnc" as the source for this XfmHandler... we might
        # TODO: run into issues here...
        # TODO: we also discard any resampled field; we might want to use part of the original xfmh
        # or do resampling here, but that seems like too much functionality to put here ... ??
        s = Stages()
        if xfm.has_inverse():
            inv_xfm = xfm.inverse.xfm
        else:
            inv_xfm = s.defer(cls.xfminvert(xfm.xfm, subdir=subdir))
        return Result(stages=s,
                      output=GenericXfmHandler(xfm=inv_xfm,
                                               source=xfm.target, target=xfm.source, resampled=None,
                                               inverse=xfm))  # TODO is it correct to have the original resampled here?

    #def concat_xfms(): pass
    #def invert_xfm(): pass

    # seemingly needs to take an xfmhandler instead of an xfm since minc_displacement need a -like volume?
    @staticmethod
    def compute_log_determinant(xfm : GenericXfmHandler) -> I: pass   # TODO add fwhm argument


# TODO not *actually* generic; should take a type as a field, but this is annoying to write down
class NLIN():  #, metaclass=ABCMeta):
#class NLIN(metaclass=ABCMeta):

  I : type
  X : type

  # TODO replace with actual I and X modules which include these extensions!
  img_ext = NotImplemented   # type: str
  xfm_ext = NotImplemented   # type: str

  class Conf: pass

  class MultilevelConf: pass

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

  #@classmethod
  @staticmethod
  @abstractmethod
  def register(#cls,
               fixed : I,
               moving : I, *,
               conf : Conf,
               resample_moving : bool,
               resample_subdir : str,
               transform_name_wo_ext : str = None,
               initial_moving_transform : Optional[I] = None): raise NotImplementedError

# TODO possibly these can be the same class, thus also allowing NLIN_BUILD_MODEL -> BUILD_MODEL etc.
# TODO everything in sight should probably use dataclasses instead of the 'class C( ... typevars A B C ...)' nonsense to simulate dependent records?
class LIN(NLIN): pass

class REG(NLIN): pass


class NLIN_BUILD_MODEL(NLIN):  #, metaclass=ABCMeta):

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
