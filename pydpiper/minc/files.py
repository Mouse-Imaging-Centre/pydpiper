import copy

# from pydpiper.core.util  import NotProvided
from abc import abstractstaticmethod, ABCMeta

from typing import Generic, TypeVar

from pydpiper.core.stages import identity_result
from pydpiper.core.files import NotProvided, FileAtom, ImgAtom

# NB: the types for this module are defined in a stub file in order to constrain the type
# of `newname_with_*` functions to return an object of the same class rather than a general FileAtom.
# If you add a function here and the typechecker can't see it being exported, that's why.


class MincAtom(ImgAtom):
    pass


class NiiAtom(ImgAtom):
    pass


class XfmAtom(FileAtom):
    """
    We create this just to be able to type check xfms. They don't need
    any more fields/information than a FileAtom, so the class functionality
    remains unchanged
    """
    pass


# nasty coercion just because newname_with returns an object of the same type
def xfmToMinc(xfm):
    mnc = copy.deepcopy(xfm)
    mnc.__class__ = MincAtom
    mnc.mask = None
    mnc.labels = None
    return mnc


I, X = TypeVar("T"), TypeVar("I")

class ToMinc(Generic[I, X], metaclass=ABCMeta):
  @abstractstaticmethod
  def to_mnc(i : I) -> MincAtom: pass

  @abstractstaticmethod
  def from_mnc(i : MincAtom) -> I: pass

  @abstractstaticmethod
  def to_mni_xfm(x : X) -> XfmAtom: pass

  @abstractstaticmethod
  def from_mni_xfm(x : XfmAtom) -> X: pass


class IdMinc(ToMinc):
  @staticmethod
  def to_mnc(i): return identity_result(i)
  @staticmethod
  def from_mnc(i): return identity_result(i)
  @staticmethod
  def to_mni_xfm(x): return identity_result(x)
  @staticmethod
  def from_mni_xfm(x): return identity_result(x)