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


def mincToXfm(mnc):
    xfm = copy.deepcopy(mnc)
    del xfm.mask
    del xfm.labels
    xfm.__class__ = XfmAtom
    return xfm


I, X = TypeVar("T"), TypeVar("I")