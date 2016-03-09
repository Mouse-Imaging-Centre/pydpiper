import copy
from typing import Optional

from pydpiper.core.util  import ensure_nonnull_return
from pydpiper.minc.files import MincAtom, XfmAtom

#TODO: might move elsewhere?

class XfmHandler(object):
    """An xfm together with src/target images and the resampled src image.
    The idea is that a registration stage producing a transform
    will use an XfmHandler containing the transform together with the resampled image produced.
    (The initial image may be used, for instance, in a later resampling.)
    TODO give invariants such a quadruple should satisfy - e.g., may replace the images
    but in a manner which changes only intensity, not geometry.
    
    resampled -- this is the file we care about in terms of 
                 continuing with the pipeline. It does not necessarily have to be
                 the source file resampled with the xfm. For instace, if
                 the intensities of the source were first manipulated, and
                 then the xfm applied, that file will be stored in resampled.
    """
    def __init__(self,
                 source    : MincAtom,
                 xfm       : XfmAtom,
                 target    : MincAtom,
                 resampled : Optional[MincAtom] = None) -> None:
        self.source     = source
        self.xfm        = xfm
        self.target     = target
        self._resampled = resampled
    # We thought about including an inverse transform which could be generated automagically (when/how??)
    # although s.defer(XfmHandler(...)) to collect the relevant stages is a bit nasty ...

    def __repr__(self) -> str:
        return "%s(xfm=%s)" % (self.__class__, self.xfm.path) 

    # convenience accessor to allow `xfm.path` instead of `xfm.xfm.path`:
    # (turned off because it was a bad idea due to lack of types)
    #path = property(lambda self: self.xfm.path, "`path` property")

    # accessing a null `resampled` field is almost certainly a programming error; throw an exception:
    @property
    def resampled(self) -> MincAtom:
        return ensure_nonnull_return(lambda self: self._resampled)(self)
    #resampled = property(ensure_nonnull_return(lambda self: self._resampled), "`resampled` property")

    def replace(self, **kwargs):
        o = copy.copy(self)
        for k, v in kwargs.items():
            if hasattr(o, k):
                setattr(o, k, v)
            else:
                raise ValueError("no field %s" % k)
        return o

    # # some methods we haven't needed yet...
    # def update_src(self, new_src):
    #     o = copy.copy(self)
    #     o.src = new_src
    #     return o
    # def update_dest(self, new_dest):
    #     o = copy.copy(self)
    #     o.dest = new_dest
    #     return o
