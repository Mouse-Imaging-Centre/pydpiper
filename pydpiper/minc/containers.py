from collections import namedtuple

from pydpiper.core.util import ensure_nonnull_return

class XfmHandler(object):
    """An xfm together with src/target images and the resampled src image.
    The idea is that a registration stage producing a transform
    will use an XfmHandler containing the transform together with the resampled image produced.
    (The initial image may be used, for instance, in a later resampling.)
    TODO give invariants such a quadruple should satisfy - e.g., may replace the images
    but in a manner which changes only intensity, not geometry."""
    # MincAtom^4 -> XfmHandler
    def __init__(self, source, xfm, target, resampled = None):
        self.source     = source
        self.xfm        = xfm
        self.target     = target
        self._resampled = resampled
    # We thought about including an inverse transform which could be generated automagically (when/how??)
    # although p.defer(XfmHandler(...)) to collect the relevant stages is a bit nasty ...

    # convenience accesor to allow `xfm.path` instead of `xfm.xfm.path`:
    path = property(lambda self: self.xfm.path, "`path` property")

    # accessing a null `resampled` field is almost certainly a programming error; throw an exception:
    resampled = property(ensure_nonnull_return(lambda self: self._resampled), "`resampled` property")

    # # some methods we haven't needed yet...
    # def update_src(self, new_src):
    #     o = copy.copy(self)
    #     o.src = new_src
    #     return o
    # def update_dest(self, new_dest):
    #     o = copy.copy(self)
    #     o.dest = new_dest
    #     return o

"""The code currently uses Result and Registration, below, but I propose the following interface
for results of pipeline components: each such component such return an object having at least
the fields of one of the following classes:"""

"""Interface for a stage returning only a MincAtom"""
MincResult     = namedtuple('MincResult',     ['stages', 'img'])
"""Interface for a stage returning an XfmHandler"""
XfmResult      = namedtuple('XfmResult',      ['stages', 'xfm'])
"""Interface for a stage returning a number of transforms and an average.
We also return an array of avg_imgs to facilitate adding additional stages to the pipeline
at a later time."""
CompoundResult = namedtuple('CompoundResult', ['stages', 'xfms', 'avg_img', 'avg_imgs'])
"""This is subject to change and currently not used since the `defer` method of a `stages`
doesn't do any kind of case analysis (though one possibility is to copy the object with `stages`
removed; the caller would still have to select out the relevant field.
This could also be done with python abstract classes or something."""

"""The API user is currently free to implement parts of a pipeline using either
functions or classes, as long as something with one of the above interfaces is returned.
Hence, an arbitrary set of inputs is allowed but only specific combinations of attributes
may be returned in order to facilitate compositionality."""

"""Result of some atom or module - a set of stages plus whatever other outputs,
which might be a single file atom (for, e.g., mincblur), a transform handler (from, e.g., minctracc),
or an array of transform handlers plus an average (for a top-level registration module such as lsq6)."""
#Result = namedtuple('Result', ['stages', 'output']) #moved to ..core/utils.py ... should be ..core/containers.py
# TODO it'd be nice to be able to say `stage=stage` instead of `stages=Stages([stage])` ...
# TODO instead of this tuple, should we flatten and require results return an object with a 'result' method?

"""A helper class with the appropriate fields for returning results from a registration such as lsq6/lsq12/nlin.
We return all (possibly only one) averages since intermediate ones may be needed later,
e.g., in order to add some images to a pipeline without re-running everything."""
Registration = namedtuple('Registration', ['xfms', 'avg_img', 'avg_imgs'])

# TODO should these atoms/modules be remade as distinct classes with public outf and stage fields (only)?
