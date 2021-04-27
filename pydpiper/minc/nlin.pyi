
from typing import Callable, List, Any, Optional, Type, TypeVar, Generic, Sequence
## don't have proper signature ascription yet, but perhaps can just symlink to ANTS.pyi, minctracc.pyi, ... ?!

from pydpiper.core.stages import Result
from pydpiper.minc.files import MincAtom, XfmAtom
from pydpiper.minc.containers import XfmHandler
#from pydpiper.minc.registration import WithAvgImgs

# In order to make Pydpiper extensible, we want to provide a way to add new components in a somewhat dynamic but
# also comprehensible way.  To aid comprehensibility, we want to take advantage of Python 3.5's type hinting.  To
# aid extensibility, we'll package functionality into more self-contained units (so that, e.g., a user who packages
# a new nonlinear registration need only pass the new unit into MBM).
#
# In Python, interface-like functionality might be implemented by either a module or a class.  We consider both of these.
#
# In the module approach, a user writes a file containing some functions and data which conforms to the relevant signature.
# Here we encounter our first problem: existing Python tooling only checks a module against an interface file with the same
# name (e.g., ANTS.py would be ascribed to ANTS.pyi).  One could imagine that symlinking might "solve" this problem,
# although this might be rather counterintuitive to users and difficult to achieve across project/repository boundaries.
# A second important problem is that code at the use site of the module has no way to specify the signature expected.
# Imagine:
#   def foo(nlin_module : typing.ModuleType):
#     #  ... invocation of nlin_module.source_to_target ...
# Here the user of nlin_module gets no guidance from type hinting (a future Python might allow
# "nlin_module : NLIN" or something).  One might think that this is already the case for dereferencing modules, but
# in the static case, e.g., Pycharm determines the interface of the statically available module.
#
# In the class approach, a user writes an abstract class and uses type hints (in the same file or in a stub file)
# to specify the interface of the component.  To implement a component, one inherits from the class.  (It doesn't seem
# possible to write the signature in a stub file and then instead implement a class with the same signature, since types
# are looked up by the name of the binding.)  One can specify default implementations which are then overridden via this
# inheritance mechanism.  Also, we don't instantiate any objects of these classes, preferring static methods.  In order
# for methods and fields to refer to each other, one may use the @classmethod decorator to pass the class as the first
# argument to the method, something which isn't necessary in the module approach.  (Alternately, one can define one's
# functions of interest outside the class definition and then refer to them in the class definition.)
#
# In general, the class-based approach doesn't suffer the limitations of the module-based one, and, although OO
# weirdness makes it seem slightly less relevant, it is probably closer in spirit to what one would write in, say, SML.
# Thus, we decide to adopt it.  Note that one other difference is that modules are singletons, i.e., importing one
# a second time has no effect.  This promotes efficiency but means that users should generally avoid IO operations that
# run at import time in their modules.
#
# Two aspects are currently lacking:
# (1) dynamic loading: we would like the ability to swap a component without modifying the main Pydpiper source tree.
# For this purpose it seems like adding some CLI/env var functionality to specify modules would be useful, combined
# with Python's __import__ functionality (as wrapped by, e.g., pydoc.locate)
# (2) existing Pydpiper functions don't automatically pass these components to their subcomponents a la Nix.
# These could be done 'by hand' at all the relevant sites, but finding a more general solution would be convenient.
#
# One other implementation issue: initialization should be done as high up as possible, to avoid repeated IO/parsing, etc.
# Currently doing this.
#
# In principle, classes aren't really needed: we could just make the type of NLIN units be a Namedtuple/Namespace type.
# (One drawback is that by accepting a `cls` arg the methods gain access to other methods in the component;
# this could be simulated by, e.g., defining the various methods as functions in the top level of the module
# and then packaging them into the record).
# If supplying default implementations isn't too hard, let's switch to this once the core ideas are working.

# TODO there is nothing NLIN-specific here; rename?
# TODO remove the build_model method since this can probably be determined from `register`, `accepts_initial_transform`,
#   and some details about the conf
#  (e.g., ANTS vs minctracc difference involves ANTS being able to run multilevel in a single call)?

I = TypeVar('I')
X = TypeVar('X')


class NLIN(Generic[I,X]):  # TODO inherit from something better; provide some default methods for parsing and model building?

  #HandlerT  # type : Type

  class Conf: ...

  class MultilevelConf: ...

  @staticmethod
  def get_default_conf() -> Optional[Conf]: ...

  @staticmethod
  def get_default_multilevel_conf() -> Optional[MultilevelConf]: ...

  @staticmethod
  def hierarchical_to_single(m : MultilevelConf) -> Sequence[Conf]: ...

  # TODO change to parse a string, not a filename? or both ...?
  @classmethod
  def parse_protocol_file(cls, filename : str, resolution : float) -> Conf: ...

  @classmethod
  def parse_multilevel_protocol_file(cls, filename : str, resolution : float) -> MultilevelConf: ...

  # TODO relying on this is a bit dangerous; instead we should wrap the return XfmHandler in some enum
  # say IncludingInitialXfm(...) or WithoutInitialXfm(...)
  # TODO this should be a property instead of a method since otherwise people might test the function
  # (always true ...) rather than its return value
  @staticmethod
  def accepts_initial_transform() -> bool: ...

  @staticmethod
  def register(source: I,
               target: I,
               initial_source_transform: Optional[X],
               conf: Conf,
               resample_source: bool,
               invert_xfm: bool,
               transform_name_wo_ext: Optional[str],
               resample_subdir: str)  \
          -> Result[XfmHandler]: ...
  # TODO you might think the resample_source field is annoying since this is conceptually separate from
  # the registration itself.  However, this can be more efficient in some cases since some tools
  # create the resampled image as part of registration.  Also, you don't have to update the xfmhandler
  # with this information later.
  # Since many algorithms probably don't create this, one could imagine writing a wrapper to fix
  # the interface and do the resampling.
  # Similar remarks apply for `invert_xfm`.

  ### the next four methods exist since our base format is Minc; with these implemented, we can automatically
  # 'mincify' any NLIN or NLIN_BUILD_MODEL implementation.  One might object that the NLIN
  # class is the wrong location for these procedures, and indeed we could include them in smaller
  # format modules (e.g. I : IMG where IMG contains the type of the image as well as to_mnc, ...,
  # and possibly even blur, average, concat, ...)


class NLIN_BUILD_MODEL(NLIN):

    class BuildModelConf: ...

    @staticmethod
    def build_model(imgs     : List[I],
                    conf     : BuildModelConf,
                    nlin_dir : str,
                    nlin_prefix : str,
                    initial_target : MincAtom,
                    #mincaverage,  # probably have to get rid of this field in this interface?
                                   # who could use it?
                    output_name_wo_ext : Optional[str] = None): ...

    @staticmethod
    def parse_build_model_protocol(filename : str, resolution : float) -> BuildModelConf: ...
