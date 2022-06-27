import os

import ordered_set
import shlex
from typing import Any, Callable, Generic, Iterable, List, Set, Tuple, TypeVar, Union, Optional, Dict
from pydpiper.core.files import FileAtom

class CmdStage(object):
    """A simplified command stage - one could write a simple conversion
    function or simply adopt the old one.  I prefer separating static
    (command, memory limits) from dynamic (status, retries) information
    in part because it avoids many empty fields for stages which never
    become part of a `live` pipeline."""
    def __init__(self,
                 # `List`s don't work here because mutable containers must be _invariant_
                 # (see en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science));
                 # instead, we abuse variadic tuples to simulate immutable lists:
                 inputs   : Tuple[FileAtom, ...],
                 outputs  : Tuple[FileAtom, ...],
                 cmd      : List[str],
                 memory   : float = None,
                 procs    : int = 1,
                 log_file : Optional[str] = None,
                 env_vars : Dict[str,str] = None,
                 category : Optional[str] = None) -> None:
        # TODO: rather than having separate cmd_stage fn, might want to make inputs/outputs optional here
        self.inputs  = inputs          # type: Tuple[FileAtom, ...]
        # TODO: might be better to dereference inputs -> inputs.path here to save mem
        self.outputs = outputs         # type: Tuple[FileAtom, ...]
        #self.conf    = conf           # not needed at present -- see note on render_fn
        self._cmd    = [str(x) for x in cmd] # type: List[str]
        # TODO: why not expose this publicly?
        self.when_runnable_hooks = []  # type: List[Callable[[], Any]]
        # TODO: make the hooks accessible via the constructor?
        self.when_finished_hooks = []  # type: List[Callable[[], Any]]
        self.memory = memory
        self.procs = procs
        self.category = category or self._cmd[0]  # TODO using the cmd name seems like a good heuristic, but what about the case of wrapping programs ??
        # some cosmetics: we would like the log files to reside in the "log" subdirectory. For most mincAtoms, we can
        # access this directory by using its "dir" and adding "../log". This is not true for files that live in the
        # _nlin or _lsq12 directories though. They live in their own top level directory, so we should just create
        # a log directory in there

        self.log_file = log_file or (os.path.join(self.outputs[0].dir,
                                       ".." if self.outputs[0].dir != self.outputs[0].pipeline_sub_dir else "" ,
                                       "log",
                                       cmd[0], "%s.log" % self.outputs[0].filename_wo_ext)
                         if len(self.outputs) >= 1 else None)  # FIXME: for |self.outputs| > 1, this is a fragile hack
        self.env_vars = env_vars if env_vars is not None else {}
    # NB: __hash__ and __eq__ ignore hooks, memory
    # Also, we assume cmd determines inputs, outputs so ignore it in hash/eq calculations
    # FIXME: we should make the CmdStage fields immutable (via properties) to prevent hashing-related bugs
    def __hash__(self) -> int:
        return hash(self.cmd_to_string())
    def __eq__(self, c) -> bool:
        return self._cmd == c._cmd
    # Originally I had `render_fn` : inputs, outputs, conf -> [str] instead of `cmd` : [str] to
    # (1) reduce duplication by not encoding the inputs/outputs in two places
    # (2) abstract away some of the boring parts, like getting the name fields out of atoms
    # the Input(...), Output(...) idea used in the old-style CmdStage is also OK, at the expense
    # of heterogeneous arrays (and problems with object equality/hashing/reading/printing/...)
    # Using a property, one could write stage.cmd instead of stage.cmd()
    #TODO: not very clear what render means?
    def render(self) -> str:
        #return self.render_fn(self.conf, self.inputs, self.outputs)
        return self.cmd_to_string()
    def cmd_to_string(self) -> str:
        return ' '.join(str(x) for x in self._cmd)
    def to_array(self) -> List[str]:
        """Form usable for Python subprocess call."""
        return self._cmd
    def set_log_file(self, log_file_name: str) -> None:
        self.log_file = log_file_name

    #def execute(self, backend):    # could also be elsewhere...
    #    raise NotImplemented

# def cmd_stage(cmd : List[Union[str, InputFile, OutputFile]]) -> CmdStage:
#     """'Smart constructor' for command stages using heterogeneous list API of the old command stage"""
#     inputs  = [s for s in cmd if isinstance(s, InputFile)]    # type: List[InputFile]
#     outputs = [s for s in cmd if isinstance(s, OutputFile)]   # type: List[OutputFile]
#     # we used to store strings/filenames in PipelineFiles (and the InputFile
#     # and OutputFile children). Currently what we store are FileAtoms and
#     # MincAtoms. So it might seem strange in the next line to see s.filename, 
#     # but that is where we currently store those File/MincAtoms. This will
#     # eventually all go away.
#     _cmd = [(s.filename.path if isinstance(s, PipelineFile) else s) for s in cmd]  # type: List[str]
#     return CmdStage(inputs = inputs, outputs = outputs, cmd = _cmd)

T = TypeVar('T')

class Stages(ordered_set.OrderedSet):
    """A set of stages to be run.  In addition to the usual set operations,
    contains a single extra method, `defer`.  The idea here is as follows: procedures
    to create various commands and pipelines will return both
    (1) a set of stages to be run, and
    (2) a data structure representing the files created as a result.
    In PydPiper 1.x, (2) is done implicitly by modifying the procedures' inputs,
    alleviating the annoyance of handling two conceptually unrelated return values.
    One option would be to make (1) implicit by creating a global (i.e., module-level)
    Stages variable to which all stages would be added (though this can be rather
    mysterious, and loses the flexibility to, e.g., add stages to various separate pipelines,
    such as on both local/remote processors).  I currently opt to return both values, hence 
    `defer`, which accumulates the stages from a command and passes on the relevant files unchanged.

    The name is not too descriptive and I'm open to suggestions,
    but it's short since it will be called often.  Possible names: extract_stages,
    extract_stages_and_return_output_files, defer(red(_result)), output, ....
    Part of the problem is that this method really is somewhat silly (since
    two return values is silly as well).
    
    One could always just manually unpack the return value and avoid this method:
      xfms = p.defer(cmd(...)); ...xfms...
    is the same as
      stages, xfms = cmd(...)
      p.add(stages)
      ...xfms...

    Note: PydPiper 1.x uses a Pipeline where we use a Stages, but 
    we create many intermediate structures for which the extra fields of a pipeline
    don't have any meaning, so this might be worth changing for clarity.
    """
    def __init__(self, e : Union[Iterable[CmdStage], List[CmdStage]] = ()) -> None:
        super(self.__class__, self).__init__(iter(e))
    def defer(self, result : 'Result[T]') -> T:
        self.update(result.stages)
        return result.output
    def defer_all(self, results : Tuple['Result[T]']) -> Tuple[T]:
        # eventually this will have better semantics than `map defer`
        results = tuple(results)  # TODO decide what type this should be (n-tuple, iterable, etc)
        for result in results:
          self.update(result.stages)
        return tuple(result.output for result in results)
    # TODO this now remembers the order stages were added (due to use of the strangely-named `OrderedSet` package)
    # but due to randomization in iteration order over various data structures, the pipeline_stages files will
    # still be reordered across runs, which is annoying ... might want to fix the random seed or something ...

# TODO make it possible to inline many inputs somehow (using cooperation from the string formatter?)
def parse(cmd_str : str) -> CmdStage:
    """Create a CmdStage object from a string.  (Per Jason's suggestion, we could make
    an even more clever parser that simply guesses the output based on position
    or a few flags like -o and tags anything else that looks like a file as input.)
    >>> val = 2
    >>> c = parse('mincfoo -flag {val} ,input1.mnc ,input2.mnc @output.mnc'.format(**vars()))
    >>> c.outputs
    ['output.mnc']
    >>> c.inputs
    ['input1.mnc', 'input2.mnc']
    """
    cmd = shlex.split(cmd_str)
    inputs  = tuple([FileAtom(s[1:]) for s in cmd if s[0] == ','])  #TODO what about I(), O() ?
    outputs = tuple([FileAtom(s[1:]) for s in cmd if s[0] == '@'])
    s = CmdStage(inputs = inputs, outputs = outputs, cmd = [c if c[0] not in [',','@'] else c[1:] for c in cmd])
    return s


class Result(Generic[T]):
    def __init__(self, stages : Stages, output : T) -> None:
        self.stages = stages # type: Stages
        self.output = output # type: T


def identity_result(x : T) -> Result[T]:
    return Result(stages=Stages(), output=x)
