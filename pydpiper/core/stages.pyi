# Stubs for pydpiper.core.stages (Python 3.5)

from typing import Any, Callable, Generic, Sequence, Set, Union, TypeVar
#from typing import Any, Generic, TypeVar

from pydpiper.core.files import FileAtom
from pydpiper.execution.pipeline import InputFile, OutputFile  # type: ignore

class CmdStage:
    inputs  = ...             # type: Sequence[FileAtom]
    outputs = ...             # type: Sequence[FileAtom]
    when_runnable_hooks = ... # type: List[Callable[[], Any]]
    when_finished_hooks = ... # type: List[Callable[[], Any]]
    memory = ...              # type: float
    def __init__(self, inputs  : Sequence[FileAtom],
                       outputs : Sequence[FileAtom],
                       cmd     : str,
                       memory  : float = None) -> None: ...
    def __hash__(self)      -> int: ...
    def __eq__(self, c)     -> bool: ...
    def render(self)        -> str: ...
    def cmd_to_string(self) -> str: ...
    def to_array(self)      -> List[str]: ...

def cmd_stage(cmd : List[Union[str, InputFile, OutputFile]]) -> CmdStage: ...

def parse(cmd_str : str) -> CmdStage: ...

T = TypeVar('T')

class Stages(set):
    def defer(self, result : 'Result[T]') -> T: ...

class Result(Generic[T]):
    stages = ... # type: 'Stages'
    output = ... # type: T
    def __init__(self, stages : Stages, output : T) -> None: ...

