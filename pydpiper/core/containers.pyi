# Stubs for pydpiper.core.containers (Python 3.5)

from pydpiper.core.stages import Stages
from typing import Any, Generic, TypeVar

T = TypeVar('T') 
 # had type: Any

class Result(Generic[T]):
    stages = ... # type: Stages
    output = ... # type: T
    def __init__(self, stages : Stages, output : T) -> None: ...
