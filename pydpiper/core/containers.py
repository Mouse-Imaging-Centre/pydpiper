from typing import TypeVar, Generic

T = TypeVar('T')

class Result(Generic[T]):
    def __init__(self, stages, output):
        self.stages = stages
        self.output = output
