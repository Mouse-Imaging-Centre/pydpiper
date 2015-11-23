# Stubs for pydpiper.minc.files (Python 3.5)

from typing import Callable
from pydpiper.core.files import FileAtom

class MincAtom(FileAtom):
    mask   = ... # type: MincAtom
    labels = ... # type: MincAtom
    def __init__(self,
                 name : str,
                 orig_name : str = ...,
                 pipeline_sub_dir : str = None,
                 output_sub_dir : str = None,
                 mask : 'MincAtom' = None,
                 labels : 'MincAtom' = None) -> None: ...

    def newname_with_fn(self, fn : Callable[[str], str], ext : str = ..., subdir : str = ...) -> MincAtom: ...
    def newname_with_suffix(self, suffix : str, ext : str = ..., subdir : str = ...) -> MincAtom: ...

class XfmAtom(FileAtom):
    def newname_with_suffix(self, suffix : str, ext : str = ..., subdir : str = ...) -> XfmAtom: ...
    def newname_with_fn(self, fn : Callable[[str], str], ext : str = ..., subdir : str = ...) -> XfmAtom: ...
