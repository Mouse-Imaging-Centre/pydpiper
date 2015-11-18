# Stubs for pydpiper.minc.files (Python 3.5)

import typing

from pydpiper.core.files import FileAtom

class MincAtom(FileAtom):
    mask   = ... # type: MincAtom
    labels = ... # type: MincAtom
    def __init__(self, name : str,
                       orig_name : str = ...,
                       pipeline_sub_dir : str = None,
                       output_sub_dir : str = None,
                       mask : MincAtom = None,
                       labels : MincAtom = None,
                       resolution : MincAtom = None) -> None: ...

class XfmAtom(FileAtom): ...
