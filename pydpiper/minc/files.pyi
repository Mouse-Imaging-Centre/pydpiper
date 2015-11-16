# Stubs for pydpiper.minc.files (Python 3.5)

import typing

from pydpiper.core.files import FileAtom

class MincAtom(FileAtom):
    mask   = ... # type: MincAtom
    labels = ... # type: MincAtom
    def __init__(self, name, orig_name=..., pipeline_sub_dir : str = None, output_sub_dir : str = None, mask=None, labels=None, resolution=None) -> None: ...
