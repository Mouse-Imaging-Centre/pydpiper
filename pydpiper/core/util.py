import os
import functools  # type: ignore
from operator import add
from enum import Enum
import typing
from typing import Any, Callable, List, Set, Tuple, TypeVar

def maybe_deref_path(x):
    # ugh ... just a convenience to allow using applymap in a 'generic' way ...
    #if isinstance(x, FileAtom) or isinstance(x, XfmAtom):
    #    return x.path
    # doesn't work in typing 3.6.1:
    #elif isinstance(x, GenericXfmHandler) or isinstance(x, XfmHandler):
    #    return x.xfm.path
    # workaround:
    if hasattr(x, 'xfm'):
        return x.xfm.path
    elif hasattr(x, 'path'):
        return x.path
    else:
        return x


class AutoEnum(Enum):
    """An enumeration that doesn't require you to explicitly give the injection.
    Copied from docs.python.org/3/library/enum.html since apparently it's
    not worth including in the stdlib ..."""
    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


def NamedTuple(name : str, fields : List[Tuple[str, Callable[[Any], Any]]]):
    """Like typing.NamedTuple, but with some extra functions for (non-destructively) updating fields."""
    F = typing.NamedTuple(name, fields + [("flags_", set)])
    F.replace = F._replace
    def with_flags(f):
        def g(*args, **kwargs):
            # this doesn't work for some slightly unclear reason - perhaps one NamedTuple is inheriting from another,
            # causing `with_flags` to be called multiple times on `__new__`:
            #return f(*args, flags_=set(), **kwargs)
            # this works:
            l = len(args) + len(kwargs)
            if l == len(fields) + 1:  # the extra 1 is for the class name
                return f(*args, flags_=set(), **kwargs)
            elif l == len(fields) + 2:
                return f(*args, **kwargs)
            else:
                raise ValueError("!?")
        return g
    F.__new__ = with_flags(F.__new__)
    F.maybe_replace = lambda self, **kwargs: F._replace(self, **{k:v for (k,v) in kwargs.items() if v is not None})
    return F


T = TypeVar('T')
U = TypeVar('U')


def raise_(err: BaseException):
    """`raise` is a keyword and `raise e` isn't an expression, so can't be used freely"""
    raise err




def pairs(lst: List[T]) -> List[Tuple[T, T]]:
    return list(zip(lst[:-1], lst[1:]))


def flatten(*xs):
    return functools.reduce(add, xs, [])


def output_directories(stages: Set['pydpiper.execution.pipeline.CmdStage']) -> Set[str]:
    """Directories to be created (currently rather redundant in the presence of subdirectories).
    No need to consider stage inputs - any input already exists or is also the output
    of some previous stage."""
    # What if an output file IS a directory - should there be a special
    # way of tagging this other than ending the path in a trailing "/"?
    all_files_to_consider = []
    for stage in stages:
        all_files_to_consider = all_files_to_consider + stage.outputFiles
        if stage.logFile:
            all_files_to_consider.append(stage.logFile)
        else:
            print("no log file for stage:")
            # old style CmdStage (from execution/pipeline.py) uses __repr__ to print itself
            print(stage)
            raise ValueError("No logfile!!")
    return {os.path.dirname(filename) for filename in all_files_to_consider}

# this doesn't seem possible to do properly ... but we could turn a namespace into another type
# and use that as a default object, say
# def make_parser_and_class(**kwargs):
#    """Create both an argparser and a normal class with the same constraints"""
#    p = argparse.ArgumentParser(**kwargs)
#    fields = [v.... for v in p._option_store_actions.values()]
#    pass #TODO use p._option_string_actions ?

