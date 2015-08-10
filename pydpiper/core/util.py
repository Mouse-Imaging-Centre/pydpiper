#!/usr/bin/env python2.7

import os.path
from operator import add

def lift_to_keys(f):
    """Take a length-preserving list->list function and lift to work on values
    of a dict, preserving the 'shape' (easy to formalize). In fact, we (currently)
    guarantee the stronger property that `f` gets the list ordered by keys,
    but in applications the `f` given generally shouldn't care about the order
    of the list (??? - what if it chooses a target based on 1st elt?)
    so all we care about is that each key gets a new value from the same position in the
    list as the old value.

    >>> def f(l): return enumerate(l)
    >>> (lift_to_keys(f)({'a':9, 'b':2, 'c':2, 'd':4})
    ...   == {'a':(0,9), 'b':(1,2), 'c':(2,2), 'd':(3,4)})
    True
    """
    def g(m):
        l  = [(k,v) for k, v in sorted(m.iteritems())]
        ks = [k for k, _ in l]
        vs = [v for _, v in l]
        return dict(zip(ks,f(vs)))
    return g

def unlift_to_list(f):
    """Take a 'shape-preserving' function acting on a dictionary
    (but preserving the keys) and convert it to one operating
    on a list by making the indexing explicit.

    # TODO doesn't have to be strictly 'elementwise' - make a better example
    >>> def f(d): return dict([(k,v+1) for k,v in d.iteritems()])
    >>> (unlift_to_list(f)([1,3,4,7,11]) == [2,4,5,8,12])
    True
    """
    return lambda l: [v for _, v in sorted(f(dict(enumerate(l))).iteritems())]

def raise_(err):
    """`raise` is a keyword and `raise e` isn't an expression, so can't be used freely"""
    raise err

def explode(filename):
    # TODO should this be a namedtuple instead of a tuple? Probably...can still match against, etc.
    """Split a filename into a directory, 'base', and extension

    >>> explode('/path/to/some/file.ext')
    ('/path/to/some', 'file', '.ext')
    >>> explode('./relative_path_to/file.ext')
    ('./relative_path_to', 'file', '.ext')
    >>> explode('file')
    ('', 'file', '')
    """
    base, ext = os.path.splitext(filename)
    directory, name = os.path.split(base)
    return (directory, name, ext)

def pairs(lst):
    return zip(lst[:-1], lst[1:])

def flatten(*xs):
    return reduce(add, xs, [])

class NotProvided(BaseException):
    """To be used as a datatype indicating no argument with this name was supplied
    in the situation when None has another sensible meaning, e.g., when a sensible
    default file is available but None indicates *no* file should be used."""

class NoneError(ValueError): pass

def ensure_nonnull_return(f):
    def g(*args):
        result = f(*args)
        if result is None:
            raise NoneError
        else:
            return result
    return g

def output_directories(stages):
    """Directories to be created (currently rather redundant in the presence of subdirectories)"""
    return { os.path.dirname(o) for c in stages for o in c.outputs }

# this doesn't seem possible to do properly ... but we could turn a namespace into another type
# and use that as a default object, say
#def make_parser_and_class(**kwargs):
#    """Create both an argparser and a normal class with the same constraints"""
#    p = argparse.ArgumentParser(**kwargs)
#    fields = [v.... for v in p._option_store_actions.values()]
#    pass #TODO use p._option_string_actions ?
