from pydpiper.core.util  import NotProvided
from pydpiper.core.files import FileAtom

# NB: the types for this module are defined in a stub file in order to constrain the type
# of `newname_with_*` functions to return an object of the same class rather than a general FileAtom.
# If you add a function here and the typechecker can't see it being exported, that's why.

class MincAtom(FileAtom):
    def __init__(self, name, orig_name=NotProvided(), pipeline_sub_dir=None,
                 output_sub_dir=None, mask=None, labels=None):
        super(self.__class__, self).__init__(name=name, orig_name=orig_name,
                                             pipeline_sub_dir=pipeline_sub_dir,
                                             output_sub_dir=output_sub_dir)
        self.mask   = mask
        self.labels = labels
    # TODO should newname_with be overloaded with new behaviour for mask/labels???  We could get a different
    # behaviour for free if the FileAtom.newname_with used copy.copy() internally
    # some operations (blurring) preserve mask/labels, but others don't (resampling) ... add a preserve_labels=<bool> arg?

class XfmAtom(FileAtom):
    """
    We create this just to be able to type check xfms. They don't need
    any more fields/information than a FileAtom, so the class functionality
    remains unchanged
    """
    pass

# nasty coercion just because newname_with returns an object of the same type
def xfmToMinc(xfm):
    mnc = copy.deepcopy(xfm)
    mnc.__class__ = MincAtom
    mnc.mask = None
    mnc.labels = None
    return mnc
