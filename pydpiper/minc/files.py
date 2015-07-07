from pydpiper.core.util  import NotProvided
from pydpiper.core.files import FileAtom

class MincAtom(FileAtom):
    def __init__(self, name, orig_name=NotProvided, curr_dir=None, work_dir=None, mask=None, labels=None, resolution=None):
        super(self.__class__, self).__init__(name=name, orig_name=orig_name, work_dir=work_dir, curr_dir=curr_dir)
        self.mask   = mask
        self.labels = labels
        #if resolution is not None: #    self.resolution = resolution
        # FIXME try to go out to disk for the resolution.
        # If done excessively, this could be quite slow on some systems ... add caching on self.path?
        # Even better, we could have a separate constructor inputMincAtom which would be the only one to try to do this
        #elif ...:
        #else:
        #    raise ValueError
    # TODO should newname_with be overloaded with new behaviour for mask/labels???  We could get a different
    # behaviour for free if the FileAtom.newname_with used copy.copy() internally
    # some operations (blurring) preserve mask/labels, but others don't (resampling) ... add a preserve_labels=<bool> arg?
