import os

from traits.api import *

from .util import explode, NotProvided

class FileAtom(HasTraits):
    """
        What is stored:
        self.orig_path -- original input file, full file name and path (/path/filename.ext), can be None
        self.dir
        self.filename_wo_ext
        self.ext
        self.output_dir - in terms of the old code, this is the directory that will hold
                          the resampled/, tmp/, stats-volume/ etc. directories.
    """
    def __init__(self, name, orig_name=NotProvided, output_dir=None):
        self.dir, self.filename_wo_ext, self.ext = explode(os.path.abspath(name))
        if orig_name is NotProvided:
            self.orig_path = self.path
        elif orig_name is None:
            # is this case even needed? if no orig file, when _isn't_ file itself the orig file?
            self.orig_path = None
        else:
            self.orig_path = os.path.abspath(orig_name)
        #self.orig_path = self.path if orig_name == NotProvided else (os.path.abspath if orig_name else id)(orig_name)
        # NOTE we use NotProvided here as a default, since None has another meaning (no associated original file)
        # curr_dir seems more sensible than original dir as work_dir
        # TODO make it possible to avoid creating a subdirectory??
        """Derived files created by commands will be placed by default into (a subdir of) the work_dir"""
        # TODO this is a bit questionable since if you create a new minc file from scratch (as opposed to cloning)
        # then work_dir is reset to curr_dir, which is probably not what you want, but you'll forget to fix that...
        # OTOH, the problem with always cloning is that you keep too much (orig_name, mask/labels, ...)
        #self.work_dir  = os.path.join(os.path.abspath(work_dir) if work_dir else os.getcwd(), self.name)  # was sort of working ...
        # FIXME the problem here is that work_dir internally means something different from work_dir in the __init__ call,
        # which is more like a cwd, so the work_dir will be created inside it
        if output_dir is not None:
            self.output_dir = os.path.abspath(output_dir)
        else:
            #TODO: is this what we want...?
            self.output_dir = os.path.join(os.getcwd(), self.filename_wo_ext)
        #self.work_dir = os.path.join(curr_dir if curr_dir else os.getcwd(), self.name) #work_dir if work_dir else os.path.join(os.getcwd(), self.name)

    def get_path(self):
        #TODO: in order to avoid storing duplicate information, is this what we want? 
        (d,n,e) = (self.dir, self.filename_wo_ext, self.ext)
        return os.path.join(d, n + e)

    def newname_with_fn(self, fn, ext=None, subdir=None):
        """Create a new FileAtom from one representing <dirs>/<file><.ext>
        now representing <original file's output_dir>/[<subdir>]/f(<file>)<.ext>.

        This isn't particularly general since one might want the result in a completely
        different location, but in this case one can just call the constructor again,
        since other properties (especially of subclasses like MincAtom) are likely to
        change as well in this case.  Similarly, we don't yet provide an option to
        override the output_dir.  It seems that this situation isn't optimal since,
        whether one calls newname_with_fn or creates a FileAtom by hand, the API doesn't force
        one to consider which properties should be cloned, set by hand, or reset to some defaults.
        One could in principle write a copy_with method specifying exactly how things should go,
        but it's not clear this would work well with subclassing (e.g.,
        no reuse might be possible).
       
        # (we use absolute paths so the tests are deterministic without requiring postprocessing)
        >>> f1 = FileAtom(name='/project/images/img_1.mnc', output_dir="/scratch/pipeline/")
        >>> f2 = f1.newname_with_fn(fn=lambda n: n + '_fwhm0.056')
        >>> f2.path
        '/scratch/pipeline/img_1_fwhm0.056.mnc'
        >>> f1.path
        '/project/images/img_1.mnc'
        """

        _dir, name, old_ext = (self.dir, self.filename_wo_ext, self.ext)
        output_dir = self.output_dir
        ext  = ext or old_ext
        name = os.path.join(os.path.join(output_dir, subdir) if subdir else output_dir, fn(name) + ext)
        # FIXME should return something of the same class, but can't call self.__class__.__init__(...)
        # here as init fn may have different signature => need to copy and set attrs and/or override
        # this method/make it abstract
        # As an example of the considerations involved, when updating a minc_atom to an XFM (should this even happen???)
        # we want to discard mask/orig_file, but when blurring we want to keep these things
        # possible solution: save entire args dict (can we even get this without losing matching on known args?) and then
        # call init with this dict, suitably overriden
        # could use locals() but this seems like a huge hack and will probably break
        # I feel there's no proper way to do this since we have no idea what additional behaviour/data subclasses may define
        return FileAtom(name=name, orig_name=self.orig_path, output_dir=output_dir)

    def newname_with_suffix(self, suffix, ext=None, subdir=None):
        """Create a FileAtom representing <dirs>/<file><suffix><.ext>
        from one representing <dirs>/<file><.ext>

        >>> os.path.basename(FileAtom(name='img_1.mnc').newname_with_suffix(suffix='_fwhm0.056').path)
        'img_1_fwhm0.056.mnc'
        """
        return self.newname_with(lambda n: n + suffix, ext=ext, subdir=subdir)

    # TODO merge into __init__ making name optional and taking base and/or a copy/clone method ?? e.g.
    #def __init__(self, name=None, orig_atom=None, suffix=None, update_fn=None, orig_name=None):
    #   ...
    # or
    #def clone/copy(self, update_fn=None, suffix=None, ext=None, subdir=None):
    #   assert not (suffix and update_fn), "That can't be right!"
    #   ...
