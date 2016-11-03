import copy
import os

from typing import Callable, Union, Tuple


def explode(filename: str) -> Tuple[str, str, str]:
    # TODO should this be a namedtuple instead of a tuple? Probably...can still match against, etc.
    """Split `filename` into a directory, 'base', and extension.
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


class NotProvided(object):
    """To be used as a datatype indicating no argument with this name was supplied
    in the situation when None has another sensible meaning, e.g., when a sensible
    default file is available but None indicates *no* file should be used."""


class FileAtom(object):
    """
    What is stored:
        self.orig_path -- original input file, full file name and path (/path/filename.ext), can be None
        self.dir
        self.filename_wo_ext
        self.ext
    The last two fields (and their corresponding constructor parameters) are used only when creating derived
    files from this file via its newname_* methods:
        self.pipeline_sub_dir - a path to a directory to act as a sort of root directory for deriving new
                            files from this one.  The file need not be inside its pipeline_sub_dir.
                            If not provided, the current working directory is used.  (MICe pipeline examples:
                            the _lsq6, _lsq12, _nlin, and _processed directories.)
        self.output_sub_dir - a relative path to a directory in which to create new files derived from the file
                            (inside of the pipeline_sub_dir).  The value of this field is inherited
                            unchanged by the new file.  If `output_sub_dir` is not provided when constructing
                            a FileAtom, its filename_wo_ext is used; for instance,
                            if the filename is "relative/img_1.mnc", the output_sub_dir becomes "img_1/".
    """
    # TODO this documentation should still be more clear and explain why you'd want to use these features/fields
    
    def __init__(self,
                 name             : str,
                 orig_name        : Union[str, None, NotProvided] = NotProvided(),
                 pipeline_sub_dir : str = None,
                 output_sub_dir   : str = None) -> None:
        self.dir, self.filename_wo_ext, self.ext = explode(name)
        if isinstance(orig_name, NotProvided):
            self.orig_path = name  # type: str
        elif isinstance(orig_name, type(None)):
            # is this case even needed? if no orig file, when _isn't_ file itself the orig file?
            self.orig_path = None
        else:
            self.orig_path = orig_name
            
        if pipeline_sub_dir:
            self.pipeline_sub_dir = pipeline_sub_dir
        else:
            self.pipeline_sub_dir = ''

        if output_sub_dir:
            self.output_sub_dir = output_sub_dir
        else:
            self.output_sub_dir = self.filename_wo_ext

    @property
    def path(self) -> str:
        #return self.get_path()
        return os.path.join(self.dir, self.filename_wo_ext + self.ext)

    #def get_path(self) -> str:
    #    return os.path.join(self.dir, self.filename_wo_ext + self.ext)

    # TODO: are these the most reasonable definitions of __eq__, __cmp__, and __hash__?
    def __eq__(self, other) -> bool:
        return (self is other or
                (self.__class__ == other.__class__
                 and self.path == other.path))

    def __hash__(self) -> int:
        return self.path.__hash__()

    def __lt__(self, other) -> bool:
        return self.path < other.path
    
    def __repr__(self) -> str:
        return "%s(path=%s, ...)" % (self.__class__, self.path)

    # path = property(get_path, "`path` property") # type: ignore
    
    def get_basename(self) -> str:
        return self.filename_wo_ext + self.ext

    def newname_with_fn(self,
                        fn     : Callable[[str], str],
                        ext    : str = None,
                        subdir : str = None) -> 'FileAtom':
        """Create a new FileAtom from one which has:
        <filename_wo_ext>.<ext>
        to: 
        <self.pipeline_sub_dir>/<self.output_sub_dir>/[<subdir>/]f(<filename_wo_ext>)<.ext>
       
        # (we use absolute paths so the tests are deterministic without requiring postprocessing)
        >>> f1 = FileAtom(name='/project/images/img_1.mnc', pipeline_sub_dir="/scratch/pipeline/")
        >>> f2 = f1.newname_with_fn(fn=lambda n: n + '_fwhm0.056')
        >>> f2.path
        '/scratch/pipeline/img_1/img_1_fwhm0.056.mnc'
        >>> f1.path
        '/project/images/img_1.mnc'
        >>> f3 = FileAtom(name='/project/images/img_3.mnc', pipeline_sub_dir="/scratch/pipeline/")
        >>> f4 = f3.newname_with_fn(fn=lambda n: n + '_inormalize', subdir="tmp")
        >>> f4.path
        '/scratch/pipeline/img_3/tmp/img_3_inormalize.mnc'
        >>> f3.path
        '/project/images/img_3.mnc'
        """

        new_dir = os.path.join(self.pipeline_sub_dir, 
                               self.output_sub_dir if self.output_sub_dir else "",
                               subdir if subdir else "")
        filename_wo_ext = fn(self.filename_wo_ext)
        fa = copy.copy(self)
        fa.dir = new_dir
        fa.ext = ext or self.ext
        fa.filename_wo_ext = filename_wo_ext
        return fa

    def newname_with_suffix(self,
                            suffix : str,
                            ext    : str = None,
                            subdir : str = None) -> 'FileAtom':
        """Create a FileAtom representing <dirs>/<file><suffix><.ext>
        from one representing <dirs>/<file><.ext>

        >>> os.path.basename(FileAtom(name='img_1.mnc').newname_with_suffix(suffix='_fwhm0.056').get_basename())
        'img_1_fwhm0.056.mnc'
        """
        return self.newname_with_fn(lambda n: n + suffix, ext=ext, subdir=subdir)

    def newname(self,
                name   : str,
                ext    : str = None,
                subdir : str = None) -> 'FileAtom':
        """Create a new FileAtom from an old one, ignoring the existing name (but possibly recycling the extension).

        >>> os.path.basename(FileAtom(name='img_1.mnc').newname(name='img_2').get_basename())
        'img_2.mnc'
        """
        return self.newname_with_fn(lambda _: name, ext=ext, subdir=subdir)
     
