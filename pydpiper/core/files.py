import copy
import os

from .util import explode, NotProvided

class FileAtom(object):
    """
        What is stored:
        self.orig_path -- original input file, full file name and path (/path/filename.ext), can be None
        self.dir
        self.filename_wo_ext
        self.ext
        
        self.pipeline_sub_dir - in pipeline terms, this would be the _lsq6, _lsq12, _nlin or
                            _processed directory. If not provided, the current working directory
                            is used.
        self.output_sub_dir - in terms of the old code, this is the sub directory that is the 
                            main output directory for this file. For instance for an input file
                            called img_1.mnc, the value for this would/could be "img_1". If the 
                            output_sub_dir is not provided, the self.filename_wo_ext is used. 
                          
                          
    """
    def __init__(self, name, orig_name=NotProvided, pipeline_sub_dir=None, output_sub_dir=None):
        self.dir, self.filename_wo_ext, self.ext = explode(name)
        if orig_name is NotProvided:
            self.orig_path = self.get_path()
        elif orig_name is None:
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
            
    def __eq__(self, other):
        return (self is other or
                (self.get_path() == other.get_path()
                 and self.__class__ == other.__class__))
    
    def __repr__(self):
        return "%s(path=%s, ...)" % (self.__class__,self.get_path()) 

    def get_path(self): 
        return os.path.join(self.dir, self.filename_wo_ext + self.ext)

    path = property(get_path, "`path` property")
    
    def get_basename(self):
        return self.filename_wo_ext + self.ext

    def newname_with_fn(self, fn, ext=None, subdir=None):
        """Create a new FileAtom from one which has:
        <filename_wo_ext>.<ext>
        to: 
        <self.pipeline_sub_dir>/<self.output_sub_dir>/[<subdir>/]f(<filename_wo_ext>)<.ext>
       
        # (we use absolute paths so the tests are deterministic without requiring postprocessing)
        >>> f1 = FileAtom(name='/project/images/img_1.mnc', pipeline_sub_dir="/scratch/pipeline/")
        >>> f2 = f1.newname_with_fn(fn=lambda n: n + '_fwhm0.056')
        >>> f2.get_path()
        '/scratch/pipeline/img_1/img_1_fwhm0.056.mnc'
        >>> f1.get_path()
        '/project/images/img_1.mnc'
        >>> f3 = FileAtom(name='/project/images/img_3.mnc', pipeline_sub_dir="/scratch/pipeline/")
        >>> f4 = f3.newname_with_fn(fn=lambda n: n + '_inormalize', subdir="tmp")
        >>> f4.get_path()
        '/scratch/pipeline/img_3/tmp/img_3_inormalize.mnc'
        >>> f3.get_path()
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

    def newname_with_suffix(self, suffix, ext=None, subdir=None):
        """Create a FileAtom representing <dirs>/<file><suffix><.ext>
        from one representing <dirs>/<file><.ext>

        >>> os.path.basename(FileAtom(name='img_1.mnc').newname_with_suffix(suffix='_fwhm0.056').get_basename())
        'img_1_fwhm0.056.mnc'
        """
        return self.newname_with_fn(lambda n: n + suffix, ext=ext, subdir=subdir)
     
