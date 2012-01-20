#!/usr/bin/env python

from pydpiper.pipeline import * 
from os.path import abspath, basename
import pydpiper.file_handling as fh
import networkx as nx
import inspect

Pyro.config.PYRO_MOBILE_CODE=1

class RegistrationGroupedFiles():
    """A class to keep together all bits for a RegistrationPipeFH stage"""
    def __init__(self, inputVolume):
        self.basevol = [inputVolume]
        self.labels = []
        self.blurs = {}
        self.lastblur = None
        self.transforms = []
        
    def getBlur(self, fwhm=None):
        """returns file with specified blurring kernel
        If no blurring kernel is specified, return the last blur"""
        if not fwhm:
            fwhm = self.lastblur
        return(self.blurs[fwhm])

    def addBlur(self, filename, fwhm):
        """adds the blur with the specified kernel"""
        self.blurs[fwhm] = filename
        self.lastblur = fwhm
            
class RegistrationPipeFH():
    """A class to provide file-handling support for registration pipelines

    Each input file will have a separate directory underneath the
    specified base directory. This will in turn be populated by
    different output directories for transforms, resampled files,
    temporary files, etc. The final directory tree will look like the
    following:

    basedir/filenamebase/log -- log files
    basedir/filenamebase/resampled -- resampled files go here
    basedir/filenamebase/transforms -- transforms (xfms, grids, etc.) go here
    basedir/filenamebase/labels -- any resampled labels (if necessary) go here
    basedir/filenamebase/tmp -- intermediate temporary files go here

    The RegistrationPipeFH can be passed to different processing
    functions (minctracc, blur, etc.) which will use it to derive
    proper filenames. The RegistrationPipeFH can moreover group
    related files (blurs, transforms, resamples) by using the newGroup
    call.

    """
    def __init__(self, filename, basedir):
        """two inputs required - an inputFile file and a base directory."""
        self.groupedFiles = [RegistrationGroupedFiles(filename)]
        self.currentGroupIndex = -1 #MF why -1 instead of 0?
        #MF TODO: verify below with Jason to verify correct interpretation
        self.inputFileName = filename
        # Check to make sure that basedir exists, otherwise create:
        self.basedir = fh.makedirsIgnoreExisting(basedir)
        # groups can be referred to by either name or index number
        self.groupNames = {'base' : 0}       
        # create directories
        self.setupNames()
    
    def newGroup(self, inputVolume = None, groupName = None):
        """create a new set of grouped files"""
        groupIndex = len(self.groupedFiles) + 1
        if not inputVolume:
            inputVolume = self.getLastBasevol()
        
        if not groupName:
            groupName = groupIndex

        self.groupedFiles.append(RegistrationGroupedFiles(inputVolume))
        self.groupNames[groupName] = groupIndex
        self.currentGroupIndex = groupIndex

    def setupNames(self):
        """string munging to create necessary basenames and such"""
        self.basename = self.FH.removeFileExt(self.inputFilename)
        self.subjDir = self.FH.createSubDir(self.basedir, self.basename)
        self.logDir = self.FH.createLogDir(self.subjDir)
        self.resampledDir = self.FH.createSubDir(self.subjDir, "resampled")
        self.transformsDir = self.FH.createSubDir(self.subjDir, "transforms")
        self.labelsDir = self.FH.createSubDir(self.subjDir, "labels")
        self.tmpDir = self.FH.createSubDir(self.subjDir, "tmp")
        
    def logFromFile(self, inFile):
        """ creates a log file from an input filename

        Takes the input file, strips out any extensions, and returns a 
        filename with the same basename, in the log directory, and 
        with a .log extension"""

        #MF TODO: Can we move to fileHandling class?  
        logBase = self.FH.removeBaseAndExtension(inFile)
        log = self.FH.createLogFile(self.logDir, logBase)
        return(log)

    def registerVolume(self, targetFH, arglist):
        """create the filenames for a single registration call

        Two input arguments - a RegistrationPipeFH instance for the
        target volume and an argument list; the argument list is
        derived from function introspection in the minctracc or
        mincANTS calls and will be used to provide a unique filename.

        """
        sourceFilename = self.getBlur()
        
        
    def getBlur(self):
        return(self.groupedFiles[self.currentGroupIndex].getBlur())
    def setBlurToUse(self, fwhm):
        self.groupedFiles[self.currentGroupIndex].lastblur = fwhm
    def getLastBasevol(self):
        return(self.groupedFiles[self.currentGroupIndex].basevol[-1])

    def blurFile(self, fwhm, gradient=False):
        """create filename for a mincblur call

        Return a triplet of the basename, which mincblur needs as its
        input, the full filename, which mincblur will create after its done,
        and the log file"""
        
        #MF TODO: Handle if there is no lastBaseVol --> what to do?
        lastBaseVol = self.getLastBasevol()

        outputbase = self.FH.removeBaseAndExtension(lastBaseVol)
        outputbase = "%s/%s_fwhm%g" % (self.tmpDir, outputbase, fwhm)
        #MF TODO: What if we want both gradient and regular. Do we need to
        # call this twice?
        if gradient:
            withext = "%s_dxyz.mnc", outputbase
        else:
            withext = "%s_blur.mnc", outputbase
        
        log = self.logFromFile(withext)

        outlist = { "base" : outputbase,
                    "file" : withext,
                    "log"  : log }

        self.groupedFiles[self.currentGroupIndex].addBlur(withext, fwhm)
        return(outlist)
        
class minctracc(CmdStage):
    def __init__(self, 
                 inSource,
                 inTarget,
                 output=None,
                 logFile=None,
                 linearparam="nlin",
                 source_mask=None, 
                 target_mask=None,
                 iterations=40,
                 step=0.5,
                 transform=None,
                 weight=0.8,
                 stiffness=0.98,
                 similarity=0.3,
                 w_translations=0.2):
        """an efficient way to add a minctracc call to a pipeline

        The constructor needs two inputFile arguments, the source and the
        target for the registration, and multiple optional arguments
        for specifying parameters. The source and the target can be
        specified as either RegistrationPipeFH instances or as strings
        representing filenames. In the latter case an output and a
        logfile filename are required as well (these are filled in
        automatically in the case of RegistrationPipeFH instances.)

        """
        CmdStage.__init__(self, None) #don't do any arg processing in superclass
        try: # try with inputs as RegistrationPipeFH instances
            # MF TODO: get lastBlur from currentGroupIndex? 
            self.source = inSource.lastBlur
            self.target = inTarget.lastBlur
            # get the arguments passed to this function - that will be used
            # by the RegistrationPipeFH to build an output filename
            frame = inspect.currentframe()
            args,_,_,arglocals = inspect.getargvalues(frame)
            arglist = [(i, arglocals[i]) for i in args]
            targetList = inSource.registerVolume(inTarget, arglist)
            self.output = "tmp.xfm" # output will be based on targetList
            self.logFile = "tmp.log"
        except AttributeError: # go with filename inputs
            self.source = inSource
            self.target = inTarget
            # MF TODO: use registerVolume here/make general? 
            self.output = output
            self.logFile = logFile
        
        self.linearparam = linearparam
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.iterations = str(iterations)
        self.lattice_diameter = str(step*3)
        self.step = str(step)
        self.transform = transform
        self.weight = str(weight)
        self.stiffness = str(stiffness)
        self.similarity = str(similarity)
        self.w_translations = str(w_translations)

        self.addDefaults()
        self.finalizeCommand()
        self.setName()
        self.colour = "red"

    def setName(self):
        self.name = "minctracc nlin step: " + self.step 
    def addDefaults(self):
        self.cmd = ["minctracc",
                    "-clobber",
                    "-similarity", self.similarity,
                    "-weight", self.weight,
                    "-stiffness", self.stiffness,
                    "-w_translations", self.w_translations,self.w_translations,self.w_translations,
                    "-step", self.step, self.step, self.step,

                    self.source,
                    self.target,
                    self.output]
        
        # adding inputs and outputs
        # adding inputs and outputs
        self.inputFiles = [self.source, self.target]
        if self.source_mask:
            self.inputFiles += [self.source_mask]
            self.cmd += ["-source_mask", self.source_mask]
        if self.target_mask:
            self.inputFiles += [self.target_mask]
            self.cmd += ["-model_mask", self.target_mask]
        if self.transform:
            self.inputFiles += [self.transform]
            self.cmd += ["-transform", self.transform]
        self.outputFiles = [self.output]

    def finalizeCommand(self):
        """add the options for non-linear registration"""
        # build the command itself
        self.cmd += ["-iterations", self.iterations,
                    "-nonlinear", "corrcoeff", "-sub_lattice", "6",
                    "-lattice_diameter", self.lattice_diameter,
                     self.lattice_diameter, self.lattice_diameter]

class linearminctracc(minctracc):
    def __init__(self, source, target, output=None, logFile=None, 
                 linearparam, source_mask=None, target_mask=None):
        minctracc.__init__(self,source,target,output, 
                           logFile,linearparam,
                           source_mask=source_mask,
                           target_mask=target_mask)

    def finalizeCommand(self):
        """add the options for a linear fit"""
        _numCmd = "-" + self.linearparam
        self.cmd += ["-xcorr", _numCmd]
    def setName(self):
        self.name = "minctracc" + self.linearparam + " "

class blur(CmdStage):
    def __init__(self, inFile, fwhm):
        """calls mincblur with the specified 3D Gaussian kernel

        The inputs can be in one of two styles. The first argument can
        be an instance of RegistrationPipeFH, in which case the last
        volume in that instance (i.e. inFile.lastBasevol) will be
        blurred and the output will be determined by its blurFile
        method. Alternately, the inFile can be a string representing a
        filename, in which case the output and logfile will be set based on 
        the inFile name.

        """
        CmdStage.__init__(self, None)

        # first try to generate the filenames if inputFile was a filehandler
        #MF TODO: In both instances, better handle gradient option
        try:
            blurlist = inFile.blurFile(fwhm)
            self.base = blurlist["outputbase"]
            self.inputFiles = [inFile.lastBaseVol]
            self.outputFiles = [blurlist["withext"]]
            self.logFile = blurlist["log"]
        except AttributeError:
            # this means it wasn't a filehandler - now assume it's a file
            self.base = inFile.replace(".mnc", "")
            self.inputFiles = [inFile]
            blurBase = "".join([self.base, "_fwhm", str(fwhm), "_blur"])
            self.outputFiles = ["".join([blurBase, ".mnc"])]
            # Check for log directory and create logfile
            self.logFile = fh.createLogDirandFile(abspath(os.curdir), blurBase)

        self.cmd = ["mincblur", "-clobber", "-fwhm", str(fwhm),
                    inFile, self.base]
        self.name = "mincblur " + str(fwhm) + " " + basename(inFile)
        self.colour="blue"

class mincresample(CmdStage):
    def __init__(self, inFile, outFile=None, logFile=None, 
                 argarray=[], likeFile=None, cxfm=None):
        """calls mincresample with the specified options

        The inFile and likeFile can be in one of two styles. 
        The first argument can be an instance of RegistrationPipeFH. 
        In this case the last volume in that instance (i.e. inFile.lastBasevol) 
        will be resampled and the output will be determined accordingly.
        Alternatively, the inFile can be a string representing a
        filename, in which case the output and logfile will be set based on 
        the inFile name.

        inFile is required, everything else optional
        This class assuming use of the most commonly used flags (-2, -clobber, -like, -transform)
        Any commands above and beyond the standard will be read in from argarray
        argarray could contain inFile and/or output files

        """
        #MF TODO: What if we don't want to use lastBasevol? 
        #MF TODO: Do we need to parse argarray for input or output files? 
        if argarray:
            argarray.insert(0, "mincresample")
        else:
            argarray = ["mincresample"]
        CmdStage.__init__(self, argarray)
        # first try to generate the filenames if inFile was a filehandler
        try:
            inputFile = inFile.getLastBasevol()
            lFile = likeFile.getLastBasevol()
            outDir = inFile.tmpDir
            logDir = inFile.logDir
            logAndOutBase = str(fh.removeBaseAndExtension(inputFile)) + "-resample.mnc"
            self.logFile = fh.createLogFile(logDir, logAndOutBase) 
            outfile= "%s/%s" % (outDir, logAndOutBase) 
        except AttributeError:
            # this means it wasn't a filehandler - now assume it's a file
            inputFile = inFile
            lFile = likeFile
            #MF TODO: check for output and log directories with name of file?
            outfile=outFile
            self.logFile=logFile
          
        self.inputFiles += [inputFile]   
        self.outputFiles += [outfile]
        
        if likeFile:
            self.cmd += ["-like", lFile] # include as an input file?
        if cxfm:
            self.inputFiles += [cxfm]
            self.cmd += ["-transform", cxfm]       
        self.cmd += ["-2", "-clobber", inputFile, outfile]
