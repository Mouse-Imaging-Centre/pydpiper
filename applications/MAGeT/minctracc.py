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
        self.lastLabels = None
        self.blurs = {}
        self.gradients = {}
        self.lastblur = None #This can be a gradient.
        self.transforms = []
        self.lastTransform = None
        
    def getBlur(self, fwhm=None):
        """returns file with specified blurring kernel
        If no blurring kernel is specified, return the last blur"""
        if not fwhm:
            fwhm = self.lastblur
        return(self.blurs[fwhm])

    def addBlur(self, filename, fwhm, gradient=None):
        """adds the blur with the specified kernel"""
        self.blurs[fwhm] = filename
        self.lastblur = fwhm
        if gradient:
            self.gradients[fwhm] = gradient
    
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
        self.currentGroupIndex = -1 #MF why -1 instead of 0? TEST
        #MF TODO: verify below with Jason to verify correct interpretation
        self.inputFileName = filename
        # Check to make sure that basedir exists, otherwise create:
        self.basedir = fh.makedirsIgnoreExisting(basedir)
        # groups can be referred to by either name or index number
        self.groupNames = {'base' : 0}       
        # create directories
        self.setupNames()
        # set lastBaseVol
        self.lastBaseVol = filename
    
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
        """string munging to create necessary basenames and directories"""
        self.basename = fh.removeFileExt(self.inputFileName)
        self.subjDir = fh.createSubDir(self.basedir, self.basename)
        self.logDir = fh.createLogDir(self.subjDir)
        self.resampledDir = fh.createSubDir(self.subjDir, "resampled")
        self.transformsDir = fh.createSubDir(self.subjDir, "transforms")
        self.labelsDir = fh.createSubDir(self.subjDir, "labels")
        self.tmpDir = fh.createSubDir(self.subjDir, "tmp")
        
    def logFromFile(self, inFile):
        """ creates a log file from an input filename

        Takes the input file, strips out any extensions, and returns a 
        filename with the same basename, in the log directory, and 
        with a .log extension"""

        #MF TODO: Can we move to fileHandling class?  
        logBase = fh.removeBaseAndExtension(inFile)
        log = fh.createLogFile(self.logDir, logBase)
        return(log)

    def registerVolume(self, targetFH, arglist):
        """create the filenames for a single registration call

        Two input arguments - a RegistrationPipeFH instance for the
        target volume and an argument list; the argument list is
        derived from function introspection in the minctracc or
        mincANTS calls and will be used to provide a unique filename.

        """
        # Could arglist contain blur information?
        sourceFilename = self.basename
        targetFilename = targetFH.basename
        # check to make sure blurs match, otherwise throw error?
        # Go through argument list to build file name
        xfmFileName = [sourceFilename, "to", targetFilename]
        for i, j in enumerate(arglist):
            k, l = j
            # MF TODO: Think about how to handle gradient case
            if k == 'blur':
                xfmFileName += [str(l), "blur"]
            elif k == 'linearparam':
                xfmFileName += [l]
            elif k == 'iterations':
                xfmFileName += ["iter", str(l)]
        xfmFileWithExt = "_".join(xfmFileName) + ".xfm"
        return(fh.createBaseName(self.tmpDir, xfmFileWithExt))
    #MF TODO: This code is getting a bit repetitive. Lets see if we can't
    # consolidate a bit.     
    def getBlur(self, fwhm=None): 
        return(self.groupedFiles[self.currentGroupIndex].getBlur(fwhm))
    def setBlurToUse(self, fwhm):
        self.groupedFiles[self.currentGroupIndex].lastblur = fwhm
    #MF TODO: Fix the function below?
    def getLastBasevol(self):
        return(self.groupedFiles[self.currentGroupIndex].basevol[-1])
    def getLastXfm(self):
        return(self.groupedFiles[self.currentGroupIndex].lastTransform)
    def addAndSetXfmToUse(self, xfm):
        currGroup = self.groupedFiles[self.currentGroupIndex]
        if not currGroup.transforms.__contains__(xfm):
            currGroup.transforms.append(xfm)
        currGroup.lastTransform = xfm
    def getLastLabels(self):
        return(self.groupedFiles[self.currentGroupIndex].lastLabels)
    def addAndSetLabelsToUse(self, labels):
        currGroup = self.groupedFiles[self.currentGroupIndex]
        if not currGroup.labels.__contains__(labels):
            currGroup.labels.append(labels)
        currGroup.lastLabels = labels
    def blurFile(self, fwhm, gradient=False):
        """create filename for a mincblur call

        Return a triplet of the basename, which mincblur needs as its
        input, the full filename, which mincblur will create after its done,
        and the log file"""
        
        #MF TODO: Error handling if there is no lastBaseVol
        lastBaseVol = self.getLastBasevol()
        outputbase = fh.removeBaseAndExtension(lastBaseVol)
        outputbase = "%s/%s_fwhm%g" % (self.tmpDir, outputbase, fwhm)
        
        withext = "%s_blur.mnc" % outputbase     
        log = self.logFromFile(withext)

        outlist = { "base" : outputbase,
                    "file" : withext,
                    "log"  : log }
        
        if gradient:
            gradWithExt = "%s_dxyz.mnc", outputbase
            outlist["gradient"] = gradWithExt
        else:
            gradWithExt=None

        self.groupedFiles[self.currentGroupIndex].addBlur(withext, fwhm, gradWithExt)
        return(outlist)
        
class minctracc(CmdStage):
    def __init__(self, 
                 inSource,
                 inTarget,
                 output=None,
                 logFile=None,
                 blur=None,
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
            # if blur = None, getBlur returns lastblur 
            self.source = inSource.getBlur(blur)
            self.target = inTarget.getBlur(blur)
            if not transform:
                # Note: this may also be None and should be for initial call
                self.transform = inSource.getLastXfm()
            # get the arguments passed to this function - these will be used
            # by the RegistrationPipeFH to build an output filename
            frame = inspect.currentframe()
            args,_,_,arglocals = inspect.getargvalues(frame)
            arglist = [(i, arglocals[i]) for i in args]
            outputXfm = inSource.registerVolume(inTarget, arglist)
            self.output = outputXfm
            self.logFile = inSource.logFromFile(outputXfm)
            # Set new xfm as input for next time
            inSource.addAndSetXfmToUse(outputXfm)
        except AttributeError: # go with filename inputs
            self.source = inSource
            self.target = inTarget
            self.output = output
            self.logFile = logFile
            self.transform = transform
        
        self.linearparam = linearparam
        self.source_mask = source_mask
        self.target_mask = target_mask
        self.iterations = str(iterations)
        self.lattice_diameter = str(step*3)
        self.step = str(step)       
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
    def __init__(self, 
                 inSource, 
                 inTarget, 
                 output=None, 
                 logFile=None, 
                 blur=None,
                 linearparam="lsq12", 
                 source_mask=None, 
                 target_mask=None):
        minctracc.__init__(self,
                           inSource,
                           inTarget,
                           output, 
                           logFile,
                           blur, 
                           linearparam,
                           source_mask=source_mask,
                           target_mask=target_mask)

    def finalizeCommand(self):
        """add the options for a linear fit"""
        _numCmd = "-" + self.linearparam
        self.cmd += ["-xcorr", _numCmd]
    def setName(self):
        self.name = "minctracc" + self.linearparam + " "

class blur(CmdStage):
    def __init__(self, inFile, fwhm, gradient=False):
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
            blurlist = inFile.blurFile(fwhm, gradient)
            self.base = blurlist["base"]
            self.inputFiles = [inFile.lastBaseVol]
            self.outputFiles = [blurlist["file"]]
            self.logFile = blurlist["log"]
            self.name = "mincblur " + str(fwhm) + " " + inFile.basename
            if gradient:
                self.outputFiles.append(blurlist["gradient"])
        except AttributeError:
            #print "Unexpected error: ", sys.exc_info()
            #print "In blur attribute error except"
            # this means it wasn't a filehandler - now assume it's a file
            self.base = str(inFile).replace(".mnc", "")
            self.inputFiles = [inFile]
            blurBase = "".join([self.base, "_fwhm", str(fwhm), "_blur"])
            self.outputFiles = ["".join([blurBase, ".mnc"])]
            self.logFile = fh.createLogDirandFile(abspath(os.curdir), blurBase)
            self.name = "mincblur " + str(fwhm) + " " + basename(inFile)
            if gradient:
                gradientBase = blurBase.replace("blur", "dxyz")
                self.outputFiles += ["".join([gradientBase, ".mnc"])]
            # Check for log directory and create logfile
            
        self.cmd = ["mincblur", "-clobber", "-fwhm", str(fwhm),
                    self.inputFiles[0], self.base]
        if gradient:
            self.cmd += ["-gradient"]       
        self.colour="blue"

class mincresample(CmdStage):
    def __init__(self, 
                 inFile, 
                 outFile=None, 
                 logFile=None, 
                 likeFile=None, 
                 cxfm=None, 
                 argArray=None):
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
        if argArray:
            argArray.insert(0, "mincresample")
            CmdStage.__init__(self, argArray)
        else:
            CmdStage.__init__(self, None)
            self.cmd += ["mincresample"]
        
        #If no like file is specified, use inputFile
        if not likeFile:
            likeFile=inFile
        
        # first try to generate the filenames if inFile was a filehandler
        try:
            #MF TODO: What if we don't want to use lastBasevol?  
            self.inFile = self.getFileToResample(inFile)
            self.likeFile = likeFile.getLastBasevol()
            self.cxfm = inFile.getLastXfm()
            self.outfile = self.setOutputFile(likeFile)
            self.logFile = inFile.logFromFile(self.outfile) 
        except AttributeError:
            print "In resample attribute error except"
            print "Unexpected error: ", sys.exc_info()
            # this means it wasn't a filehandler - now assume it's a file
            self.inFile = inFile
            self.likeFile = likeFile
            self.logFile=logFile
            self.outfile=outFile
            self.cxfm = cxfm
        self.addDefaults()
        self.finalizeCommand()
        self.setName()
        
    def addDefaults(self):
        self.inputFiles += [self.inFile]   
        self.outputFiles += [self.outfile]       
        self.cmd += ["-like", self.likeFile] 
        if not self.inputFiles.__contains__(self.likeFile): 
                self.inputFiles += [self.likeFile]
        if self.cxfm:
            self.inputFiles += [self.cxfm]
            self.cmd += ["-transform", self.cxfm]              
    def finalizeCommand(self):
        """Add -2, clobber, input and output files """
        self.cmd += ["-2", "-clobber", self.inFile, self.outfile]      
    def setName(self):
        self.name = "mincresample " 
    def setOutputFile(self, likeFile=None):
        return(fh.removeFileExt(self.cxfm) + "-resampled.mnc")  
    def getFileToResample(self, inputFile):
        return(inputFile.getLastBasevol())  

class mincresampleLabels(mincresample):
    def __init__(self, 
                 inFile, 
                 outFile=None, 
                 logFile=None, 
                 likeFile=None, 
                 cxfm=None, 
                 argArray=None):
        mincresample.__init__(self,
                           inFile, 
                           outFile, 
                           logFile, 
                           likeFile, 
                           cxfm, 
                           argArray)
    def finalizeCommand(self):
        """additional arguments needed for resampling labels"""
        self.cmd += ["-keep_real_range", "-nearest_neighbour"]
        mincresample.finalizeCommand(self)
    def setOutputFile(self, likeFile):
        """set name of output and add labels to likeFile labels array"""
        labelFile = fh.removeFileExt(self.cxfm) + "-resampled-labels.mnc"
        likeFile.addAndSetLabelsToUse(labelFile)
        return(labelFile)  
    def getFileToResample(self, inputFile):
        return(inputFile.getLastLabels())     