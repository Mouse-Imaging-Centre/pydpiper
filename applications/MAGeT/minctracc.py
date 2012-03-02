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
        self.basevol = inputVolume
        self.labels = []
        self.inputLabels = None
        self.blurs = {}
        self.gradients = {}
        self.lastblur = None #This can be a gradient.
        self.transforms = {}
        self.lastTransform = {}
        self.mask = None
        
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
        self.basename = fh.removeBaseAndExtension(self.inputFileName)
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
        sourceFilename = fh.removeBaseAndExtension(self.getLastBasevol())
        targetFilename = fh.removeBaseAndExtension(targetFH.getLastBasevol())
        # check to make sure blurs match, otherwise throw error?
        # Go through argument list to build file name
        xfmFileName = [sourceFilename, "to", targetFilename]
        for k, l in arglist:
            # MF TODO: Think about how to handle gradient case
            if k == 'blur':
                xfmFileName += [str(l), "blur"]
            elif k == 'linearparam':
                xfmFileName += [l]
            elif k == 'iterations':
                xfmFileName += ["iter", str(l)]
        xfmFileWithExt = "_".join(xfmFileName) + ".xfm"
        outputXfm = fh.createBaseName(self.tmpDir, xfmFileWithExt)
        self.addAndSetXfmToUse(targetFilename, outputXfm)
        return(outputXfm)
    #MF TODO: This code is getting a bit repetitive. Lets see if we can't
    # consolidate a bit.     
    def getBlur(self, fwhm=None): 
        return(self.groupedFiles[self.currentGroupIndex].getBlur(fwhm))
    def setBlurToUse(self, fwhm):
        self.groupedFiles[self.currentGroupIndex].lastblur = fwhm
    def getLastBasevol(self):
        return(self.groupedFiles[self.currentGroupIndex].basevol)
    def getLastXfm(self, targetFilename):
        currGroup = self.groupedFiles[self.currentGroupIndex]
        lastXfm = None
        if targetFilename in currGroup.lastTransform:
            lastXfm = currGroup.lastTransform[targetFilename]
        return(lastXfm)
    def setLastXfm(self, targetFilename, xfm):
        self.groupedFiles[self.currentGroupIndex].lastTransform[targetFilename] = xfm
    def addAndSetXfmToUse(self, targetFilename, xfm):
        currGroup = self.groupedFiles[self.currentGroupIndex]
        if not targetFilename in currGroup.transforms:
            currGroup.transforms[targetFilename] = []
        if not currGroup.transforms[targetFilename].__contains__(xfm):
            currGroup.transforms[targetFilename].append(xfm)
        self.setLastXfm(targetFilename, xfm)
    def getInputLabels(self):
        return(self.groupedFiles[self.currentGroupIndex].inputLabels)
    def setInputLabels(self, labels):
        self.groupedFiles[self.currentGroupIndex].inputLabels = labels
    def addLabels(self, newLabel):
        """Add labels to array."""
        currGroup = self.groupedFiles[self.currentGroupIndex]
        if not currGroup.labels.__contains__(newLabel):
            currGroup.labels.append(newLabel)
    def returnLabels(self):
        return(self.groupedFiles[self.currentGroupIndex].labels)
    def setMask(self, inputMask):
        self.groupedFiles[self.currentGroupIndex].mask = inputMask
    def getMask(self):
        return(self.groupedFiles[self.currentGroupIndex].mask)
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

def isFileHandler(inSource, inTarget=None):
    """Source and target types can be either RegistrationPipeFH or strings
    Regardless of which is chosen, they must both be the same type.
    If this function returns True - both types are fileHandlers. If it returns
    false, both types are strings. If there is a mismatch, the assert statement
    should cause an error to be raised."""
    isFileHandlingClass = True
    assertMsg = 'source and target files must both be same type: RegistrationPipeFH or string'
    if isinstance(inSource, RegistrationPipeFH):
        if inTarget:
            assert isinstance(inTarget, RegistrationPipeFH), assertMsg
    else:
        if inTarget:
            assert not isinstance(inTarget, RegistrationPipeFH), assertMsg
        isFileHandlingClass = False
    return(isFileHandlingClass)

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
        try: 
            if isFileHandler(inSource, inTarget):
                # if blur = None, getBlur returns lastblur 
                self.source = inSource.getBlur(blur)
                self.target = inTarget.getBlur(blur)
                if not transform:
                    # Note: this may also be None and should be for initial call
                    targetFilename = fh.removeBaseAndExtension(inTarget.getLastBasevol())
                self.transform = inSource.getLastXfm(targetFilename)
                frame = inspect.currentframe()
                args,_,_,arglocals = inspect.getargvalues(frame)
                arglist = [(i, arglocals[i]) for i in args]
                outputXfm = inSource.registerVolume(inTarget, arglist)
                self.output = outputXfm
                self.logFile = inSource.logFromFile(outputXfm)
                self.source_mask = inSource.getMask()
                self.target_mask = inTarget.getMask()
            else:
                self.source = inSource
                self.target = inTarget
                self.output = output
                self.logFile = logFile
                self.transform = transform
                self.source_mask = source_mask
                self.target_mask = target_mask 
        except:
            print "Failed in putting together minctracc command."
            print "Unexpected error: ", sys.exc_info()
        
        self.linearparam = linearparam       
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
        #MF TODO: In both instances, better handle gradient option
        try:
            if isFileHandler(inFile):
                blurlist = inFile.blurFile(fwhm, gradient)
                self.base = blurlist["base"]
                self.inputFiles = [inFile.lastBaseVol]
                self.outputFiles = [blurlist["file"]]
                self.logFile = blurlist["log"]
                self.name = "mincblur " + str(fwhm) + " " + inFile.basename
                if gradient:
                    self.outputFiles.append(blurlist["gradient"])
            else:
                print "inFile is string"
                self.base = str(inFile).replace(".mnc", "")
                self.inputFiles = [inFile]
                blurBase = "".join([self.base, "_fwhm", str(fwhm), "_blur"])
                self.outputFiles = ["".join([blurBase, ".mnc"])]
                self.logFile = fh.createLogDirandFile(abspath(os.curdir), blurBase)
                self.name = "mincblur " + str(fwhm) + " " + basename(inFile)
                if gradient:
                    gradientBase = blurBase.replace("blur", "dxyz")
                    self.outputFiles += ["".join([gradientBase, ".mnc"])] 
        except:
            print "Failed in putting together blur command."
            print "Unexpected error: ", sys.exc_info()
            
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
        if not argArray:
            argArray = ["mincresample"] 
        else:      
            argArray.insert(0, "mincresample")
        CmdStage.__init__(self, argArray)
            
        #If no like file is specified, use inputFile
        if not likeFile:
            likeFile=inFile

        try:
            #MF TODO: What if we don't want to use lastBasevol?  
            if isFileHandler(inFile, likeFile):
                self.inFile = self.getFileToResample(inFile)
                self.likeFile = likeFile.getLastBasevol()
                self.cxfm = inFile.getLastXfm(fh.removeBaseAndExtension(self.likeFile))
                self.outfile = self.setOutputFile(likeFile)
                self.logFile = inFile.logFromFile(self.outfile)
            else:
                self.inFile = inFile
                self.likeFile = likeFile
                self.logFile=logFile
                self.outfile=outFile
                self.cxfm = cxfm
        except:
            print "Failed in putting together resample command"
            print "Unexpected error: ", sys.exc_info()
            
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
    def setOutputFile(self, likeFile):
        outBase = fh.removeBaseAndExtension(self.cxfm) + "-resampled.mnc"
        return(fh.createBaseName(likeFile.tmpDir, outBase))  
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
        outBase = fh.removeBaseAndExtension(self.cxfm) + "-resampled-labels.mnc"
        labelFile = fh.createBaseName(likeFile.tmpDir, outBase)  
        # If the likeFile has no InputLabels, use these labels. 
        # Otherwise, add to labels array
        if not likeFile.getInputLabels():
            likeFile.setInputLabels(labelFile)
        else:
            likeFile.addLabels(labelFile)
        return(labelFile)  
    def getFileToResample(self, inputFile):
        return(inputFile.getInputLabels())     