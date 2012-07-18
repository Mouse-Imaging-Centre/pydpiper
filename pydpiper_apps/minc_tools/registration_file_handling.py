#!/usr/bin/env python

import pydpiper.file_handling as fh
from os.path import abspath, join
from os import curdir

class RegistrationGroupedFiles():
    """A class to keep together all bits for a RegistrationPipeFH stage"""
    def __init__(self, inputVolume):
        self.basevol = inputVolume
        self.labels = []
        self.inputLabels = []
        self.blurs = {}
        self.gradients = {}
        self.lastblur = None 
        self.lastgradient = None
        self.transforms = {}
        self.lastTransform = {}
        self.mask = None
        
    def getBlur(self, fwhm=None, gradient=False):
        """returns file with specified blurring kernel
        If no blurring kernel is specified, return the last blur
        If gradient is specified, return gradient instead of blur
        If fwhm = -1, return basevol"""
        blurToReturn = None
        if fwhm == -1:
            blurToReturn = self.basevol
        else:
            if not fwhm:
                fwhm = self.lastblur
                if gradient:
                    fwhm = self.lastgradient
            if gradient:
                blurToReturn = self.gradients[fwhm]
            else:
                blurToReturn = self.blurs[fwhm]
        return(blurToReturn)

    def addBlur(self, filename, fwhm, gradient=None):
        """adds the blur with the specified kernel"""
        self.blurs[fwhm] = filename
        self.lastblur = fwhm
        if gradient:
            self.gradients[fwhm] = gradient
            self.lastgradient = fwhm
    
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
        xfmOutputDir = self.tmpDir
        for k, l in arglist:
            if k == 'blur':
                xfmFileName += [str(l) + "b"]
            elif k == 'gradient':
                if l:
                    xfmFileName += ["dxyz"]
            elif k == 'linearparam':
                xfmFileName += [l]
            elif k == 'iterations':
                xfmFileName += ["i" + str(l)]
            elif k == 'step':
                xfmFileName += ["s" + str(l)]
            elif k == 'defaultDir':
                xfmOutputDir = self.setOutputDirectory(str(l))
        xfmFileWithExt = "_".join(xfmFileName) + ".xfm"
        outputXfm = fh.createBaseName(xfmOutputDir, xfmFileWithExt)
        self.addAndSetXfmToUse(targetFilename, outputXfm)
        return(outputXfm)
    
    def setOutputDirectory(self, defaultDir):
        """sets output directory based on defaults for each type of call
        allows for the possibility that an entirely new directory may be specified
        e.g. pipeline_name_nlin or pipeline_name_lsq6 that does not depend on 
        existing file handlers. Additional cases may be added in the future"""
        outputDir = curdir
        if defaultDir=="tmp":
            outputDir = self.tmpDir
        elif defaultDir=="resampled":
            outputDir = self.resampledDir
        elif defaultDir=="labels":
            outputDir = self.labelsDir
        elif defaultDir=="transforms":
            outputDir = self.transformsDir
        else:
            outputDir = abspath(defaultDir)
        return(outputDir)
    
    #MF TODO: This code is getting a bit repetitive. Lets see if we can't
    # consolidate a bit.     
    def getBlur(self, fwhm=None, gradient=False): 
        return(self.groupedFiles[self.currentGroupIndex].getBlur(fwhm, gradient))
    def setBlurToUse(self, fwhm):
        self.groupedFiles[self.currentGroupIndex].lastblur = fwhm
    def getLastBasevol(self):
        return(self.groupedFiles[self.currentGroupIndex].basevol)
    def setLastBasevol(self, newBaseVol, setMain=False):
        self.groupedFiles[self.currentGroupIndex].basevol = newBaseVol
        if setMain:
            self.lastBaseVol = newBaseVol
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
        if not xfm in currGroup.transforms[targetFilename]:
            currGroup.transforms[targetFilename].append(xfm)
        self.setLastXfm(targetFilename, xfm)
    def addLabels(self, newLabel, inputLabel=False):
        """Add labels to array."""
        labelArray = self.returnLabels(inputLabel)
        if not newLabel in labelArray:
            labelArray.append(newLabel)
    def returnLabels(self, inputLabel=False):
        """Return appropriate set of labels"""
        currGroup = self.groupedFiles[self.currentGroupIndex]
        if inputLabel:
            labelArray = currGroup.inputLabels
        else:
            labelArray = currGroup.labels
        return(labelArray)
    def clearLabels(self, inputLabel):
        currGroup = self.groupedFiles[self.currentGroupIndex]
        if inputLabel:
            del currGroup.inputLabels[:]
        else:
            del currGroup.labels[:]
    def setMask(self, inputMask):
        self.groupedFiles[self.currentGroupIndex].mask = inputMask
    def getMask(self):
        return(self.groupedFiles[self.currentGroupIndex].mask)
    def blurFile(self, fwhm, gradient=False, defaultDir="tmp"):
        """create filename for a mincblur call

        Return a triplet of the basename, which mincblur needs as its
        input, the full filename, which mincblur will create after its done,
        and the log file"""
        
        #MF TODO: Error handling if there is no lastBaseVol
        lastBaseVol = self.getLastBasevol()
        outputbase = fh.removeBaseAndExtension(lastBaseVol)
        outputDir = self.setOutputDirectory(defaultDir)
        outputbase = "%s/%s_fwhm%g" % (outputDir, outputbase, fwhm)
        
        withext = "%s_blur.mnc" % outputbase     
        log = self.logFromFile(withext)

        outlist = { "base" : outputbase,
                    "file" : withext,
                    "log"  : log }
        
        if gradient:
            gradWithExt = "%s_dxyz.mnc" % outputbase
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
