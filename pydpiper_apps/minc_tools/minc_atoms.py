#!/usr/bin/env python

from pydpiper.pipeline import CmdStage, Pipeline
from pydpiper_apps.minc_tools.registration_functions import isFileHandler
import pydpiper_apps.minc_tools.registration_functions as rf
from os.path import abspath, basename, join
from os import curdir
import pydpiper.file_handling as fh
import sys
import fnmatch
import Pyro
import re
import copy
from os.path import splitext

Pyro.config.PYRO_MOBILE_CODE=1

class mincANTS(CmdStage):
    def __init__(self,
                 inSource,
                 inTarget,
                 output=None,
                 logFile=None,
                 defaultDir="transforms", 
                 blur=[-1, 0.056],
                 gradient=[False, True],
                 target_mask=None, #ANTS only uses one mask
                 similarity_metric=["CC", "CC"],
                 weight=[1,1],
                 iterations="100x100x100x150",
                 radius_or_histo=[3,3],
                 transformation_model="SyN[0.1]", 
                 regularization="Gauss[2,1]",
                 useMask=True):
        CmdStage.__init__(self, None) #don't do any arg processing in superclass
        try: 
            if isFileHandler(inSource, inTarget):
                """Same defaults as minctracc class:
                    blur = None --> return lastblur
                    gradient = True --> return gradient instead of blur
                    if blur = -1 --> lastBaseVol returned and gradient ignored"""
                self.source = []
                self.target = []
                # Need to check that length of blur, gradient, similarity, weight
                # and radius_or_histo are the same
                self.checkArrayLengths(blur, 
                                       gradient, 
                                       similarity_metric,
                                       weight,
                                       radius_or_histo)
                for i in range(len(blur)):
                    self.source.append(inSource.getBlur(blur[i], gradient[i]))
                    self.target.append(inTarget.getBlur(blur[i], gradient[i]))
                """If no output transform is specified, use registerVolume to create a default.
                   If an output transform name is specified, use this as the output, and add it as the last xfm between source and target. 
                   Note: The output file passed in must be a full path."""
                if not output:
                    outputXfm = inSource.registerVolume(inTarget, defaultDir)
                    self.output = outputXfm
                else:
                    self.output = output
                    inSource.addAndSetXfmToUse(inTarget, self.output)
                self.logFile = fh.logFromFile(inSource.logDir, self.output)
                self.useMask=useMask
                if self.useMask:
                    self.target_mask = inTarget.getMask()
            else:
                self.source = inSource
                self.target = inTarget
                #MF TODO: Need to find a way to specify multiple source and targets
                #based on blur and gradient 
                self.output = output
                if not logFile:
                    self.logFile = fh.logFromFile(abspath(curdir), output)
                else:
                    self.logFile = logFile
                self.useMask=useMask
                if self.useMask:
                    self.target_mask = target_mask
        except:
            print "Failed in putting together mincANTS command."
            print "Unexpected error: ", sys.exc_info()
        
        self.similarity_metric = similarity_metric
        self.weight = weight 
        self.iterations = iterations
        self.radius_or_histo = radius_or_histo
        """Single quotes needed on the command line for 
           transformation_model and regularization
        """
        self.transformation_model = "'" + transformation_model + "'" 
        self.regularization = "'" + regularization + "'"
        
        self.addDefaults()
        self.finalizeCommand()
        self.setName()
        self.colour = "red"
        
    def setName(self):
        self.name = "mincANTS"
    def addDefaults(self):
        cmd = []
        for i in range(len(self.similarity_metric)):
            cmd.append("-m")
            subcmd = ",".join([str(self.source[i]), str(self.target[i]), 
                      str(self.weight[i]), str(self.radius_or_histo[i])])
            cmd.append("".join(["'", str(self.similarity_metric[i]), "[", subcmd, "]", "'"]))
        self.cmd = ["mincANTS", "3", "--number-of-affine-iterations", "0"]
        for c in cmd:
            self.cmd += [c]
        self.cmd += ["-t", self.transformation_model,
                    "-r", self.regularization,
                    "-i", self.iterations,
                    "-o", self.output]
        for i in range(len(self.source)):
            self.inputFiles += [self.source[i], self.target[i]]
        self.outputFiles = [self.output]
        if self.useMask and self.target_mask:
            self.cmd += ["-x", str(self.target_mask)]
            self.inputFiles += [self.target_mask]
    def finalizeCommand(self):
        pass
    
    def checkArrayLengths(self, blur, gradient, metric, weight, radius):
        arrayLength = len(blur)
        errorMsg = "Array lengths for mincANTS command do not match."
        if (len(gradient) != arrayLength 
            or len(metric) != arrayLength
            or len(weight) != arrayLength
            or len(radius) != arrayLength):
            print errorMsg
            raise
        else:
            return
        
class minctracc(CmdStage):
    def __init__(self, 
                 inSource,
                 inTarget,
                 output=None,
                 logFile=None,
                 defaultDir="transforms", 
                 blur=None,
                 gradient=False,
                 linearparam="nlin",
                 source_mask=None, 
                 target_mask=None,
                 iterations=40,
                 step=0.5,
                 transform=None,
                 weight=0.8,
                 stiffness=0.98,
                 similarity=0.8,
                 w_translations=0.4,
                 w_rotations=0.0174533,
                 w_scales=0.02,
                 w_shear=0.02,
                 simplex=1,
                 optimization="-use_simplex",
                 useMask=True):
        #MF TODO: Specify different w_translations, rotations, scales shear in each direction?
        # Now assumes same in all directions
        # Go to more general **kwargs?
        
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
                """ if blur = None, getBlur returns lastblur
                if gradient is true, getBlur returns gradient instead of blur 
                if blur = -1, lastBaseVol is returned and gradient is ignored.
                
                self.transform will be None if there is no previous transform
                between input and target. If this is the case, lsq6 and lsq12
                defaults are added in the setTransforms function
                """
                self.source = inSource.getBlur(blur, gradient)
                self.target = inTarget.getBlur(blur, gradient)
                self.transform = inSource.getLastXfm(inTarget)
                """If no output transform is specified, use registerVolume to create a default.
                   If an output transform name is specified, use this as the output, and add it as the last xfm between source and target. 
                   Note: The output file passed in must be a full path."""
                if not output:
                    outputXfm = inSource.registerVolume(inTarget, defaultDir)
                    self.output = outputXfm
                else:
                    self.output = output
                    inSource.addAndSetXfmToUse(inTarget, self.output)
                    outputXfm = output
                self.logFile = fh.logFromFile(inSource.logDir, outputXfm)
                self.useMask = useMask
                if self.useMask:
                    self.source_mask = inSource.getMask()
                    self.target_mask = inTarget.getMask()
            else:
                self.source = inSource
                self.target = inTarget
                self.output = output
                if not logFile:
                    self.logFile = fh.logFromFile(abspath(curdir), output)
                else:
                    self.logFile = logFile
                self.transform = transform
                self.useMask = useMask
                if self.useMask:
                    self.source_mask = source_mask
                    self.target_mask = target_mask 
        except:
            print "Failed in putting together minctracc command."
            print "Unexpected error: ", sys.exc_info()
        
        self.linearparam = linearparam       
        self.iterations = str(iterations)
        self.lattice_diameter = str(step*3.0)
        self.step = str(step)       
        self.weight = str(weight)
        self.stiffness = str(stiffness)
        self.similarity = str(similarity)
        self.w_translations = str(w_translations)
        self.w_rotations = str(w_rotations)
        self.w_scales = str(w_scales)
        self.w_shear = str(w_shear)
        self.simplex = str(simplex)
        self.optimization = str(optimization)

        self.addDefaults()
        self.finalizeCommand()
        self.setTransform()
        self.setName()
        self.colour = "red"

    def setName(self):
        if self.linearparam == "nlin":
            self.name = "minctracc nlin step: " + self.step 
        else:
            self.name = "minctracc" + self.linearparam + " "
    def addDefaults(self):
        self.cmd = ["minctracc",
                    "-clobber",
                    "-w_translations", self.w_translations,self.w_translations,self.w_translations,
                    "-w_rotations", self.w_rotations, self.w_rotations, self.w_rotations,
                    "-w_scales", self.w_scales, self.w_scales, self.w_scales,
                    "-w_shear", self.w_shear, self.w_shear, self.w_shear,
                    "-step", self.step, self.step, self.step,
                    "-simplex", self.simplex, self.optimization,
                    "-tol", str(0.0001), 
                    self.source,
                    self.target,
                    self.output]
        
        # adding inputs and outputs
        self.inputFiles = [self.source, self.target]
        if self.useMask:
            if self.source_mask:
                self.inputFiles += [self.source_mask]
                self.cmd += ["-source_mask", self.source_mask]
            if self.target_mask:
                self.inputFiles += [self.target_mask]
                self.cmd += ["-model_mask", self.target_mask]
        self.outputFiles = [self.output]

    def setTransform(self):
        """If there is no last transform between the input and target (if using file handlers)
           or if there is no transform specified as an argument (if not using file handlers)
           set defaults based on linear parameter. If a transform is specified, use that one.
           Note that nothing is specified for nonlinear registrations with no transform,
           the minctracc defaults are fine. 
        """
        if not self.transform:
            if self.linearparam == "lsq6":
                self.cmd += ["-est_center", "-est_translations"]
            elif self.linearparam == "lsq12" or self.linearparam=="nlin" or self.linearparam == "lsq6-identity":
                self.cmd += ["-identity"]
        else:
            self.inputFiles += [self.transform]
            self.cmd += ["-transform", self.transform]
        
    def finalizeCommand(self):
        """add the options to finalize the command"""
        if self.linearparam == "nlin":
            """add options for non-linear registration"""
            self.cmd += ["-iterations", self.iterations,
                         "-similarity", self.similarity,
                         "-weight", self.weight,
                         "-stiffness", self.stiffness,
                         "-nonlinear", "corrcoeff", "-sub_lattice", "6",
                         "-lattice_diameter", self.lattice_diameter,
                         self.lattice_diameter, self.lattice_diameter, 
                         "-max_def_magnitude", str(1),
                         "-debug", "-xcorr"]
        else:
            #MF TODO: Enforce that options must be lsq6/7/9/12?
            """add the options for a linear fit"""
            _numCmd = "-" + self.linearparam
            self.cmd += ["-xcorr", _numCmd]

class blur(CmdStage):
    def __init__(self, 
                 inFile, 
                 fwhm, 
                 defaultDir="tmp",
                 gradient=False):
        """calls mincblur with the specified 3D Gaussian kernel

        The inputs can be in one of two styles. The first argument can
        be an instance of RegistrationPipeFH, in which case the last
        volume in that instance (i.e. inFile.lastBasevol) will be
        blurred and the output will be determined by its blurFile
        method. Alternately, the inFile can be a string representing a
        filename, in which case the output and logfile will be set based on 
        the inFile name. If the fwhm specified is -1, we do not construct 
        a command.

        """
        
        if fwhm == -1:
            return
        
        CmdStage.__init__(self, None)
        try:
            if isFileHandler(inFile):
                blurlist = inFile.blurFile(fwhm, gradient, defaultDir)
                self.base = blurlist["base"]
                self.inputFiles = [inFile.getLastBasevol()]
                self.outputFiles = [blurlist["file"]]
                self.logFile = blurlist["log"]
                self.name = "mincblur " + str(fwhm) + " " + inFile.basename
                if gradient:
                    self.outputFiles.append(blurlist["gradient"])
            else:
                self.base = str(inFile).replace(".mnc", "")
                self.inputFiles = [inFile]
                blurBase = "".join([self.base, "_fwhm", str(fwhm), "_blur"])
                output = "".join([blurBase, ".mnc"])
                self.outputFiles = [output]
                self.logFile = fh.logFromFile(abspath(curdir), output)
                self.name = "mincblur " + str(fwhm) + " " + basename(inFile)
                if gradient:
                    gradientBase = blurBase.replace("blur", "dxyz")
                    self.outputFiles += ["".join([gradientBase, ".mnc"])] 
        except:
            print "Failed in putting together blur command."
            print "Unexpected error: ", sys.exc_info()
            
        self.cmd = ["mincblur", "-clobber", "-no_apodize", "-fwhm", str(fwhm),
                    self.inputFiles[0], self.base]
        if gradient:
            self.cmd += ["-gradient"]       
        self.colour="blue"

class autocrop(CmdStage):
    def __init__(self, 
                 resolution, 
                 inFile,
                 output=None,
                 logFile=None,
                 defaultDir="resampled"):
        
        """Resamples the input file to the resolution specified
           using autocrop. The -resample flag forces the use of
           mincresample.
           
           Resolutions should be specified in mm. 
           e.g. 56 microns should be specified as 0.056    
        """
           
        CmdStage.__init__(self, None)
        self.resolution = str(resolution)
        try:  
            if isFileHandler(inFile):
                self.inFile = inFile.getLastBasevol()               
                self.outfile = self.setOutputFile(inFile, defaultDir)
                self.logFile = fh.logFromFile(inFile.logDir, self.outfile)
            else:
                self.inFile = inFile
                self.outfile = output
                if not logFile:
                    self.logFile = fh.logFromFile(abspath(curdir), output)
                else:
                    self.logFile = logFile
    
        except:
            print "Failed in putting together autocrop command"
            print "Unexpected error: ", sys.exc_info()
            
        self.addDefaults()
        self.finalizeCommand()
        self.setName()
        
    def addDefaults(self):
        self.inputFiles += [self.inFile]   
        self.outputFiles += [self.outfile]       
        self.cmd += ["autocrop",
                     "-resample",
                     "-isostep", self.resolution] 
                 
    def finalizeCommand(self):
        self.cmd += ["-clobber", self.inFile, self.outfile]    
    def setName(self):
        self.name = "autocrop " 
    def setOutputFile(self, inFile, defaultDir):
        outDir = inFile.setOutputDirectory(defaultDir)
        outBase = (fh.removeBaseAndExtension(inFile.getLastBasevol()) + "_"
                   + self.resolution + "res.mnc")
        outputFile = fh.createBaseName(outDir, outBase)
        inFile.setLastBasevol(outputFile)
        return(outputFile)  
    
class mincresampleFileAndMask(object):
    """
        If the input file to mincresample(CmdStage) is a file handler, and there is
        a mask associated with the file, the most intuitive thing to do is 
        to resample both the file and the mask.   However, a true atom/command stage
        can only create a single stage, and a such mincresample(CmdStage) can not 
        resample both.  When using a file handler, the mask file associated with it
        is used behind the scenes without the user explicitly specifying this behaviour.
        That's why it is important that the mask always remains current/up-to-date.  The
        best way to do that is to automatically resample the associated mask when the 
        main file is being resampled.  And that is where this class comes in.  It serves
        as a wrapper around mincresample(CmdStage) and  mincresampleMask(CmdStage).  It 
        will check whether the input file is a file handler, and if so, will resample 
        the mask that is associated with it (if it exists).
        
        This class is not truly an atom/command stage, so technically should not live in 
        the minc_atoms module.  It is still kept here because in essence it serves as a 
        single indivisible stage.  (and because the user is more likely to call/find it
        when looking for the mincresample stage) 
    """
    def __init__(self,
                 inFile,
                 targetFile,
                 nameForStage=None,
                 **kwargs):
        self.p = Pipeline()
        self.outputFiles = [] # this will contain the outputFiles from the mincresample of the main MINC file
        self.outputFilesMask = [] # this will contain the outputFiles from the mincresample of the mask belonging to the main MINC file
        

        # the first step is to simply run the mincresample command:
        fileRS = mincresample(inFile,
                              targetFile,
                              **kwargs)
        if(nameForStage):
            fileRS.name = nameForStage
        self.p.addStage(fileRS)
        self.outputFiles = fileRS.outputFiles
        
        # initialize the array of outputs for the mask in case there is none to be resampled
        self.outputFilesMask = [None] * len(self.outputFiles)
        
        # next up, is this a file handler, and if so is there a mask that needs to be resampled?
        if(isFileHandler(inFile)):
            if(inFile.getMask()):
                # there is a mask associated with this file, should be updated
                # we have to watch out in terms of interpolation arguments, if 
                # the original resample command asked for "-sinc" or "-tricubic"
                # for instance, we should remove that argument for the mask resampling
                # these options would reside in the argArray...
                maskArgs = copy.deepcopy(kwargs)
                if(maskArgs["argArray"]):
                    argList = maskArgs["argArray"]
                    for i in range(len(argList)):
                        if(re.match("-sinc", argList[i]) or
                           re.match("-trilinear", argList[i]) or
                           re.match("-tricubic", argList[i]) ):
                            del argList[i]
                    maskArgs["argArray"] = argList                   
                
                # if the output file for the mincresample command was already
                # specified, add "_mask.mnc" to it
                if(maskArgs["output"]):
                    maskArgs["output"] = re.sub(".mnc", "_mask.mnc", maskArgs["output"])
                    
                maskRS = mincresampleMask(inFile,
                                          targetFile,
                                          **maskArgs)
                if(nameForStage):
                    maskRS.name = nameForStage + "--mask--"
                self.p.addStage(maskRS)   
                self.outputFilesMask = maskRS.outputFiles
        
    
class mincresample(CmdStage):  
    def __init__(self,              
                 inFile,
                 targetFile,
                 **kwargs):
        
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
        
        argArray = kwargs.pop("argArray", None)
        if not argArray:
            CmdStage.__init__(self, ["mincresample"])
        else:
            CmdStage.__init__(self, ["mincresample"] + argArray)
          
        try:
            #MF TODO: What if we don't want to use lastBasevol?  
            if isFileHandler(inFile, targetFile):              
                self.inFile = self.getFileToResample(inFile, **kwargs)
                self.targetFile = targetFile.getLastBasevol()
                likeFile=kwargs.pop("likeFile", None)
                if likeFile:
                    if isFileHandler(likeFile):
                        self.likeFile = likeFile.getLastBasevol() 
                    else:
                        print "likeFile must be RegistrationPipeFH or RegistrationFHBase."
                        raise 
                invert = False
                for cmd in self.cmd:
                    if fnmatch.fnmatch(cmd, "*-invert*"):
                        invert = True
                        break
                xfm = kwargs.pop("transform", None)
                if xfm:
                    self.cxfm = xfm
                else:
                    if invert:
                        self.cxfm = targetFile.getLastXfm(inFile)
                    else:
                        self.cxfm = inFile.getLastXfm(targetFile)
                self.outputLocation=kwargs.pop("outputLocation", None)
                if not self.outputLocation: 
                    self.outputLocation=inFile
                else:
                    if not isFileHandler(self.outputLocation):
                        print "outputLocation must be RegistrationPipeFH or RegistrationFHBase."
                        raise
                default = kwargs.pop("defaultDir", None)
                if not default:
                    defaultDir = "resampled"
                else:
                    defaultDir = default
                """If an output file is specified, then use it, else create a default file name.
                   Note: The output file passed in must be a full path."""
                output = kwargs.pop("output", None)
                if not output:
                    self.outfile = self.setOutputFile(self.outputLocation, defaultDir)
                else:
                    self.outfile = output
                self.logFile = fh.logFromFile(self.outputLocation.logDir, self.outfile)
            else:
                self.inFile = inFile
                self.targetFile = targetFile
                self.likeFile = kwargs.pop("likeFile", None)
                self.cxfm = kwargs.pop("transform", None)
                self.outfile=kwargs.pop("output", None)
                logFile=kwargs.pop("logFile", None)
                if not logFile:
                    self.logFile = fh.logFromFile(abspath(curdir), self.outfile)
                else:
                    self.logFile = logFile
    
        except:
            print "Failed in putting together resample command"
            print "Unexpected error: ", sys.exc_info()
            
        self.addDefaults()
        self.finalizeCommand()
        self.setName()
        if isFileHandler(inFile, targetFile):
            self.setLastResampledFile()
        
    def addDefaults(self):
        self.inputFiles += [self.inFile, self.targetFile]   
        self.outputFiles += [self.outfile] 
        if self.likeFile:
            self.cmd += ["-like", self.likeFile] 
            if not self.likeFile in self.inputFiles:
                self.inputFiles += [self.likeFile]
        if self.cxfm:
            self.inputFiles += [self.cxfm]
            self.cmd += ["-transformation", self.cxfm]              
    def finalizeCommand(self):
        """Add -2, clobber, input and output files """
        self.cmd += ["-2", "-clobber", self.inFile, self.outfile]
    def setName(self):
        self.name = "mincresample " 
    def setOutputFile(self, FH, defaultDir):
        outBase = fh.removeBaseAndExtension(self.cxfm) + "-resampled.mnc"
        outDir = FH.setOutputDirectory(defaultDir)
        return(fh.createBaseName(outDir, outBase))  
    def getFileToResample(self, inputFile, **kwargs):
        return(inputFile.getLastBasevol())
    def setLastResampledFile(self):
        # We want to keep track of the last file that was resampled.  This can be 
        # useful when we want to set the lastBaseVol or mask related to this file
        # handler.  This function needs to be overridden by the children of this
        # class, because depending on what we resample (main file, mask, labels)
        # a different "setLast...Vol" needs to be called.
        #
        # For the main mincresample class, it should be the setLastResampledVol
        self.outputLocation.setLastResampledVol(self.outputFiles[0])

class mincresampleLabels(mincresample):
    def __init__(self, 
                 inFile, 
                 targetFile,
                 **kwargs):
        self.initInputLabels(kwargs.pop("setInputLabels", None))
        mincresample.__init__(self,
                           inFile,
                           targetFile, 
                           **kwargs)
        if isFileHandler(self.outputLocation):
            #After other initialization, addLabels to appropriate array
            self.addLabelsToArray(self.outputLocation)
    
    def initInputLabels(self, setLabels):
        if setLabels:
            self.setInputLabels = setLabels
        else:
            self.setInputLabels = False    
    def finalizeCommand(self):
        """additional arguments needed for resampling labels"""
        self.cmd += ["-keep_real_range", "-nearest_neighbour"]
        mincresample.finalizeCommand(self)
    def setOutputFile(self, FH, defaultDir): 
        """set name of output and add labels to appropriate likeFile labels array"""
        outBase = self.setOutputFileName(FH, append="labels")
        outDir = FH.setOutputDirectory(defaultDir)
        return(fh.createBaseName(outDir, outBase))    
    def setOutputFileName(self, FH, **funcargs):
        endOfFile = "-" + funcargs["append"] + ".mnc"
        if self.setInputLabels:
            outBase = fh.removeBaseAndExtension(self.cxfm)
            if fnmatch.fnmatch(outBase, "*_minctracc_*"):
                outputName = outBase.split("_minctracc_")[0]
            elif fnmatch.fnmatch(outBase, "*_ANTS_*"):
                outputName = outBase.split("_ANTS_")[0]
            else:
                outputName = outBase
            outBase = outputName + "-input"
        else:
            labelsToResample = fh.removeBaseAndExtension(self.inFile)
            likeBaseVol = fh.removeBaseAndExtension(FH.getLastBasevol())
            outBase = labelsToResample + "_to_" + likeBaseVol 
        outBase += endOfFile
        return outBase
    def addLabelsToArray(self, FH):
        FH.addLabels(self.outfile, inputLabel=self.setInputLabels)
    def getFileToResample(self, inputFile, **kwargs):
        index = kwargs.pop("labelIndex", None)
        if index > -1:
            # We always resample from inputLabels, so use returnLabels(True)
            labelArray=inputFile.returnLabels(True)
        else:
            labelArray[index] = None
        return(labelArray[index]) 
    def setLastResampledFile(self):
        # Currently we do not keep track of the last label file that is 
        # resampled
        pass

class mincresampleMask(mincresampleLabels):
    def __init__(self, 
                 inFile, 
                 targetFile,
                 **kwargs):
        mincresampleLabels.__init__(self,
                                    inFile,
                                    targetFile, 
                                    **kwargs)
    def getFileToResample(self, inputFile, **kwargs):
        #MF TODO: We will have to adjust this if we allow for pairwise
        # crossing to calculate masks. 
        """ Assume we are using mask from inputFile. If this does not exist,
            we assume inputLabels are also masks from previous iteration
            and we can use same logic as for mask=False. 
        """              
        maskToUse = inputFile.getMask()
        if maskToUse:
            return maskToUse
        else:
            index = kwargs.pop("labelIndex", None)
            labelArray=inputFile.returnLabels(True)
            return(labelArray[index])
    def setOutputFile(self, FH, defaultDir):
        # add -mask to appended file
        outBase = self.setOutputFileName(FH, append="mask")
        outDir = FH.setOutputDirectory(defaultDir)
        return(fh.createBaseName(outDir, outBase))
    def setLastResampledFile(self):
        # Instead of setting the LastResampledVol, here we need to set the
        # LastResampledMaskVol
        self.outputLocation.setLastResampledMaskVol(self.outputFiles[0])

class mincAverage(CmdStage):
    def __init__(self, 
                 inputArray, 
                 outputAvg,
                 output=None, 
                 logFile=None, 
                 defaultDir="tmp"):
        CmdStage.__init__(self, None)
        
        try:  
            """If output is fileHandler, we assume input array is as well"""
            if isFileHandler(outputAvg):
                self.filesToAvg = []
                for i in range(len(inputArray)):
                    self.filesToAvg.append(inputArray[i].getLastBasevol()) 
                """If no output file is specified, create default, using file handler
                   otherwise use what is specified."""              
                if not output:
                    self.output = self.setOutputFile(outputAvg, defaultDir)
                else:
                    self.output = output
                self.logFile = fh.logFromFile(outputAvg.logDir, self.output)
            else:
                self.filesToAvg = inputArray
                self.output = outputAvg
                if not logFile:
                    self.logFile = fh.logFromFile(abspath(curdir), outputAvg)
                else:
                    self.logFile = logFile
    
        except:
            print "Failed in putting together mincaverage command"
            print "Unexpected error: ", sys.exc_info()
            
        self.addDefaults()
        self.finalizeCommand()
        self.setName()
        
    def addDefaults(self):
        for i in range(len(self.filesToAvg)):
            self.inputFiles.append(self.filesToAvg[i]) 
        self.sd = splitext(self.output)[0] + "-sd.mnc"  
        self.outputFiles += [self.output, self.sd]       
        self.cmd += ["mincaverage",
                     "-clobber", "-normalize", "-sdfile", self.sd, "-max_buffer_size_in_kb", str(409620)] 
                 
    def finalizeCommand(self):
        for i in range(len(self.filesToAvg)):
            self.cmd.append(self.filesToAvg[i])
        self.cmd.append(self.output)    
    def setName(self):
        self.name = "mincaverage " 
    def setOutputFile(self, inFile, defaultDir):
        outDir = inFile.setOutputDirectory(defaultDir)
        outBase = (fh.removeBaseAndExtension(inFile.getLastBasevol()) + "_" + "avg.mnc")
        outputFile = fh.createBaseName(outDir, outBase)
        return(outputFile)  

class mincAverageDisp(mincAverage):
    def __init__(self, 
                 inputArray, 
                 output, 
                 logFile=None, 
                 defaultDir=None):
        mincAverage.__init__(self, inputArray,output,logFile,defaultDir)
        
    def addDefaults(self):
        for i in range(len(self.filesToAvg)):
            self.inputFiles.append(self.filesToAvg[i]) 
        self.outputFiles += [self.output]       
        self.cmd += ["mincaverage", "-clobber"] 

class RotationalMinctracc(CmdStage):
    """
        This class runs a rotational_minctracc.py call on its two input 
        files.  That program performs a 6 parameter (rigid) registration
        by doing a brute force search in the x,y,z rotation space.  Normally
        the input files have unknown orientation.  Input and output files
        can be specified either as file handlers, or as string representing
        the filenames.
        
        * The input files are assumed to have already been blurred appropriately
        
        There are a number of parameters that have to be set and this 
        will be done using factors that depend on the resolution of the
        input files.  The blur parameter is given in mm, not as a factor 
        of the input resolution.  The blur parameter is necessary in order
        to retrieve the correct blur file from the file handler.  Here is the list:
        
        argument to be set   --  default (factor)  -- (for 56 micron, translates to)
                blur                  0.56 (mm)                 (560 micron) (note, this is in mm, not a factor)
          resample stepsize            4                        (224 micron)
        registration stepsize         10                        (560 micron)
          w_translations               8                        (448 micron)
         
        Specifying -1 for the blur argument will result in retrieving an unblurred file.
        The two other parameters that can be set are (in degrees) have defaults:
        
            rotational range          50
            rotational interval       10
        
        Whether or not a mask will be used is based on the presence of a mask 
        in the target file.  Alternatively, a mask can be specified using the
        maskFile argument.
    """
    def __init__(self, 
                 inSource, 
                 inTarget,
                 output = None, # ability to specify output transform when using strings for input
                 logFile = None,
                 maskFile = None,
                 defaultDir="transforms",
                 blur=0.56,
                 resample_step=4,
                 registration_step=10,
                 w_translations=8,
                 rotational_range=50,
                 rotational_interval=10,
                 mousedata=False):
        
        CmdStage.__init__(self, None) #don't do any arg processing in superclass
        # handling of the input files
        try: 
            if rf.isFileHandler(inSource, inTarget):
                self.source = inSource.getBlur(fwhm=blur)
                self.target = inTarget.getBlur(fwhm=blur)  
                if(output == None):
                    self.output = inSource.registerVolume(inTarget, defaultDir)
                else:
                    self.output = output
                if(logFile == None):
                    self.logFile = fh.logFromFile(inSource.logDir, self.output)
                else:
                    self.logFile = logFile
            else:
                # TODO: fix this to work with string input files
                self.source = inSource
                self.target = inTarget
        except:
            print "Failed in putting together RotationalMinctracc command."
            print "Unexpected error: ", sys.exc_info()
            raise
        
        highestResolution = rf.getFinestResolution(inSource)
        
        # TODO: finish the following if clause... hahaha
        #if(mousedata):
            
        
        self.addDefaults(resample_step     * highestResolution,
                      registration_step * highestResolution,
                      w_translations    * highestResolution,
                      int(rotational_range),
                      int(rotational_interval))
        # potentially add a mask to the command
        self.finalizeCommand(inTarget, maskFile)
        self.setName()
        self.colour = "green"

    def setName(self):
        self.name = "rotational-minctracc" 

    def addDefaults(self,
                 resamp_step,
                 reg_step,
                 w_trans,
                 rot_range,
                 rot_interval):
        
        w_trans_string = str(w_trans) + ',' + str(w_trans) + ',' + str(w_trans)
        cmd = ["rotational_minctracc.py", 
               "-t", "/dev/shm/", 
               "-w", w_trans_string,
               "-s", str(resamp_step),
               "-g", str(reg_step),
               "-r", str(rot_range),
               "-i", str(rot_interval),
               self.source,
               self.target,
               self.output,
               "/dev/null"]
        self.inputFiles = [self.source, self.target] 
        self.outputFiles = [self.output] 
        self.cmd = cmd
        
    def finalizeCommand(self,
                        inTarget,
                        maskFile):
        if(maskFile):
            # a mask file have been given directly, choose
            # this one over the potential mask present
            # in the target
            self.cmd += ["-m", maskFile]
            self.inputFiles.append(maskFile)
        else:
            try:
                mask = inTarget.getMask()
                if mask:
                    self.cmd += ["-m", mask]
                    self.inputFiles.append(mask)
            except:
                print "Failed retrieving information about a mask for the target in RotationalMinctracc."
                print "Unexpected error: ", sys.exc_info()
                raise


class xfmConcat(CmdStage):
    """
        Calls xfmconcat on one or more input transformations
        
        inputFiles: these are assumed to be passed in as input filename  
        strings.  If more than one input file is passed, they should be 
        passed as a list
        
        outputFile: string representing the output filename 
        
        logFile: string representing the output filename for the log
        file for this command. If unspecified, self.logFile will be set in
        CmdStage.__init__ (or subsequently, using the setLogFile function) 
    """
    def __init__(self, 
                 inputFiles,
                 outputFile,
                 logFile=None):
        CmdStage.__init__(self, None)
        
        # in case there is a single input file... (it's actually possible)
        if(not(type(inputFiles) is list)):
            inputFiles = [inputFiles]
        
        self.inputFiles = inputFiles
        self.outputFiles = [outputFile]
        self.logFile = logFile
        self.cmd = ["xfmconcat", "-clobber"]
        self.cmd += inputFiles
        self.cmd += [outputFile]
        self.name   = "xfm-concat"
        self.colour = "yellow"
                        
                        
                        
                        
