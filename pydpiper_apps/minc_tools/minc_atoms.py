#!/usr/bin/env python

from pydpiper.pipeline import CmdStage 
from pydpiper_apps.minc_tools.registration_functions import isFileHandler
from os.path import abspath, basename, join
from os import curdir
import pydpiper.file_handling as fh
import inspect
import sys
import Pyro

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
                 source_mask=None, #ANTS only uses one mask
                 similarity_metric=["CC", "CC"],
                 weight=[1,1],
                 iterations="100x100x100x0",
                 radius_or_histo=[3,3],
                 transformation_model="SyN[0.3]", 
                 regularization="Gauss[5,1]",
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
                frame = inspect.currentframe()
                args,_,_,arglocals = inspect.getargvalues(frame)
                arglist = [(i, arglocals[i]) for i in args]
                outputXfm = inSource.registerVolume(inTarget, arglist, "mincANTS")
                self.output = outputXfm
                self.logFile = fh.logFromFile(inSource.logDir, self.output)
                self.useMask=useMask
                if self.useMask:
                    self.source_mask = inSource.getMask()
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
        if self.useMask and self.source_mask:
            self.cmd += ["-x", str(self.source_mask)]
            self.inputFiles += [self.source_mask]
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
                 similarity=0.3,
                 w_translations=0.2,
                 simplex=1,
                 useMask=True):
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
                """
                self.source = inSource.getBlur(blur, gradient)
                self.target = inTarget.getBlur(blur, gradient)
                if not transform:
                    # Note: this may also be None and should be for initial call
                    targetFilename = fh.removeBaseAndExtension(inTarget.getLastBasevol())
                self.transform = inSource.getLastXfm(targetFilename)
                frame = inspect.currentframe()
                args,_,_,arglocals = inspect.getargvalues(frame)
                arglist = [(i, arglocals[i]) for i in args]
                outputXfm = inSource.registerVolume(inTarget, arglist)
                self.output = outputXfm
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
        self.lattice_diameter = str(step*3)
        self.step = str(step)       
        self.weight = str(weight)
        self.stiffness = str(stiffness)
        self.similarity = str(similarity)
        self.w_translations = str(w_translations)
        self.simplex = str(simplex)

        self.addDefaults()
        self.finalizeCommand()
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
                    "-step", self.step, self.step, self.step,
                    "-simplex", self.simplex,
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
        if self.transform:
            self.inputFiles += [self.transform]
            self.cmd += ["-transform", self.transform]
        self.outputFiles = [self.output]

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
                         self.lattice_diameter, self.lattice_diameter]
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
                self.inputFiles = [inFile.lastBaseVol]
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
                 outFile=None,
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
                self.outfile = outFile
                if not logFile:
                    self.logFile = fh.logFromFile(abspath(curdir), outFile)
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
        inFile.setLastBasevol(outputFile, setMain=False)
        return(outputFile)  
    
class mincresample(CmdStage):
    def __init__(self, 
                 inFile, 
                 outFile=None, 
                 logFile=None,
                 defaultDir="resampled", 
                 likeFile=None, 
                 cxfm=None, 
                 argArray=None,
                 labelIndex=-1,
                 setInputLabels=False, 
                 mask=False):
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
                self.setInputLabels = setInputLabels
                self.inFile = self.getFileToResample(inFile, labelIndex, mask)
                self.likeFile = likeFile.getLastBasevol()
                #If we want to invert the transform, assume that xfm is from likeFile to inFile
                #otherwise, we assume transform is from likeFile to inFile
                # MF TODO: We should not be using __contains__ here, but since 
                # this class likely needs some work, I'm leaving for now. 
                if (self.cmd.__contains__("-invert") 
                    or self.cmd.__contains__("-invert_transform")
                    or self.cmd.__contains__("-invert_transformation")):
                    self.cxfm = likeFile.getLastXfm(fh.removeBaseAndExtension(inFile.getLastBasevol()))
                else:
                    self.cxfm = inFile.getLastXfm(fh.removeBaseAndExtension(self.likeFile))
                self.outfile = self.setOutputFile(likeFile, defaultDir, mask)
                self.logFile = fh.logFromFile(inFile.logDir, self.outfile)
            else:
                self.inFile = inFile
                self.likeFile = likeFile
                self.cxfm = cxfm
                self.outfile=outFile
                if not logFile:
                    self.logFile = fh.logFromFile(abspath(curdir), outFile)
                else:
                    self.logFile = logFile
    
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
        if not self.likeFile in self.inputFiles: 
                self.inputFiles += [self.likeFile]
        if self.cxfm:
            self.inputFiles += [self.cxfm]
            self.cmd += ["-transform", self.cxfm]              
    def finalizeCommand(self):
        """Add -2, clobber, input and output files """
        self.cmd += ["-2", "-clobber", self.inFile, self.outfile]    
    def setName(self):
        self.name = "mincresample " 
    def setOutputFile(self, likeFile, defaultDir, mask=False):
        outDir = likeFile.setOutputDirectory(defaultDir)
        outBase = fh.removeBaseAndExtension(self.cxfm) + "-resampled.mnc"
        return(fh.createBaseName(outDir, outBase))  
    def getFileToResample(self, inputFile, index=-1, mask=False):
        return(inputFile.getLastBasevol())  

class mincresampleLabels(mincresample):
    def __init__(self, 
                 inFile, 
                 outFile=None, 
                 logFile=None, 
                 defaultDir="labels",
                 likeFile=None, 
                 cxfm=None, 
                 argArray=None,
                 labelIndex=-1, 
                 setInputLabels=False,
                 mask=False):
        mincresample.__init__(self,
                           inFile, 
                           outFile, 
                           logFile, 
                           defaultDir,
                           likeFile, 
                           cxfm, 
                           argArray,
                           labelIndex,
                           setInputLabels,
                           mask)
        
        if isFileHandler(likeFile):
            #After other initialization, addLabels to appropriate array
            self.addLabelsToArray(likeFile)
        
    def finalizeCommand(self):
        """additional arguments needed for resampling labels"""
        self.cmd += ["-keep_real_range", "-nearest_neighbour"]
        mincresample.finalizeCommand(self)
    def setOutputFile(self, likeFile, defaultDir, mask=False):
        """set name of output and add labels to appropriate likeFile labels array"""
        if self.setInputLabels:
            outBase = fh.removeBaseAndExtension(self.cxfm)
        else:
            labelsToResample = fh.removeBaseAndExtension(self.inFile)
            likeBaseVol = fh.removeBaseAndExtension(likeFile.getLastBasevol())
            #MF TODO: Might want to make startName more generic
            startName = labelsToResample.split("b_")
            outBase = startName[0] + "b-labels_to_" + likeBaseVol 
        if mask:
            outBase += "-mask.mnc"
        else:
            outBase += "-labels.mnc"
        outDir = likeFile.setOutputDirectory(defaultDir)
        labelFile = fh.createBaseName(outDir, outBase)    
        return(labelFile) 
    def addLabelsToArray(self, likeFile):
        likeFile.addLabels(self.outfile, inputLabel=self.setInputLabels)
    def getFileToResample(self, inputFile, index=-1, mask=False):
        if index > -1:
            # Note: we are always resampling from inputLabels here
            labelArray=inputFile.returnLabels(True)
        else:
            labelArray[index] = None
        if mask:
            #MF TODO: We will have to adjust this if we allow for pairwise
            # crossing to calculate masks. 
            """Assume we are using mask from inputFile. If this does not exist,
                we assume inputLabels are also masks from previous iteration
                and we can use same logic as for mask=False. 
            """
            maskToUse = inputFile.getMask()
            if maskToUse:
                return maskToUse
            else:
                return(labelArray[index]) 
        else:
            return(labelArray[index]) 
