#!/usr/bin/env python

from pydpiper.pipeline import Pipeline, InputFile, OutputFile, LogFile, CmdStage
import atoms_and_modules.LSQ12 as lsq12
import atoms_and_modules.NLIN as nlin
import atoms_and_modules.minc_atoms as ma
import atoms_and_modules.registration_file_handling as rfh
import atoms_and_modules.stats_tools as st
import atoms_and_modules.minc_parameters as mp
import atoms_and_modules.registration_functions as rf
from pydpiper.file_handling import removeBaseAndExtension, createBaseName, logFromFile
from atoms_and_modules.registration_functions import isFileHandler
import pydpiper.file_handling as fh
from pyminc.volumes.factory import volumeFromFile
import sys
from os.path import splitext

class SetResolution:
    def __init__(self, filesToResample, resolution):
        """During initialization make sure all files are resampled
           at resolution we'd like to use for each pipeline stage
        """
        self.p = Pipeline()
        
        for FH in filesToResample:
            dirForOutput = self.getOutputDirectory(FH)
            currentRes = volumeFromFile(FH.getLastBasevol()).separations
            if not abs(abs(currentRes[0]) - abs(resolution)) < 0.01:
                crop = ma.autocrop(resolution, FH, defaultDir=dirForOutput)
                self.p.addStage(crop)
                mask = FH.getMask()
                if mask:
                    #Need to resample the mask as well.
                    cropMask = ma.mincresampleMask(FH,
                                                   FH,
                                                   outputLocation=FH,
                                                   likeFile=FH)
                    self.p.addStage(cropMask)
        
    def getOutputDirectory(self, FH):
        """Sets output directory based on whether or not we have a full
        RegistrationPipeFH class or we are just using RegistrationFHBase"""
        if isinstance(FH, rfh.RegistrationPipeFH):
            outputDir = "resampled"
        else:
            outputDir = FH.basedir
        return outputDir

class LSQ12ANTSNlin:
    """Class that runs a basic LSQ12 registration, followed by a single mincANTS call.
       Currently used in MAGeT, registration_chain and pairwise_nlin."""
    def __init__(self,
                 inputFH,
                 targetFH,
                 lsq12_protocol=None,
                 nlin_protocol=None,
                 subject_matter=None,
                 defaultDir="tmp"):
        
        self.p = Pipeline()
        self.inputFH = inputFH
        self.targetFH = targetFH
        self.lsq12_protocol = lsq12_protocol
        self.nlin_protocol = nlin_protocol
        self.subject_matter = subject_matter
        self.defaultDir = defaultDir
        
        if ((self.lsq12_protocol == None and self.subject_matter==None) or self.nlin_protocol == None):
            # always base the resolution to be used on the target for the registrations
            self.fileRes = rf.returnFinestResolution(self.targetFH)
        else:
            self.fileRes = None
        
        self.buildPipeline()    
    
    def buildPipeline(self):
        # Run lsq12 registration prior to non-linear
        self.lsq12Params = mp.setLSQ12MinctraccParams(self.fileRes, 
                                                      subject_matter=self.subject_matter,
                                                      reg_protocol=self.lsq12_protocol)
        lsq12reg = lsq12.LSQ12(self.inputFH, 
                               self.targetFH, 
                               blurs=self.lsq12Params.blurs,
                               step=self.lsq12Params.stepSize,
                               gradient=self.lsq12Params.useGradient,
                               simplex=self.lsq12Params.simplex,
                               w_translations=self.lsq12Params.w_translations,
                               defaultDir=self.defaultDir)
        self.p.addPipeline(lsq12reg.p)
        
        #Resample using final LSQ12 transform and reset last base volume. 
        res = ma.mincresample(self.inputFH, self.targetFH, likeFile=self.targetFH, argArray=["-sinc"])   
        self.p.addStage(res)
        self.inputFH.setLastBasevol(res.outputFiles[0])
        lsq12xfm = self.inputFH.getLastXfm(self.targetFH)
        
        #Get registration parameters from nlin protocol, blur and register
        #Assume a SINGLE generation here. 
        self.nlinParams = mp.setOneGenMincANTSParams(self.fileRes, reg_protocol=self.nlin_protocol)
        for b in self.nlinParams.blurs:
            for j in b:
                #Note that blurs for ANTS params in an array of arrays. 
                if j != -1:            
                    self.p.addStage(ma.blur(self.targetFH, j, gradient=True))
                    self.p.addStage(ma.blur(self.inputFH, j, gradient=True))
                    
        sp = ma.mincANTS(self.inputFH,
                         self.targetFH,
                         defaultDir=self.defaultDir, 
                         blur=self.nlinParams.blurs[0],
                         gradient=self.nlinParams.gradient[0],
                         similarity_metric=self.nlinParams.similarityMetric[0],
                         weight=self.nlinParams.weight[0],
                         iterations=self.nlinParams.iterations[0],
                         radius_or_histo=self.nlinParams.radiusHisto[0],
                         transformation_model=self.nlinParams.transformationModel[0], 
                         regularization=self.nlinParams.regularization[0],
                         useMask=self.nlinParams.useMask[0])
        self.p.addStage(sp)
        nlinXfm = sp.outputFiles[0]
        #Reset last base volume to original input for future registrations.
        self.inputFH.setLastBasevol(setToOriginalInput=True)
        #Concatenate transforms to get final lsq12 + nlin. Register volume handles naming and setting of lastXfm
        output = self.inputFH.registerVolume(self.targetFH, "transforms")
        xc = ma.xfmConcat([lsq12xfm, nlinXfm], output, fh.logFromFile(self.inputFH.logDir, output))
        self.p.addStage(xc)

class HierarchicalMinctracc:
    """Default HierarchicalMinctracc currently does:
        1. A standard three stage LSQ12 alignment. (See defaults for LSQ12 module.)
        2. A six generation non-linear minctracc alignment. 
       To override these defaults, lsq12 and nlin protocols may be specified. """
    def __init__(self, 
                 inputFH, 
                 targetFH,
                 lsq12_protocol=None,
                 nlin_protocol=None,
                 includeLinear = True,
                 subject_matter = None,  
                 defaultDir="tmp"):
        
        self.p = Pipeline()
        self.inputFH = inputFH
        self.targetFH = targetFH
        self.lsq12_protocol = lsq12_protocol
        self.nlin_protocol = nlin_protocol
        self.includeLinear = includeLinear
        self.subject_matter = subject_matter
        self.defaultDir = defaultDir
        
        if ((self.lsq12_protocol == None and self.subject_matter==None) or self.nlin_protocol == None):
            # the resolution of the registration should be based on the target
            self.fileRes = rf.returnFinestResolution(self.targetFH)
        else:
            self.fileRes = None
        
        self.buildPipeline()
        
    def buildPipeline(self):
            
        # Do LSQ12 alignment prior to non-linear stages if desired
        if self.includeLinear: 
            self.lsq12Params = mp.setLSQ12MinctraccParams(self.fileRes,
                                            subject_matter=self.subject_matter,
                                            reg_protocol=self.lsq12_protocol)
            lsq12reg = lsq12.LSQ12(self.inputFH, 
                                   self.targetFH, 
                                   blurs=self.lsq12Params.blurs,
                                   step=self.lsq12Params.stepSize,
                                   gradient=self.lsq12Params.useGradient,
                                   simplex=self.lsq12Params.simplex,
                                   w_translations=self.lsq12Params.w_translations,
                                   defaultDir=self.defaultDir)
            self.p.addPipeline(lsq12reg.p)
        
        # create the nonlinear registrations
        self.nlinParams = mp.setNlinMinctraccParams(self.fileRes, reg_protocol=self.nlin_protocol)
        for b in self.nlinParams.blurs: 
            if b != -1:           
                self.p.addStage(ma.blur(self.inputFH, b, gradient=True))
                self.p.addStage(ma.blur(self.targetFH, b, gradient=True))
        for i in range(len(self.nlinParams.stepSize)):
            #For the final stage, make sure the output directory is transforms.
            if i == (len(self.nlinParams.stepSize) - 1):
                self.defaultDir = "transforms"
            nlinStage = ma.minctracc(self.inputFH, 
                                     self.targetFH,
                                     defaultDir=self.defaultDir,
                                     blur=self.nlinParams.blurs[i],
                                     gradient=self.nlinParams.useGradient[i],
                                     iterations=self.nlinParams.iterations[i],
                                     step=self.nlinParams.stepSize[i],
                                     w_translations=self.nlinParams.w_translations[i],
                                     simplex=self.nlinParams.simplex[i],
                                     optimization=self.nlinParams.optimization[i])
            self.p.addStage(nlinStage)

class FullIterativeLSQ12Nlin:
    """Does a full iterative LSQ12 and NLIN. Basically iterative model building starting from LSQ6
       and without stats at the end. Designed to be called as part of a larger application. 
       Specifying an initModel is optional, all other arguments are mandatory."""
    def __init__(self, inputs, dirs, options, avgPrefix=None, initModel=None):
        self.inputs = inputs
        self.dirs = dirs
        self.options = options
        self.avgPrefix = avgPrefix
        self.initModel = initModel
        self.nlinFH = None
        
        self.p = Pipeline()
        
        self.buildPipeline()
        
    def buildPipeline(self):
        lsq12LikeFH = None 
        resolutionForLSQ12 = None
        if self.initModel:
            lsq12LikeFH = self.initModel[0]
        elif self.options.lsq12_likeFile: 
            lsq12LikeFH = self.options.lsq12_likeFile 
        
        if lsq12LikeFH == None and self.options.lsq12_subject_matter == None:
            print "\nError: the FullIterativeLSQ12Nlin module was called without specifying either an initial model, nor an lsq12_subject_matter. Currently that means that the code can not determine the resolution at which the registrations should be run. Please specify one of the two. Exiting\n"
            sys.exit()
        
        if not (lsq12LikeFH == None):
            resolutionForLSQ12 = rf.returnFinestResolution(lsq12LikeFH)

        lsq12module = lsq12.FullLSQ12(self.inputs,
                                      self.dirs.lsq12Dir,
                                      likeFile=lsq12LikeFH,
                                      maxPairs=self.options.lsq12_max_pairs,
                                      lsq12_protocol=self.options.lsq12_protocol,
                                      subject_matter=self.options.lsq12_subject_matter,
                                      resolution=resolutionForLSQ12)
        lsq12module.iterate()
        self.p.addPipeline(lsq12module.p)
        self.lsq12Params = lsq12module.lsq12Params
        if lsq12module.lsq12AvgFH.getMask()== None:
            if self.initModel:
                lsq12module.lsq12AvgFH.setMask(self.initModel[0].getMask())
        if not self.avgPrefix:
            self.avgPrefix = self.options.pipeline_name
        # same as in MBM.py:
        # for now we can use the same resolution for the NLIN stages as we did for the 
        # LSQ12 stage. At some point we should look into the subject matter option...
        nlinModule = nlin.initializeAndRunNLIN(self.dirs.lsq12Dir,
                                               self.inputs,
                                               self.dirs.nlinDir,
                                               avgPrefix=self.avgPrefix, 
                                               createAvg=False,
                                               targetAvg=lsq12module.lsq12AvgFH,
                                               nlin_protocol=self.options.nlin_protocol,
                                               reg_method=self.options.reg_method,
                                               resolution=resolutionForLSQ12)
        self.p.addPipeline(nlinModule.p)
        self.nlinFH = nlinModule.nlinAverages[-1]
        self.nlinParams = nlinModule.nlinParams
        self.initialTarget = nlinModule.initialTarget
        # Now we need the full transform to go back to LSQ6 space
        for i in self.inputs:
            linXfm = lsq12module.lsq12AvgXfms[i]
            nlinXfm = i.getLastXfm(self.nlinFH)
            outXfm = st.createOutputFileName(i, nlinXfm, "transforms", "_with_additional.xfm")
            xc = ma.xfmConcat([linXfm, nlinXfm], outXfm, fh.logFromFile(i.logDir, outXfm))
            self.p.addStage(xc)
            i.addAndSetXfmToUse(self.nlinFH, outXfm)
       

class LongitudinalStatsConcatAndResample:
    """ For each subject:
        1. Calculate stats (displacement, absolute jacobians, relative jacobians) between i and i+1 time points 
        2. Calculate transform from subject to common space (nlinFH) and invert it. 
           For most subjects this will require some amount of transform concatenation. 
        3. Calculate the stats (displacement, absolute jacobians, relative jacobians) from common space
           to each timepoint.
    """
    def __init__(self, subjects, timePoint, nlinFH, statsKernels, commonName):
        
        self.subjects = subjects
        self.timePoint = timePoint
        self.nlinFH = nlinFH
        self.blurs = [] 
        self.setupBlurs(statsKernels)
        self.commonName = commonName
        
        self.p = Pipeline()
        
        self.buildPipeline()
    
    def setupBlurs(self, statsKernels):
        if isinstance(statsKernels, list):
            self.blurs = statsKernels
        elif isinstance(statsKernels, str):
            for i in statsKernels.split(","):
                self.blurs.append(float(i))
        else:
            print "Improper type of blurring kernels specified for stats calculation: " + str(statsKernels)
            sys.exit()
    
    def statsCalculation(self, inputFH, targetFH, xfm=None, useChainStats=True):
        """If useChainStats=True, calculate stats between input and target. 
           This happens for all i to i+1 calcs.
           
           If useChainStats=False, calculate stats in the standard way, from target to
           input, We do this, when we go from the common space to all others. """
        if useChainStats:
            stats = st.CalcChainStats(inputFH, targetFH, self.blurs)
        else:
            stats = st.CalcStats(inputFH, targetFH, self.blurs)
        self.p.addPipeline(stats.p)
        """If an xfm is specified, resample all to this common space"""
        if xfm:
            if not self.nlinFH:
                likeFH = targetFH
            else:
                likeFH = self.nlinFH
            res = resampleToCommon(xfm, inputFH, stats.statsGroup, self.blurs, likeFH)
            self.p.addPipeline(res)
    
    def statsAndConcat(self, s, i, count, beforeAvg=True):
        """Construct array to common space for this timepoint.
           This builds upon arrays from previous calls."""
        if beforeAvg:
            xfm = s[i].getLastXfm(s[i+1]) 
        else:
            xfm = s[i].getLastXfm(s[i-1])
        """Set this transform as last xfm from input to nlin and calculate nlin to s[i] stats"""
        if self.nlinFH:
            self.xfmToCommon.insert(0, xfm)
            """ Concat transforms to get xfmToCommon and calculate statistics 
                Note that inverted transform, which is what we want, is calculated in
                the statistics module. """
            xtc = createBaseName(s[i].transformsDir, s[i].basename + "_to_" + self.commonName + ".xfm")
            xc = ma.xfmConcat(self.xfmToCommon, xtc, fh.logFromFile(s[i].logDir, xtc))
            self.p.addStage(xc)
            s[i].addAndSetXfmToUse(self.nlinFH, xtc)
            self.statsCalculation(s[i], self.nlinFH, xfm=None, useChainStats=False)
        else:
            xtc=None
        """Calculate i to i+1 stats for all but final timePoint"""
        if count - i > 1:
            self.statsCalculation(s[i], s[i+1], xfm=xtc, useChainStats=True)
        
    def buildPipeline(self):
        for subj in self.subjects:
            s = self.subjects[subj]
            count = len(s)
            """Wherever iterative model building was run, the indiv --> nlin xfm is stored
               in the group with the name "final". We need to use this group for to get the
               transform and do the stats calculation, and then reset to the current group.
               Calculate stats first from average to timepoint included in average"""
               
            currGroup = s[self.timePoint].currentGroupIndex
            index = s[self.timePoint].getGroupIndex("final")
            xfmToNlin = s[self.timePoint].getLastXfm(self.nlinFH, groupIndex=index)
            
            if xfmToNlin:
                self.xfmToCommon = [xfmToNlin]
            else:
                self.xfmToCommon = []
            if self.nlinFH:
                s[self.timePoint].currentGroupIndex = index
                self.statsCalculation(s[self.timePoint], self.nlinFH, xfm=None, useChainStats=False)
                s[self.timePoint].currentGroupIndex = currGroup
            """Next: If timepoint included in average is NOT final timepoint, 
               also calculate i to i+1 stats."""
            if count - self.timePoint > 1:
                self.statsCalculation(s[self.timePoint], s[self.timePoint+1], xfm=xfmToNlin, useChainStats=True)
            if not self.timePoint - 1 < 0:
                """ Average happened at time point other than first time point. 
                    Loop over points prior to average."""
                for i in reversed(range(self.timePoint)): 
                    self.statsAndConcat(s, i, count, beforeAvg=True)
                         
            """ Loop over points after average. If average is at first time point, this loop
                will hit all time points (other than first). If average is at subsequent time 
                point, it hits all time points not covered previously. xfmToCommon needs to be reset."""
            if xfmToNlin:
                self.xfmToCommon = [xfmToNlin]
            else:
                self.xfmToCommon = []  
            for i in range(self.timePoint + 1, count):
                self.statsAndConcat(s, i, count, beforeAvg=False)
 
def resampleToCommon(xfm, FH, statsGroup, statsKernels, nlinFH):
    blurs = []
    if isinstance(statsKernels, list):
        blurs = statsKernels
    elif isinstance(statsKernels, str):
        for i in statsKernels.split(","):
            blurs.append(float(i))
    else:
        print "Improper type of blurring kernels specified for stats calculation: " + str(statsKernels)
        sys.exit()
    pipeline = Pipeline()
    outputDirectory = FH.statsDir
    filesToResample = []
    for b in blurs:
        filesToResample.append(statsGroup.relativeJacobians[b])
        if statsGroup.absoluteJacobians:
            filesToResample.append(statsGroup.absoluteJacobians[b])
    for f in filesToResample:
        outputBase = removeBaseAndExtension(f).split(".mnc")[0]
        outputFile = createBaseName(outputDirectory, outputBase + "_common" + ".mnc")
        logFile = fh.logFromFile(FH.logDir, outputFile)
        targetAndLike=nlinFH.getLastBasevol()
        res = ma.mincresample(f, 
                              targetAndLike,
                              likeFile=targetAndLike,
                              transform=xfm,
                              output=outputFile,
                              logFile=logFile,
                              argArray=["-sinc"]) 
        pipeline.addStage(res)
    
    return pipeline


class createQualityControlImages(object):
    """
    This class takes a list of input files and creates
    a set of quality control (verification) images. Optionally
    these images can be combined in a single montage image for
    easy viewing
    
    If the inputFiles are fileHandler, the last base volume
    will be used to create the images from.

    The scaling factor corresponds to the the mincpik -scale 
    parameter
    """
    def __init__(self, 
                 inputFiles, 
                 createMontage=True,
                 montageOutPut=None,
                 scalingFactor=20):
        self.p = Pipeline()
        self.individualImages = []

        if createMontage and montageOutPut == None:
            print "\nError: createMontage is specified in createQualityControlImages, but no output name for the montage is provided. Exiting...\n"
            sys.exit()

        # for each of the input files, run a mincpik call and create 
        # a triplane image.
        for file in inputFiles:
            if isFileHandler(file):
                # create command using last base vol
                inputToMincpik = file.getLastBasevol()
                outputMincpik = createBaseName(file.tmpDir,
                                            removeBaseAndExtension(inputToMincpik) + "_QC_image.png")
                cmd = ["mincpik", "-clobber",
                       "-scale", scalingFactor,
                       "-triplanar",
                       InputFile(inputToMincpik),
                       OutputFile(outputMincpik)]
                mincpik = CmdStage(cmd)
                mincpik.setLogFile(LogFile(logFromFile(file.logDir, outputMincpik)))
                self.p.addStage(mincpik)
                self.individualImages.append(outputMincpik)

        # if montageOutput is specified, create the overview image
        if createMontage:
            cmdmontage = ["montage", "-geometry", "+2+2"] \
                         + map(InputFile, self.individualImages) + [OutputFile(montageOutPut)]
            montage = CmdStage(cmdmontage)
            montage.setLogFile(splitext(montageOutPut)[0] + ".log")
            self.p.addStage(montage)
            
                
