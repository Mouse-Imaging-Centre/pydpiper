#!/usr/bin/env python

from pydpiper.pipeline import Pipeline
import atoms_and_modules.LSQ12 as lsq12
import atoms_and_modules.minc_atoms as ma
import atoms_and_modules.registration_file_handling as rfh
import atoms_and_modules.stats_tools as st
import atoms_and_modules.minc_parameters as mp
import atoms_and_modules.registration_functions as rf
import pydpiper.file_handling as fh
from pyminc.volumes.factory import volumeFromFile

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
        try: # the attempt to access the minc volume will fail if it doesn't yet exist at pipeline creation
            self.fileRes = rf.getFinestResolution(self.inputFH)
        except: 
            # if it indeed failed, get resolution from the original file specified for 
            # one of the input files, which should exist. 
            # Can be overwritten by the user through specifying a nonlinear protocol.
            self.fileRes = rf.getFinestResolution(self.inputFH.inputFileName)
        
        self.buildPipeline()    
    
    def buildPipeline(self):
        # Run lsq12 registration prior to non-linear
        lp = mp.setLSQ12MinctraccParams(self.fileRes, 
                                        subject_matter=self.subject_matter,
                                        reg_protocol=self.lsq12_protocol)
        lsq12reg = lsq12.LSQ12(self.inputFH, 
                               self.targetFH, 
                               blurs=lp.blurs,
                               step=lp.stepSize,
                               gradient=lp.useGradient,
                               simplex=lp.simplex,
                               w_translations=lp.w_translations,
                               defaultDir=self.defaultDir)
        self.p.addPipeline(lsq12reg.p)
        
        #Resample using final LSQ12 transform and reset last base volume. 
        res = ma.mincresample(self.inputFH, self.targetFH, likeFile=self.targetFH, argArray=["-sinc"])   
        self.p.addStage(res)
        self.inputFH.setLastBasevol(res.outputFiles[0])
        lsq12xfm = self.inputFH.getLastXfm(self.targetFH)
        
        #Get registration parameters from nlin protocol, blur and register
        #Assume a SINGLE generation here. 
        np = mp.setOneGenMincANTSParams(self.fileRes, reg_protocol=self.nlin_protocol)
        for b in np.blurs:
            for j in b:
                #Note that blurs for ANTS params in an array of arrays. 
                if j != -1:            
                    self.p.addStage(ma.blur(self.targetFH, j, gradient=True))
                    self.p.addStage(ma.blur(self.inputFH, j, gradient=True))
                    
        sp = ma.mincANTS(self.inputFH,
                         self.targetFH,
                         defaultDir=self.defaultDir, 
                         blur=np.blurs[0],
                         gradient=np.gradient[0],
                         similarity_metric=np.similarityMetric[0],
                         weight=np.weight[0],
                         iterations=np.iterations[0],
                         radius_or_histo=np.radiusHisto[0],
                         transformation_model=np.transformationModel[0], 
                         regularization=np.regularization[0],
                         useMask=np.useMask[0])
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
        
        try: # the attempt to access the minc volume will fail if it doesn't yet exist at pipeline creation
            self.fileRes = rf.getFinestResolution(self.inputFH)
        except: 
            # if it indeed failed, get resolution from the original file specified for 
            # one of the input files, which should exist. 
            # Can be overwritten by the user through specifying a nonlinear protocol.
            self.fileRes = rf.getFinestResolution(self.inputFH.inputFileName)
        
        self.buildPipeline()
        
    def buildPipeline(self):
            
        # Do LSQ12 alignment prior to non-linear stages if desired
        if self.includeLinear: 
            lp = mp.setLSQ12MinctraccParams(self.fileRes,
                                            subject_matter=self.subject_matter,
                                            reg_protocol=self.lsq12_protocol)
            lsq12reg = lsq12.LSQ12(self.inputFH, 
                                   self.targetFH, 
                                   blurs=lp.blurs,
                                   step=lp.stepSize,
                                   gradient=lp.useGradient,
                                   simplex=lp.simplex,
                                   w_translations=lp.w_translations,
                                   defaultDir=self.defaultDir)
            self.p.addPipeline(lsq12reg.p)
        
        # create the nonlinear registrations
        np = mp.setNlinMinctraccParams(self.fileRes, reg_protocol=self.nlin_protocol)
        for b in np.blurs: 
            if b != -1:           
                self.p.addStage(ma.blur(self.inputFH, b, gradient=True))
                self.p.addStage(ma.blur(self.targetFH, b, gradient=True))
        for i in range(len(np.stepSize)):
            #For the final stage, make sure the output directory is transforms.
            if i == (len(np.stepSize) - 1):
                self.defaultDir = "transforms"
            nlinStage = ma.minctracc(self.inputFH, 
                                     self.targetFH,
                                     defaultDir=self.defaultDir,
                                     blur=np.blurs[i],
                                     gradient=np.useGradient[i],
                                     iterations=np.iterations[i],
                                     step=np.stepSize[i],
                                     w_translations=np.w_translations[i],
                                     simplex=np.simplex[i],
                                     optimization=np.optimization[i])
            self.p.addStage(nlinStage)

class LongitudinalStatsConcatAndResample:
    """ For each subject:
        1. Calculate stats (displacement, jacobians, scaled jacobians) between i and i+1 time points 
        2. Calculate transform from subject to common space (nlinFH) and invert it. 
           For most subjects this will require some amount of transform concatenation. 
        3. Calculate the stats (displacement, jacobians, scaled jacobians) from common space
           to each timepoint.
    """
    def __init__(self, subjects, timePoint, nlinFH, blurs, commonName):
        
        self.subjects = subjects
        self.timePoint = timePoint
        self.nlinFH = nlinFH
        self.blurs = blurs 
        self.commonName = commonName
        
        self.p = Pipeline()
        
        self.buildPipeline()
    
    def statsCalculation(self, inputFH, targetFH, xfm=None, useChainStats=True):
        """If useChainStats=True, calculate stats between input and target. 
           This happens for all i to i+1 calcs.
           
           If useChainStats=False, calculate stats in the standard way, from target to
           input, but without removal of average displacement. We do this, when we go
           from the common space to all others. """
        if useChainStats:
            stats = st.CalcChainStats(inputFH, targetFH, self.blurs)
        else:
            stats = st.CalcStats(inputFH, targetFH, self.blurs)
        stats.fullStatsCalc()
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
            xtc = fh.createBaseName(s[i].transformsDir, s[i].basename + "_to_" + self.commonName + ".xfm")
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
            # xfmToNlin will be either to lsq6 or native depending on other factors
            # may need an additional argument for this function
            xfmToNlin = s[self.timePoint].getLastXfm(self.nlinFH, groupIndex=0)
            count = len(s)
            if xfmToNlin:
                self.xfmToCommon = [xfmToNlin]
            else:
                self.xfmToCommon = []
            """Calculate stats first from average to timpoint included in average.
               If timepoint included in average is NOT final timepoint, also calculate
               i to i+1 stats."""
            if self.nlinFH:
                self.statsCalculation(s[self.timePoint], self.nlinFH, xfm=None, useChainStats=False)
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
 
def resampleToCommon(xfm, FH, statsGroup, blurs, nlinFH):
    pipeline = Pipeline()
    outputDirectory = FH.statsDir
    filesToResample = []
    for b in blurs:
        filesToResample.append(statsGroup.relativeJacobians[b])
        if statsGroup.absoluteJacobians:
            filesToResample.append(statsGroup.absoluteJacobians[b])
    for f in filesToResample:
        outputBase = fh.removeBaseAndExtension(f).split(".mnc")[0]
        outputFile = fh.createBaseName(outputDirectory, outputBase + "_common" + ".mnc")
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