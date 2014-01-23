#!/usr/bin/env python

from pydpiper.pipeline import Pipeline
import atoms_and_modules.LSQ12 as lsq12
import atoms_and_modules.minc_atoms as ma
import atoms_and_modules.registration_file_handling as rfh
import atoms_and_modules.stats_tools as st
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
                 lsq12Blurs=[0.3, 0.2, 0.15],
                 lsq12StepSize=[1,0.5,0.333333333333333],
                 lsq12UseGradient=[False,True,False],
                 lsq12Simplex=[3,1.5,1],
                 ANTSBlur=0.056,
                 defaultDir="tmp"):
        
        self.p = Pipeline()
        self.inputFH = inputFH
        self.targetFH = targetFH
        self.lsq12Blurs = lsq12Blurs
        self.lsq12StepSize = lsq12StepSize
        self.lsq12UseGradient = lsq12UseGradient
        self.lsq12Simplex = lsq12Simplex
        self.defaultDir = defaultDir
        self.ANTSBlur = ANTSBlur
        
        self.buildPipeline()    
    
    def buildPipeline(self):
        """Run lsq12 registration prior to non-linear"""
        lsq12reg = lsq12.LSQ12(self.inputFH, 
                            self.targetFH, 
                            blurs=self.lsq12Blurs,
                            step=self.lsq12StepSize,
                            gradient=self.lsq12UseGradient,
                            simplex=self.lsq12Simplex,
                            defaultDir=self.defaultDir)
        self.p.addPipeline(lsq12reg.p)
        """Resample input using final lsq12 transform"""
        res = ma.mincresample(self.inputFH, self.targetFH, likeFile=self.targetFH, argArray=["-sinc"])   
        self.p.addStage(res)
        self.inputFH.setLastBasevol(res.outputFiles[0])
        # MF TODO: For future implementations, perhaps keep track of the xfm
        # by creating a new registration group. Not necessary for current use,
        # but could be essential in the future.  
        lsq12xfm = self.inputFH.getLastXfm(self.targetFH)
        """Blur input and template files prior to running mincANTS command"""
        tblur = ma.blur(self.targetFH, self.ANTSBlur, gradient=True)
        iblur = ma.blur(self.inputFH, self.ANTSBlur, gradient=True)               
        self.p.addStage(tblur)
        self.p.addStage(iblur)
        sp = ma.mincANTS(self.inputFH,
                         self.targetFH,
                         defaultDir=self.defaultDir, 
                         blur=[-1, self.ANTSBlur])
        self.p.addStage(sp)
        nlinXfm = sp.outputFiles[0]
        """Reset last base volume to original input for future registrations."""
        self.inputFH.setLastBasevol(setToOriginalInput=True)
        """Concatenate transforms to get final lsq12 + nlin. Register volume handles naming and setting of lastXfm"""
        #MF TODO: May want to change the output name to include a "concat" to indicate lsq12 and nlin concatenation?
        output = self.inputFH.registerVolume(self.targetFH, "transforms")
        xc = ma.xfmConcat([lsq12xfm, nlinXfm], output, fh.logFromFile(self.inputFH.logDir, output))
        self.p.addStage(xc)

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
        self.xfmToCommon.insert(0, xfm)
        """ Concat transforms to get xfmToCommon and calculate statistics 
            Note that inverted transform, which is what we want, is calculated in
            the statistics module. """
        xtc = fh.createBaseName(s[i].transformsDir, s[i].basename + "_to_" + self.commonName + ".xfm")
        xc = ma.xfmConcat(self.xfmToCommon, xtc, fh.logFromFile(s[i].logDir, xtc))
        self.p.addStage(xc)
        """Set this transform as last xfm from input to nlin and calculate nlin to s[i] stats"""
        s[i].addAndSetXfmToUse(self.nlinFH, xtc)
        self.statsCalculation(s[i], self.nlinFH, xfm=None, useChainStats=False)
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
        filesToResample.append(statsGroup.jacobians[b])
        if statsGroup.scaledJacobians:
            filesToResample.append(statsGroup.scaledJacobians[b])
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