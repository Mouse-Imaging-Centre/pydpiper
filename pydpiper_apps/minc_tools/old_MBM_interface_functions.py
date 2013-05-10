#!/usr/bin/env python

from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
import pydpiper.file_handling as fh
import pydpiper_apps.minc_tools.registration_file_handling as rfh
import pydpiper_apps.minc_tools.minc_atoms as ma
import pydpiper_apps.minc_tools.stats_tools as st
import Pyro
from os.path import abspath
from os import walk
import fnmatch
import logging
import sys

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

def getXfms(nlinFH, subjects, space, mbmDir, time=None):

    """For each file in the build-model registration (associated with the specified
       time point), do the following:
       
       1. Find the to-native.xfm for that file. 
       2. Find the matching subject at the specified time point
       3. Set this xfm to be the last xfm from nlin average to subject from step #2. 
       4. Find the -from-native.xfm file.
       5. Set this xfm to be the last xfm from subject to nlin.
       
       Note: assume that the names in processedDir match beginning file 
             names for each subject
             We are also assuming subjects is either a dictionary or a list. 
    """
    
    """First handle subjects if dictionary or list"""
    if isinstance(subjects, list):
        inputs = subjects
    elif isinstance(subjects, dict):
        inputs = []
        for s in subjects:
            inputs.append(subjects[s][time])
    else:
        logger.error("getXfms only takes a dictionary or list of subjects. Incorrect type has been passed. Exiting...")
        sys.exit()
    
    pipeline = Pipeline()
    baseNames = walk(mbmDir).next()[1]
    for b in baseNames:
        if space == "lsq6":
            xfmToNative = abspath(mbmDir + "/" + b + "/transforms/" + b + "-final-to_lsq6.xfm")
        elif space == "lsq12":
            xfmToNative = abspath(mbmDir + "/" + b + "/transforms/" + b + "-final-nlin.xfm")
            xfmFromNative = abspath(mbmDir + "/" + b + "/transforms/" + b + "_inv_nonlinear.xfm")
        elif space == "native":
            xfmToNative = abspath(mbmDir + "/" + b + "/transforms/" + b + "-to-native.xfm")
            xfmFromNative = abspath(mbmDir + "/" + b + "/transforms/" + b + "-from-native.xfm")
        else:
            logger.error("getXfms can only retrieve transforms to and from native, lsq6 or lsq12 space. Invalid parameter has been passed.")
            sys.exit()
        for inputFH in inputs:
            if fnmatch.fnmatch(inputFH.getLastBasevol(), "*" + b + "*"):
                if space=="lsq6":
                    invXfmBase = fh.removeBaseAndExtension(xfmToNative).split("-final-to_lsq6")[0]
                    xfmFromNative = fh.createBaseName(inputFH.transformsDir, invXfmBase + "_lsq6-to-final.xfm")
                    cmd = ["xfminvert", "-clobber", InputFile(xfmToNative), OutputFile(xfmFromNative)]
                    invertXfm = CmdStage(cmd)
                    invertXfm.setLogFile(LogFile(fh.logFromFile(inputFH.logDir, xfmFromNative)))
                    pipeline.addStage(invertXfm)
                nlinFH.setLastXfm(inputFH, xfmToNative)
                inputFH.setLastXfm(nlinFH, xfmFromNative)
    return pipeline
                
def getLsq6Files(mbmDir, subjects, time, processedDirectory):
    """For each subject, find the lsq6 file in the specified directory"""
    lsq6Files = {}
    baseNames = walk(mbmDir).next()[1]
    for b in baseNames:
        lsq6Resampled = abspath(mbmDir + "/" + b + "/resampled/" + b + "-lsq6.mnc")
        for s in subjects:
            lsq6Files[subjects[s][time]] = rfh.RegistrationFHBase(lsq6Resampled, processedDirectory)
    return lsq6Files

class LongitudinalStatsConcatAndResample:
    """ For each subject:
        1. Calculate stats (displacement, jacobians, scaled jacobians) between i and i+1 time points 
        2. Concatenate transforms between ith time point and average, and ith time point and common space.
        3. Calculate stats between ith timePoint and average time point. 
        4. Resample all stats into common space. (i to i+1 and i to average time point)
    """
    def __init__(self, subjects, timePoint, nlinFH, blurs):
        
        self.subjects = subjects
        self.timePoint = timePoint
        self.nlinFH = nlinFH
        self.blurs = blurs 
        
        self.p = Pipeline()
        
        self.buildPipeline()
    
    def statsAndResample(self, inputFH, targetFH, xfm):
        """Calculate stats between input and target, resample to common space using xfm"""
        stats = st.CalcChainStats(inputFH, targetFH, self.blurs)
        stats.calcFullDisplacement()
        stats.calcDetAndLogDet(useFullDisp=True)
        self.p.addPipeline(stats.p)
        res = resampleToCommon(xfm, inputFH, stats.statsGroup, self.blurs, self.nlinFH)
        self.p.addPipeline(res)
    
    def buildXfmArrays(self, inputFH, targetFH):
        xfm = inputFH.getLastXfm(targetFH)
        self.xfmToCommon.insert(0, xfm)
        self.xfmToAvg.insert(0, xfm)
        self.xtc = self.returnConcattedXfm(inputFH, self.xfmToCommon, fh.createBaseName(inputFH.transformsDir, "xfm_to_common_space.xfm"))
        
    def returnConcattedXfm(self, FH, xfmArray, outputName):
        xcs = concatXfm(FH, xfmArray, outputName)
        self.p.addStage(xcs)
        return xcs.outputFiles[0]
    
    def nonAdjacentTimePtToAvg(self, inputFH, targetFH):
        if len(self.xfmToAvg) > 1: 
            outputName = inputFH.registerVolume(targetFH, "transforms")
            xta = self.returnConcattedXfm(inputFH, self.xfmToAvg, outputName)
        """Resample input to average"""
        resample = ma.mincresample(inputFH, targetFH, likeFile=self.nlinFH)
        self.p.addStage(resample)
        """Calculate stats from input to target and resample to common space"""
        self.statsAndResample(inputFH, targetFH, self.xtc)
        
    def buildPipeline(self):
        for subj in self.subjects:
            s = self.subjects[subj]
            # xfmToNlin will be either to lsq6 or native depending on other factors
            # may need an additional argument for this function
            xfmToNlin = s[self.timePoint].getLastXfm(self.nlinFH, groupIndex=0)
            count = len(s)
            self.xfmToCommon = [xfmToNlin]
            self.xfmToAvg = []
            """Do timePoint with average first if average is not final subject"""
            if count - self.timePoint > 1:
                self.statsAndResample(s[self.timePoint], s[self.timePoint+1], xfmToNlin)
            if not self.timePoint - 1 < 0:
                """ Average happened at time point other than first time point. 
                    Loop over points prior to average."""
                for i in reversed(range(self.timePoint)): 
                    """Create transform arrays, concat xfmToCommon, calculate stats and resample """
                    self.buildXfmArrays(s[i], s[i+1])
                    self.statsAndResample(s[i], s[i+1], self.xtc)
                    if self.timePoint - i > 1:
                        """For timePoints not directly adjacent to average, calc stats to average."""
                        self.nonAdjacentTimePtToAvg(s[i], s[self.timePoint])
                                     
            """ Loop over points after average. If average is at first time point, this loop
                will hit all time points (other than first). If average is at subsequent time 
                point, it hits all time points not covered previously."""
            self.xfmToCommon=[xfmToNlin]  
            self.xfmToAvg = []  
            for i in range(self.timePoint + 1, count-1):
                """Create transform arrays, concat xfmToCommon, calculate stats and resample """
                self.buildXfmArrays(s[i], s[i-1])
                self.statsAndResample(s[i], s[i+1], self.xtc)
                if i - self.timePoint > 1:
                    """For timePoints not directly adjacent to average, calc stats to average."""
                    self.nonAdjacentTimePtToAvg(s[i], s[self.timePoint])
            
            """ Handle final time point as special case , since it will not have normal stats calc
                Only do this step if final time point is not also average time point """
            if count - self.timePoint > 1:
                self.buildXfmArrays(s[count-1], s[count-2])
                self.nonAdjacentTimePtToAvg(s[count-1], s[self.timePoint])  

def concatXfm(FH, xfmArray, output):
    # Note: we will want to move this to a more general file. 
    cmd = ["xfmconcat", "-clobber"] + [InputFile(a) for a in xfmArray] + [OutputFile(output)]
    xfmConcat = CmdStage(cmd)
    xfmConcat.setLogFile(LogFile(fh.logFromFile(FH.logDir, output)))
    return xfmConcat
    
    
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
        res = ma.mincresample(f, 
                              nlinFH.getLastBasevol(),
                              likeFile=nlinFH.getLastBasevol(),
                              transform=xfm,
                              outFile=outputFile,
                              logFile=logFile) 
        pipeline.addStage(res)
    
    return pipeline