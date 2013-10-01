#!/usr/bin/env python

from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
import pydpiper_apps.minc_tools.minc_atoms as ma
import pydpiper_apps.minc_tools.registration_file_handling as rfh
import pydpiper_apps.minc_tools.stats_tools as st
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
                 ANTSBlur=0.056,
                 defaultDir="tmp"):
        
        self.p = Pipeline()
        self.inputFH = inputFH
        self.targetFH = targetFH
        self.lsq12Blurs = lsq12Blurs
        self.defaultDir = defaultDir
        self.ANTSBlur = ANTSBlur
        
        self.buildPipeline()    
    
    def buildPipeline(self):
        """Run lsq12 registration prior to non-linear"""
        lsq12 = LSQ12(self.inputFH, 
                      self.targetFH, 
                      blurs=self.lsq12Blurs,
                      defaultDir=self.defaultDir)
        self.p.addPipeline(lsq12.p)
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
        self.inputFH.setLastBasevol()
        """Concatenate transforms to get final lsq12 + nlin. Register volume handles naming and setting of lastXfm"""
        #MF TODO: May want to change the output name to include a "concat" to indicate lsq12 and nlin concatenation?
        output = self.inputFH.registerVolume(self.targetFH, "transforms")
        cmd = ["xfmconcat", "-clobber"] + [InputFile(lsq12xfm)] + [InputFile(nlinXfm)] + [OutputFile(output)]
        xfmConcat = CmdStage(cmd)
        xfmConcat.setLogFile(LogFile(fh.logFromFile(self.inputFH.logDir, output)))
        self.p.addStage(xfmConcat)

class LSQ12:
    """Basic LSQ12 class. 
    
    This class takes an input FileHandler and a targetFileHandler as required inputs. A series of
    minctracc calls will then produce the 12-parameter alignment. The number of minctracc calls 
    and their parameters are controlled by three further arguments to the constructor:
    
    blurs: an array of floats containing the FWHM of the blurring kernel to be used for each call
    step: an array of floats containing the step used by minctracc in each call
    simplex: an array of floats containing the simplex used by minctracc in each call.
    
    The number of entries in those three (blurs, step, simplex) input arguments determines the number
    of minctracc calls executed in this module. For example, the following call:
    LSQ12(inputFH, targetFH, blurs=[10,5,2], step=[4,4,4], simplex=[20,20,20])
    will result in three successive minctracc calls, each initialized with the output transform of the 
    previous call.
    """
    def __init__(self,
                 inputFH,
                 targetFH, 
                 blurs=[0.3, 0.2, 0.15], 
                 step=[1,0.5,0.333333333333333],
                 simplex=[3,1.5,1],
                 defaultDir="tmp"):                                      

        if len(blurs) == len(step) == len(simplex):
            # do nothing - all lengths are the same and we're therefore happy
            pass
        else:
            logger.error("The same number of entries are required for blurs, step, and simplex in LSQ12")
            sys.exit()
                
        self.p = Pipeline()
        self.inputFH = inputFH
        self.targetFH = targetFH
        self.blurs = blurs
        self.step = step
        self.blurs = blurs
        self.simplex = simplex
        self.defaultDir = defaultDir
            
        self.blurFiles()
        self.buildPipeline()
    
    def blurFiles(self):
        for b in self.blurs:
            if b != -1:
                tblur = ma.blur(self.targetFH, b, gradient=True)
                iblur = ma.blur(self.inputFH, b, gradient=True)               
                self.p.addStage(tblur)
                self.p.addStage(iblur)
        
    def buildPipeline(self):
        gradient=[False,True,False]
        for i in range(len(self.blurs)):    
            linearStage = ma.minctracc(self.inputFH, 
                                       self.targetFH, 
                                       blur=self.blurs[i], 
                                       defaultDir=self.defaultDir,
                                       gradient=gradient[i],                                     
                                       linearparam="lsq12",
                                       step=self.step[i],
                                       simplex=self.simplex[i])
            self.p.addStage(linearStage)

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
        self.xtcDict = {}
        
        self.buildPipeline()
    
    def statsAndResample(self, inputFH, targetFH, xfm):
        """Calculate stats between input and target, resample to common space using xfm"""
        stats = st.CalcChainStats(inputFH, targetFH, self.blurs)
        stats.fullStatsCalc()
        self.p.addPipeline(stats.p)
        """Only resampleToCommon space if we have the appropriate transform"""
        if xfm:
            if not self.nlinFH:
                likeFH = targetFH
            else:
                likeFH = self.nlinFH
            res = resampleToCommon(xfm, inputFH, stats.statsGroup, self.blurs, likeFH)
            self.p.addPipeline(res)
    
    def buildXfmArrays(self, inputFH, targetFH):
        xfm = inputFH.getLastXfm(targetFH)
        self.xfmToCommon.insert(0, xfm)
        self.xfmToAvg.insert(0, xfm)
    
    def nonAdjacentTimePtToAvg(self, inputFH, targetFH):
        if len(self.xfmToAvg) > 1: 
            if not inputFH.getLastXfm(targetFH):
                outputName = inputFH.registerVolume(targetFH, "transforms")
                self.p.addStage(concatXfm(inputFH, self.xfmToAvg, outputName))
        """Resample input to average"""
        if not self.nlinFH:
            likeFH = targetFH
        else:
            likeFH = self.nlinFH
        resample = ma.mincresample(inputFH, targetFH, likeFile=likeFH, argArray=["-sinc"])
        self.p.addStage(resample)
        """Calculate stats from input to target and resample to common space"""
        self.statsAndResample(inputFH, targetFH, self.xtcDict[inputFH])
        
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
                    self.xtcDict[s[i]] = fh.createBaseName(s[i].transformsDir, "xfm_to_common_space.xfm")
                    self.p.addStage(concatXfm(s[i], self.xfmToCommon, self.xtcDict[s[i]]))
                    self.statsAndResample(s[i], s[i+1], self.xtcDict[s[i]])
                    if self.timePoint - i > 1:
                        """For timePoints not directly adjacent to average, calc stats to average."""
                        self.nonAdjacentTimePtToAvg(s[i], s[self.timePoint])
                                     
            """ Loop over points after average. If average is at first time point, this loop
                will hit all time points (other than first). If average is at subsequent time 
                point, it hits all time points not covered previously."""
            if xfmToNlin:
                self.xfmToCommon = [xfmToNlin]
            else:
                self.xfmToCommon = [] 
            self.xfmToAvg = []  
            for i in range(self.timePoint + 1, count-1):
                """Create transform arrays, concat xfmToCommon, calculate stats and resample """
                self.buildXfmArrays(s[i], s[i-1])
                self.xtcDict[s[i]] = fh.createBaseName(s[i].transformsDir, "xfm_to_common_space.xfm")
                self.p.addStage(concatXfm(s[i], self.xfmToCommon, self.xtcDict[s[i]]))
                self.statsAndResample(s[i], s[i+1], self.xtcDict[s[i]])
                if i - self.timePoint > 1:
                    """For timePoints not directly adjacent to average, calc stats to average."""
                    self.nonAdjacentTimePtToAvg(s[i], s[self.timePoint])
            
            """ Handle final time point as special case , since it will not have normal stats calc
                Only do this step if final time point is not also average time point """
            if count - self.timePoint > 1:
                self.buildXfmArrays(s[count-1], s[count-2])
                self.xtcDict[s[count-1]] = fh.createBaseName(s[count-1].transformsDir, "xfm_to_common_space.xfm")
                self.p.addStage(concatXfm(s[count-1], self.xfmToCommon, self.xtcDict[s[count-1]]))
                self.nonAdjacentTimePtToAvg(s[count-1], s[self.timePoint])  
                
            """Calculate stats for first time point to all others. 
               Note that stats from first to second time points should have been done previously"""
            self.xfmToAvg = [s[0].getLastXfm(s[1])]
            for i in range(1, count-1):
                self.xfmToAvg.append(s[i].getLastXfm(s[i+1]))
                self.nonAdjacentTimePtToAvg(s[0], s[i+1])

def concatXfm(FH, xfmArray, output): 
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
        likeFile=nlinFH.getLastBasevol()
        res = ma.mincresample(f, 
                              nlinFH.getLastBasevol(),
                              likeFile=nlinFH.getLastBasevol(),
                              transform=xfm,
                              output=outputFile,
                              logFile=logFile,
                              argArray=["-sinc"]) 
        pipeline.addStage(res)
    
    return pipeline