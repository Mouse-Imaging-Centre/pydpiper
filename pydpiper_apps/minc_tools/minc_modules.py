#!/usr/bin/env python

from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
import pydpiper_apps.minc_tools.minc_atoms as ma
import pydpiper_apps.minc_tools.registration_file_handling as rfh
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

class ChainAlignLSQ6:
    def __init__(self, subjects, timePoint, lsq6Files):
        
        self.p = Pipeline()
        self.lsq6Files = lsq6Files
        
        for s in subjects:
            count = len(subjects[s])
            if timePoint - 1 < 0:
                """Average happened at time point other than first time point. 
                   Loop over points prior to average."""
                for i in reversed(range(timePoint)):
                    self.resampletoLSQ6Space(s[i-1], lsq6Files[s[i]])
            """Loop over points after average. If average is at first time point, this loop
               will hit all time points (other than first). If average is at subsequent time 
               point, it hits all time points not covered previously."""
            for i in range(timePoint, count-1):
                self.resampletoLSQ6Space(s[i+1], lsq6Files[s[i]])
        
        """After LSQ6 alignment, setLastBasevol for each to be lsq6File. Get and set xfms?"""
    
    def resampletoLSQ6Space(self, inputFH, templateFH):
        """1. Rotational Minctracc from inputFH to templateFH
           2. Resample inputFH into templateFH space
           3. Add resampledFile to lsq6 array for subsequent timepoints
        """
        rmp = RotationalMinctracc(inputFH, templateFH)
        self.p.addPipeline(rmp)
        resample = ma.mincresample(inputFH, templateFH, outputLocation=inputFH, likeFile=templateFH)
        self.p.addStage(resample)
        self.lsq6Files[inputFH] = rfh.RegistrationFHBase(resample.outputFiles[0], inputFH.subjDir)
                                       
    
#Might want to move all HierarchicalMinctracc classes to their own file
class LinearHierarchicalMinctracc:
    """Default LinearHierarchicalMinctracc class
       Assumes lsq6 registration using the identity transform"""
    def __init__(self, 
                 inputPipeFH, 
                 templatePipeFH,
                 blurs=[1, 0.5, 0.3]):
        
        self.p = Pipeline()
        self.inputPipeFH = inputPipeFH
        self.templatePipeFH = templatePipeFH
        
        self.blurFiles(blurs)
        
    def blurFiles(self, blurs):
        for b in blurs:
            if b != -1:
                tblur = ma.blur(self.templatePipeFH, b, gradient=True)
                iblur = ma.blur(self.inputPipeFH, b, gradient=True)               
                self.p.addStage(tblur)
                self.p.addStage(iblur)

class RotationalMinctracc:
    """Default RotationalMinctracc class
       Currently just calls rotational_minctracc.py 
       with minimal updates. Ultimately, we will do
       a more substantial overhaul. 
    """
    def __init__(self, 
                 inputPipeFH, 
                 templatePipeFH,
                 blurs=[0.5]):
        
        self.p = Pipeline()
        self.inputPipeFH = inputPipeFH
        self.templatePipeFH = templatePipeFH
        
        self.blurFiles(blurs) 
        for b in blurs:
            self.buildCmd(b)
        
    def blurFiles(self, blurs):
        for b in blurs:
            if b != -1:
                tblur = ma.blur(self.templatePipeFH, b, gradient=True)
                iblur = ma.blur(self.inputPipeFH, b, gradient=True)               
                self.p.addStage(tblur)
                self.p.addStage(iblur)
    
    def buildCmd(self, b):
        """Only -w_translations override rotational_minctracc.py defaults. 
           Keep this here. Rather than giving the option to override other
           defaults. We will eventually re-write this code.
        """
        w_trans = str(0.4)
        cmd = ["rotational_minctracc.py", "-t", "/dev/shm/", "-w", w_trans, w_trans, w_trans]
        source = self.inputPipeFH.getBlur(b)
        target = self.templatePipeFH.getBlur(b)
        mask = self.templatePipeFH.getMask()
        if mask:
            cmd += ["-m", InputFile(mask)]
        outputXfm = self.inputPipeFH.registerVolume(self.templatePipeFH)
        cmd +=[InputFile(source), InputFile(target), OutputFile(outputXfm), "/dev/null"]
        rm = CmdStage(cmd)
        rm.setLogFile(LogFile(fh.logFromFile(self.inputPipeFH.logDir, outputXfm)))
        self.p.addStage(rm)
        

class HierarchicalMinctracc:
    """Default HierarchicalMinctracc currently does:
        1. 2 lsq12 stages with a blur of 0.25
        2. 5 nlin stages with a blur of 0.25
        3. 1 nlin stage with no blur"""
    def __init__(self, 
                 inputPipeFH, 
                 templatePipeFH,
                 steps=[1,0.5,0.5,0.2,0.2,0.1],
                 blurs=[0.25,0.25,0.25,0.25,0.25, -1], 
                 gradients=[False, False, True, False, True, False],
                 iterations=[60,60,60,10,10,4],
                 simplexes=[3,3,3,1.5,1.5,1],
                 w_translations=0.2,
                 linearparams = {'type' : "lsq12", 'simplex' : 1, 'step' : 1}, 
                 createMask=False):
        
        self.p = Pipeline()
        
        # Set default directories based on whether or not we are creating a mask
        # Note: blurs always go in whatever tmp directory is set to
        if createMask:
            defaultDirectory = "tmp"
        else:
            defaultDirectory = "transforms"
        
        for b in blurs:
            #MF TODO: -1 case is also handled in blur. Need here for addStage.
            #Fix this redundancy and/or better design?
            if b != -1:
                tblur = ma.blur(templatePipeFH, b, gradient=True)
                iblur = ma.blur(inputPipeFH, b, gradient=True)               
                self.p.addStage(tblur)
                self.p.addStage(iblur)
            
        # Two lsq12 stages: one using 0.25 blur, one using 0.25 gradient
        for g in [False, True]:    
            linearStage = ma.minctracc(inputPipeFH, 
                                       templatePipeFH,
                                       defaultDir=defaultDirectory, 
                                       blur=blurs[0], 
                                       gradient=g,                                     
                                       linearparam=linearparams["type"],
                                       step=linearparams["step"],
                                       simplex=linearparams["simplex"],
                                       w_translations=w_translations,
                                       similarity=0.5)
            self.p.addStage(linearStage)

        # create the nonlinear registrations
        for i in range(len(steps)):
            nlinStage = ma.minctracc(inputPipeFH, 
                                     templatePipeFH,
                                     defaultDir=defaultDirectory,
                                     blur=blurs[i],
                                     gradient=gradients[i],
                                     iterations=iterations[i],
                                     step=steps[i],
                                     similarity=0.8,
                                     w_translations=w_translations,
                                     simplex=simplexes[i])
            self.p.addStage(nlinStage)
