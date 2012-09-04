#!/usr/bin/env python

from pydpiper.pipeline import Pipeline
import pydpiper_apps.minc_tools.minc_atoms as ma
import pydpiper_apps.minc_tools.registration_file_handling as rfh
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
            if currentRes[0] != resolution:
                crop = ma.autocrop(resolution, FH, defaultDir=dirForOutput)
                self.p.addStage(crop)
        
    def getOutputDirectory(self, FH):
        """Sets output directory based on whether or not we have a full
        RegistrationPipeFH class or we are just using RegistrationFHBase"""
        if isinstance(FH, rfh.RegistrationPipeFH):
            outputDir = "resampled"
        else:
            outputDir = FH.basedir
        return outputDir
        
class HierarchicalMinctracc:
    def __init__(self, inputPipeFH, 
                 templatePipeFH,
                 steps=[1,0.5,0.5,0.2,0.2,0.1],
                 blurs=[0.25,0.25,0.25,0.25,0.25, -1], 
                 gradients=[False, False, True, False, True, False],
                 iterations=[60,60,60,10,10,4],
                 simplexes=[3,3,3,1.5,1.5,1],
                 w_translations=0.2,
                 linearparams = {'type' : "lsq12", 'simplex' : 1, 'step' : 1},
                 name="initial", 
                 createMask=False):
        self.p = Pipeline()
        self.name = name
        
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
                                  blur=blurs[i],
                                  gradient=gradients[i],
                                  iterations=iterations[i],
                                  step=steps[i],
                                  similarity=0.8,
                                  w_translations=w_translations,
                                  simplex=simplexes[i])
            self.p.addStage(nlinStage)
        
        
        # Resample all inputLabels 
        inputLabelArray = templatePipeFH.returnLabels(True)
        if len(inputLabelArray) > 0:
            """ for the initial registration, resulting labels should be added
                to inputLabels array for subsequent pairwise registration
                otherwise labels should be added to labels array for voting """
            if self.name == "initial":
                addOutputToInputLabels = True
            else:
                addOutputToInputLabels = False
            for i in range(len(inputLabelArray)):
                resampleStage = ma.mincresampleLabels(templatePipeFH,
                                                    likeFile=inputPipeFH,
                                                    argArray=["-invert"],
                                                    labelIndex=i,
                                                    setInputLabels=addOutputToInputLabels,
                                                    mask=createMask)
                self.p.addStage(resampleStage)
            # resample files
            resampleStage = ma.mincresample(templatePipeFH,
                                            likeFile=inputPipeFH,
                                            argArray=["-invert"])
            self.p.addStage(resampleStage)