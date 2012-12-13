#!/usr/bin/env python

from pydpiper.pipeline import CmdStage, Pipeline, InputFile, OutputFile, LogFile
import pydpiper.file_handling as fh
from pydpiper_apps.minc_tools.minc_modules import HierarchicalMinctracc
import pydpiper_apps.minc_tools.minc_atoms as ma
import Pyro
import logging

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

def maskFiles(FH, isAtlas, numAtlases=1):
    """ Assume that if there is more than one atlas, multiple
        masks were generated and we need to perform a voxel_vote. 
        Otherwise, assume we are using inputLabels from crossing with
        only one atlas. 
    """
    #MF TODO: Make this more general to handle pairwise option. 
    p = Pipeline()
    if not isAtlas:
        if numAtlases > 1:
            voxel = voxelVote(FH, False, True)
            p.addStage(voxel)
            mincMathInput = voxel.outputFiles[0]  
        else:
            mincMathInput = FH.returnLabels(True)[0]
        FH.setMask(mincMathInput)
    else:
        mincMathInput = FH.getMask()
    mincMathOutput = fh.createBaseName(FH.resampledDir, FH.basename)
    mincMathOutput += "_masked.mnc"   
    logFile = fh.logFromFile(FH.logDir, mincMathOutput)
    cmd = ["mincmath"] + ["-clobber"] + ["-mult"]
    cmd += [InputFile(mincMathInput)] + [InputFile(FH.getLastBasevol())] 
    cmd += [OutputFile(mincMathOutput)]
    mincMath = CmdStage(cmd)
    mincMath.setLogFile(LogFile(logFile))
    p.addStage(mincMath)
    FH.setLastBasevol(mincMathOutput)
    return(p)

def voxelVote(inputFH, pairwise, mask):
    # if we do pairwise crossing, use output labels for voting (Default)
    # otherwise, return inputLabels from initial atlas-input crossing
    useInputLabels = False
    if not pairwise:
        useInputLabels = True
    labels = inputFH.returnLabels(useInputLabels)
    out = fh.createBaseName(inputFH.labelsDir, inputFH.basename)
    if mask:
        out += "_mask.mnc"
    else: 
        out += "_votedlabels.mnc"
    logFile = fh.logFromFile(inputFH.logDir, out)
    cmd = ["voxel_vote.py"] + [InputFile(l) for l in labels] + [OutputFile(out)]
    voxel = CmdStage(cmd)
    voxel.setLogFile(LogFile(logFile))
    return(voxel)

def maskDirectoryStructure(FH, masking=True):
    if masking:
        FH.tmpDir = fh.createSubDir(FH.subjDir, "masking")
        FH.logDir = fh.createLogDir(FH.tmpDir)
    else:
        FH.tmpDir = fh.createSubDir(FH.subjDir, "tmp")
        FH.logDir = fh.createLogDir(FH.subjDir)
    
def MAGeTMask(atlases, inputs, numAtlases, regMethod):
    """ Masking algorithm is as follows:
        1. Run HierarchicalMinctracc or mincANTS with mask=True, 
           using masks instead of labels. 
        2. Do voxel voting to find the best mask. (Or, if single atlas,
            use that transform)
        3. mincMath to multiply original input by mask to get _masked.mnc file
            (This is done for both atlases and inputs, though for atlases, voxel
             voting is not required.)
        4. Replace lastBasevol with masked version, since once we have created
            mask, we no longer care about unmasked version. 
        5. Clear out labels arrays, which were used to keep track of masks,
            as we want to re-set them for actual labels.
                
        Note: All data will be placed in a newly created masking directory
        to keep it separate from data generated during actual MAGeT. 
        """
    p = Pipeline()
    for atlasFH in atlases:
        maskDirectoryStructure(atlasFH, masking=True)
    for inputFH in inputs:
        maskDirectoryStructure(inputFH, masking=True)
        for atlasFH in atlases:
            sp = MAGeTRegister(inputFH, 
                               atlasFH, 
                               regMethod, 
                               name="initial", 
                               createMask=True)
            p.addPipeline(sp)          
    """ Prior to final masking, set log and tmp directories as they were."""
    for atlasFH in atlases:
        """Retrieve labels for use in new group. Assume only one"""
        labels = atlasFH.returnLabels(True)
        maskDirectoryStructure(atlasFH, masking=False)
        mp = maskFiles(atlasFH, True)
        p.addPipeline(mp)
        atlasFH.newGroup()
        atlasFH.addLabels(labels[0], inputLabel=True)
    for inputFH in inputs:
        maskDirectoryStructure(inputFH, masking=False)
        mp = maskFiles(inputFH, False, numAtlases)
        p.addPipeline(mp)
        inputFH.clearLabels(True)
        inputFH.clearLabels(False) 
        inputFH.newGroup()  
    return(p)    

def MAGeTRegister(inputFH, 
                  templateFH, 
                  method,
                  name="initial", 
                  createMask=False):
    p = Pipeline()
    if method == "minctracc":
        sp = HierarchicalMinctracc(inputFH, 
                                   templateFH, 
                                   createMask=createMask)
        p.addPipeline(sp.p)
    elif method == "mincANTS":
        if createMask:
            defaultDir="tmp"
        else:
            defaultDir="transforms"
        b = 0.056
        tblur = ma.blur(templateFH, b, gradient=True)
        iblur = ma.blur(inputFH, b, gradient=True)               
        p.addStage(tblur)
        p.addStage(iblur)
        sp = ma.mincANTS(inputFH,
                      templateFH,
                      defaultDir=defaultDir,
                      similarity_metric=["MI", "CC"],
                      iterations="100x100x50x50",
                      radius_or_histo=[32,3],
                      transformation_model="SyN[0.1]", 
                      regularization="Gauss[3,1]")
        p.addStage(sp)
    
    rp = LabelAndFileResampling(inputFH, templateFH, name=name, createMask=createMask)
    p.addPipeline(rp.p)
    
    return(p)

class LabelAndFileResampling:
    def __init__(self, 
                 inputPipeFH,
                 templatePipeFH, 
                 name="initial", 
                 createMask=False):  
        self.p = Pipeline()
        self.name = name
        
        if createMask:
            resampleDefault = "tmp"
            labelsDefault = "tmp"
        else:
            resampleDefault = "resampled"
            labelsDefault = "labels"
              
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
                """Note: templatePipeFH and inputPipeFH have the reverse order
                   from how they are passed into this function. This is intentional
                   because the mincresample classes use the first argument as the 
                   one from which to get the file to be resampled. Here, either the 
                   mask or labels to be resampled come from the template."""
                if createMask:
                    resampleStage = ma.mincresampleMask(templatePipeFH,
                                                        inputPipeFH,
                                                        defaultDir=labelsDefault,
                                                        likeFile=inputPipeFH,
                                                        argArray=["-invert"],
                                                        outputLocation=inputPipeFH,
                                                        labelIndex=i,
                                                        setInputLabels=addOutputToInputLabels)
                else:
                    resampleStage = ma.mincresampleLabels(templatePipeFH,
                                                          inputPipeFH, 
                                                          defaultDir=labelsDefault,
                                                          likeFile=inputPipeFH,
                                                          argArray=["-invert"],
                                                          outputLocation=inputPipeFH,
                                                          labelIndex=i,
                                                          setInputLabels=addOutputToInputLabels)
                self.p.addStage(resampleStage)
            # resample files
            resampleStage = ma.mincresample(templatePipeFH,
                                            inputPipeFH,
                                            defaultDir=resampleDefault,
                                            likeFile=inputPipeFH,
                                            argArray=["-invert"],
                                            outputLocation=inputPipeFH)
            self.p.addStage(resampleStage)