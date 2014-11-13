#!/usr/bin/env python

from pydpiper.pipeline import CmdStage, Pipeline, InputFile, OutputFile, LogFile
import pydpiper.file_handling as fh
from atoms_and_modules.minc_modules import LSQ12ANTSNlin, HierarchicalMinctracc
import atoms_and_modules.minc_atoms as ma
from argparse import ArgumentGroup
import logging

logger = logging.getLogger(__name__)

def addMAGeTArgumentGroup(parser):
    group = ArgumentGroup(parser, "MAGeT options",
                          "Options for running MAGeT.")
    group.add_argument("--atlas-library", dest="atlas_lib",
                       type="string", default="atlas_label_pairs",
                       help="Directory of existing atlas/label pairs")
    group.add_argument("--no-pairwise", dest="pairwise",
                       action="store_false", default=True,
                       help="""Pairwise crossing of templates. [Default = %default]. If specified, only register inputs to atlases in library""")
    group.add_argument("--mask", dest="mask",
                       action="store_true", default=False,
                       help="Create a mask for all images prior to handling labels. [Default = %default]")
    group.add_argument("--mask-only", dest="mask_only",
                       action="store_true", default=False,
                       help="Create a mask for all images only, do not run full algorithm. [Default = %default]")
    group.add_argument("--max-templates", dest="max_templates",
                       default=25, type="int",
                       help="Maximum number of templates to generate. [Default = %default]")
    group.add_argument("--masking-method", dest="mask_method",
                       default="minctracc", type="string",
                       help="Specify whether to use minctracc or mincANTS for masking. [Default = %default].")
    parser.add_argument_group(group)

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
    # In response to issue #135
    # the order of the input files to mincmath matters. By default the
    # first input files is used as a "like file" for the output file. 
    # We should make sure that the mask is not used for that, because
    # it has an image range from 0 to 1; not something we want to be
    # set for the masked output file
    #            average                              mask
    cmd += [InputFile(FH.getLastBasevol())] + [InputFile(mincMathInput)]
    cmd += [OutputFile(mincMathOutput)]
    mincMath = CmdStage(cmd)
    mincMath.setLogFile(LogFile(logFile))
    p.addStage(mincMath)
    FH.setLastBasevol(mincMathOutput)
    return(p)

def voxelVote(inputFH, pairwise, mask):
    # In the main MAGeT.py code, when not only a mask is created for the
    # input files, the process works as follows:
    # 
    # 1) the template files (library) are aligned to each input upto max_templates input files
    # 2) all templates (library + newly created) are aligned to each input
    # 
    # That second stage contains alignments that have already run in the first stage.
    # And pydpiper is coded such, that this duplicated stage is not performed. In order
    # to get all labels for voxel voting, we need to combine atlases from both these 
    # stages, i.e., the "initial" and the "templates". This means that we should always
    # get the "useInputLabels". (In the special case where there is only 1 input file 
    # and pairwise is set to true, this is particularly important, because of the duplicate
    # stages, only the inputlabels will exists.)
    
    # 1) get the input templates
    # the True parameter will return "inputLabels" from the groupedFiles for inputFH
    labels = inputFH.returnLabels(True)
    
    # 2) if we do pairwise crossing, also get the output labels for voting 
    if pairwise:
        # False will return "labels" from the groupedFiles for inputFH
        outputLabels = inputFH.returnLabels(False)
        # add these labels to the "initial" or input labels:
        labels = labels + outputLabels
    
    out = fh.createBaseName(inputFH.labelsDir, inputFH.basename)
    if mask:
        out += "_mask.mnc"
    else: 
        out += "_votedlabels.mnc"
    logFile = fh.logFromFile(inputFH.logDir, out)
    cmd = ["voxel_vote"] + [InputFile(l) for l in labels] + [OutputFile(out)]
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
    
def MAGeTMask(atlases, inputs, numAtlases, regMethod, lsq12_protocol=None, nlin_protocol=None):
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
                               createMask=True,
                               lsq12_protocol=lsq12_protocol,
                               nlin_protocol=nlin_protocol)
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
        # this will remove the "inputLabels"; labels that
        # come directly from the atlas library
        inputFH.clearLabels(True)
        # this will remove the "labels"; second generation
        # labels. I.e. labels from labels from the atlas library
        inputFH.clearLabels(False) 
        inputFH.newGroup()  
    return(p)    

def MAGeTRegister(inputFH, 
                  templateFH, 
                  regMethod,
                  name="initial", 
                  createMask=False,
                  lsq12_protocol=None,
                  nlin_protocol=None):
    
    p = Pipeline()
    if createMask:
        defaultDir="tmp"
    else:
        defaultDir="transforms"
    if regMethod == "minctracc":
        sp = HierarchicalMinctracc(inputFH, 
                                   templateFH,
                                   lsq12_protocol=lsq12_protocol,
                                   nlin_protocol=nlin_protocol,
                                   defaultDir=defaultDir)
        p.addPipeline(sp.p)
    elif regMethod == "mincANTS":
        register = LSQ12ANTSNlin(inputFH, 
                                 templateFH, 
                                 lsq12_protocol=lsq12_protocol,
                                 nlin_protocol=nlin_protocol,
                                 defaultDir=defaultDir)
        p.addPipeline(register.p)
        
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
