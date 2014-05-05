#!/usr/bin/env python

import atoms_and_modules.registration_file_handling as rfh
import pydpiper.file_handling as fh
from optparse import OptionGroup
from os.path import abspath, exists, dirname, splitext, isfile
from os import curdir, walk
from datetime import date
import sys
import re
import logging
import fnmatch
from pyminc.volumes.factory import volumeFromFile

logger = logging.getLogger(__name__)

def addGenRegOptionGroup(parser):
    group = OptionGroup(parser, "General registration options",
                        "General options for running various types of registrations.")
    group.add_option("--pipeline-name", dest="pipeline_name",
                      type="string", default=None,
                      help="Name of pipeline and prefix for models.")
    group.add_option("--registration-method", dest="reg_method",
                      default="minctracc", type="string",
                      help="Specify whether to use minctracc or mincANTS for non-linear registrations. "
                           "Default is minctracc.")
    group.add_option("--mask-dir", dest="mask_dir",
                      type="string", default=None, 
                      help="Directory of masks. If not specified, no masks are used. If only one mask in directory, same mask used for all inputs.")
    parser.add_option_group(group)

class StandardMBMDirectories(object):
    def __init__(self):
        self.lsq6Dir = None
        self.lsq12Dir = None
        self.nlinDir = None
        self.processedDir = None

def setupDirectories(outputDir, pipeName, module):
    #Setup pipeline name
    if not pipeName:
        pipeName = str(date.today()) + "_pipeline"
    
    #initilize directories class:
    dirs = StandardMBMDirectories() 
    
    #create subdirectories based on which module is being run. _processed always created 
    dirs.processedDir = fh.createSubDir(outputDir, pipeName + "_processed")
    
    if (module == "ALL") or (module=="LSQ6"):
        dirs.lsq6Dir = fh.createSubDir(outputDir, pipeName + "_lsq6")
    if (module == "ALL") or (module=="LSQ12"):
        dirs.lsq12Dir = fh.createSubDir(outputDir, pipeName + "_lsq12")
    if (module == "ALL") or (module=="NLIN"):
        dirs.nlinDir = fh.createSubDir(outputDir, pipeName + "_nlin")
    
    return dirs
    

"""
    the "args" argument to this function must be a list.  If only
    one argument is supplied, make sure you don't pass it as a string 
"""
def initializeInputFiles(args, mainDirectory, maskDir=None):
    
    # initial error handling:  verify that at least one input file is specified 
    # and that it is a MINC file
    if(len(args) < 1):
        print "Error: no source image provided\n"
        sys.exit()
    for i in range(len(args)):
        ext = splitext(args[i])[1]
        if(re.match(".mnc", ext) == None):
            print "Error: input file is not a MINC file:, ", args[i], "\n"
            sys.exit()
    
    inputs = []
    # the assumption in the following line is that args is a list
    # if that is not the case, convert it to one
    if(not(type(args) is list)):
        args = [args]
    for iFile in range(len(args)):
        inputPipeFH = rfh.RegistrationPipeFH(abspath(args[iFile]), basedir=mainDirectory)
        inputs.append(inputPipeFH)
    """After file handlers initialized, assign mask to each file
       If directory of masks is specified, apply to each file handler.
       Two options:
            1. One mask in directory --> use for all scans. 
            2. Same number of masks as files, with same naming convention. Individual
                 mask for each scan.  
    """
    if maskDir:
        absMaskPath = abspath(maskDir)
        masks = walk(absMaskPath).next()[2]
        numMasks = len(masks)
        numScans = len(inputs)
        if numMasks == 1:
                for inputFH in inputs:
                    inputFH.setMask(absMaskPath + "/" + masks[0])
        elif numMasks == numScans:
            for m in masks:
                maskBase = fh.removeBaseAndExtension(m).split("_mask")[0]
                for inputFH in inputs:
                    if fnmatch.fnmatch(inputFH.getLastBasevol(), "*" + maskBase + "*"):
                        inputFH.setMask(absMaskPath + "/" + m)
        else:
            logger.error("Number of masks in directory does not match number of scans, but is greater than 1. Exiting...")
            sys.exit()
    else:
        logger.info("No mask directory specified as command line option. No masks included during RegistrationPipeFH initialization.")
    return inputs

def isFileHandler(inSource, inTarget=None):
    """Source and target types can be either RegistrationPipeFH (or its base class) 
    or strings. Regardless of which is chosen, they must both be the same type.
    If this function returns True - both types are fileHandlers. If it returns
    false, both types are strings. If there is a mismatch, the assert statement
    should cause an error to be raised."""
    isFileHandlingClass = True
    assertMsg = 'source and target files must both be same type: RegistrationPipeFH or string'
    if isinstance(inSource, rfh.RegistrationFHBase):
        if inTarget:
            assert isinstance(inTarget, rfh.RegistrationFHBase), assertMsg
    else:
        if inTarget:
            assert not isinstance(inTarget, rfh.RegistrationFHBase), assertMsg
        isFileHandlingClass = False
    return(isFileHandlingClass)

def setupInitModel(inputModel, pipeDir=None):
    """
        Creates fileHandlers for the initModel by reading files from.
        The directory where the input is specified.
        The following files and naming scheme are required:
            name.mnc --> File in standard registration space.
            name_mask.mnc --> Mask for name.mnc
        The following can optionally be included in the same directory as the above:
            name_native.mnc --> File in native scanner space.
            name_native_mask.mnc --> Mask for name_native.mnc
            name_native_to_standard.xfm --> Transform from native space to standard space
    """
    errorMsg = "Failed to properly set up initModel."
    
    try:
        imageFile = abspath(inputModel)
        imageBase = fh.removeBaseAndExtension(imageFile)
        imageDirectory = dirname(imageFile)
        if not pipeDir:
            pipeDir = abspath(curdir)
        initModelDir = fh.createSubDir(pipeDir, "init_model")
        if not exists(imageFile):
            errorMsg = "Specified --init-model does not exist: " + str(inputModel)
            raise
        else:
            mask = imageDirectory + "/" + imageBase + "_mask.mnc"
            if not exists(mask):
                errorMsg = "Required mask for the --init-model does not exist: " + str(mask)
                raise
            standardFH = rfh.RegistrationPipeFH(imageFile, mask=mask, basedir=initModelDir)            
            #if native file exists, create FH
            nativeFileName = imageDirectory + "/" + imageBase + "_native.mnc"
            if exists(nativeFileName):
                mask = imageDirectory + "/" + imageBase + "_native_mask.mnc"
                if not exists(mask):
                    errorMsg = "_native.mnc file included but associated mask not found"
                    raise
                else:
                    nativeFH = rfh.RegistrationPipeFH(nativeFileName, mask=mask, basedir=initModelDir)
                    nativeToStdXfm = imageDirectory + "/" + imageBase + "_native_to_standard.xfm"
                    if exists(nativeToStdXfm):
                        nativeFH.setLastXfm(standardFH, nativeToStdXfm)
                    else:
                        nativeToStdXfm = None
            else:
                nativeFH = None
                nativeToStdXfm = None
            return (standardFH, nativeFH, nativeToStdXfm)
    except:
        print errorMsg
        print "Exiting..."
        sys.exit()


def getFinestResolution(inSource):
    """
        This function will return the highest (or finest) resolution 
        present in the inSource file.     
    """
    imageResolution = []
    if isFileHandler(inSource):
        # make sure that pyminc does not complain about non-existing files. During the
        # creation of the overall compute graph the following file might not exist. In
        # that case, use the inputFileName, or raise an exception
        if(isfile(inSource.getLastBasevol())):
            imageResolution = volumeFromFile(inSource.getLastBasevol()).separations
        elif(isfile(inSource.inputFileName)):
            imageResolution = volumeFromFile(inSource.inputFileName).separations
        else:
            # neither the last base volume, nor the input file name exist at this point
            # this could happen when we evaluate an average for instance
            raise
    else: 
        imageResolution = volumeFromFile(inSource).separations
    
    # the abs function does not work on lists... so we have to loop over it.  This 
    # to avoid issues with negative step sizes.  Initialize with first dimension
    finestRes = abs(imageResolution[0])
    for i in range(1, len(imageResolution)):
        if(abs(imageResolution[i]) < finestRes):
            finestRes = abs(imageResolution[i])
    
    return finestRes
    
