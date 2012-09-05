#!/usr/bin/env python

import pydpiper_apps.minc_tools.registration_file_handling as rfh
import pydpiper.file_handling as fh
from os.path import abspath, exists, dirname
from os import curdir
import sys

def initializeInputFiles(args, mainDirectory):
    inputs = []
    for iFile in range(len(args)):
        inputPipeFH = rfh.RegistrationPipeFH(abspath(args[iFile]), mainDirectory)
        inputs.append(inputPipeFH)
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
            native_to_standard.xfm --> Transform from native space to standard space
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
            standardFH = rfh.RegistrationFHBase(imageFile, initModelDir)
            mask = imageDirectory + "/" + imageBase + "_mask.mnc"
            standardFH.setMask(abspath(mask))
            #if native file exists, create FH
            nativeFileName = imageDirectory + "/" + imageBase + "_native.mnc"
            if exists(nativeFileName):
                nativeFH = rfh.RegistrationFHBase(nativeFileName, initModelDir)
                mask = imageDirectory + "/" + imageBase + "_native_mask.mnc"
                if not exists(mask):
                    errorMsg = "_native.mnc file included but associated mask not found"
                    raise
                else:
                    nativeFH.setMask(abspath(mask))
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
    
    