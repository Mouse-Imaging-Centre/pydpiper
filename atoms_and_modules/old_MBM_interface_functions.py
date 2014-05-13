#!/usr/bin/env python

from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
import pydpiper.file_handling as fh
import atoms_and_modules.registration_file_handling as rfh
import atoms_and_modules.minc_atoms as ma
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
        #TODO: Fix this to connect to new build model (use new naming system) and remove LSQ12. 
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
                    ix = ma.xfmInvert(xfmToNative, inputFH)
                    pipeline.addStage(ix)
                    xfmFromNative = ix.outputFiles[0]
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
