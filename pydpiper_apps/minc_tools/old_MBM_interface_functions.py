#!/usr/bin/env python

from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
import pydpiper.file_handling as fh
import pydpiper_apps.minc_tools.registration_file_handling as rfh
import pydpiper_apps.minc_tools.minc_atoms as ma
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

def concatAndResample(subjects, subjectStats, timePoint, nlinFH, blurs):
    """For each subject, take the deformation fields and resample them into the nlin-3 space.
       The transforms that are concatenated depend on which time point is used for the average"""
    pipeline = Pipeline()
    for s in subjects:
        # xfmToNlin will be either to lsq6 or native depending on other factors
        # may need an additional argument for this function
        xfmToNlin = subjects[s][timePoint].getLastXfm(nlinFH, groupIndex=0)
        count = len(subjects[s])
        for b in blurs:
            xfmArray = [xfmToNlin]
            """Do timePoint with average first"""
            res = resampleToCommon(xfmToNlin, subjects[s][timePoint], subjectStats[s][timePoint], b, nlinFH)
            pipeline.addPipeline(res)
            if not timePoint - 1 < 0:
                """Average happened at time point other than first time point. 
                   Loop over points prior to average."""
                for i in reversed(range(timePoint)):
                    xcs = getAndConcatXfm(subjects[s][i], subjectStats[s], i, xfmArray, False)
                    pipeline.addStage(xcs)
                    res = resampleToCommon(xcs.outputFiles[0], subjects[s][i], subjectStats[s][i], b, nlinFH)
                    pipeline.addPipeline(res)
            """Loop over points after average. If average is at first time point, this loop
               will hit all time points (other than first). If average is at subsequent time 
               point, it hits all time points not covered previously."""
            xfmArray=[xfmToNlin]    
            for i in range(timePoint + 1, count-1):
                xcs = getAndConcatXfm(subjects[s][i], subjectStats[s], i, xfmArray, True)
                pipeline.addStage(xcs)
                res = resampleToCommon(xcs.outputFiles[0], subjects[s][i], subjectStats[s][i], b, nlinFH)
                pipeline.addPipeline(res)
    return pipeline

def getAndConcatXfm(s, subjectStats, i, xfmArray, inverse):
    """Insert xfms into array and concat, returning CmdStage
       Note that s is subjects[s][i] and subjectStats is subjectStats[s] from calling function
       inverse=True means that we need to retrieve inverse transforms"""
    if inverse:
        xfm = subjectStats[i-1].inverseXfm
    else:
        xfm = subjectStats[i].transform
    xfmArray.insert(0, xfm)
    output = fh.createBaseName(s.statsDir, "xfm_to_common_space.xfm")
    cmd = ["xfmconcat", "-clobber"] + [InputFile(a) for a in xfmArray] + [OutputFile(output)]
    xfmConcat = CmdStage(cmd)
    xfmConcat.setLogFile(LogFile(fh.logFromFile(s.logDir, output)))
    return xfmConcat
    
    
def resampleToCommon(xfm, FH, statsGroup, b, nlinFH):
    pipeline = Pipeline()
    outputDirectory = FH.statsDir
    filesToResample = [statsGroup.jacobians[b]]
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