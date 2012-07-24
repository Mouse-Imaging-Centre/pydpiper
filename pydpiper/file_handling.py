#!/usr/bin/env python

import Pyro.core
import Pyro.naming
from os.path import basename,isdir,splitext, abspath
from os import mkdir,makedirs

Pyro.config.PYRO_MOBILE_CODE=1 

"""File handling methods for creating subdirectories/base file names as needed"""


def removeFileExt(inputFile):
    """removes extension only from end of abspath"""
    base, ext = splitext(inputFile)
    return(base)
def removeBaseAndExtension(filename):
    """removes path as well as extension from filename"""
    bname = basename(filename)
    root,ext = splitext(bname)
    return(root)
def createSubDir(input_dir, subdir):
    # check for input_dir/subdir format?
    # at this point, assume all / properly accounted for
    _newdir = input_dir + "/" + subdir
    if not isdir(_newdir):
        mkdir(_newdir)
    return (_newdir)
def createLogDir(input_dir):
    _logDir = createSubDir(input_dir, "log")
    return (_logDir)
def createBaseName(input_dir, base):
    # assume all / in input_dir accounted for? or add checking
    return (input_dir + "/" + base)
def createSubDirSubBase(input_dir, subdir, input_base):
    _subDir = createSubDir(input_dir, subdir)
    _subBase = createBaseName(_subDir, input_base)
    return (_subDir, _subBase)
def createLogDirLogBase(input_dir, input_base):
    #MF TODO: This will go away and be replaced 
    _logDir = createLogDir(input_dir)
    _logBase = createBaseName(_logDir, input_base)
    return (_logDir, _logBase)
def createLogDirandFile(inputDir, baseName):
    """Check to see if log directory is created
       before creating fileName"""
    _logDir = createLogDir(inputDir)
    _logFile = createLogFile(_logDir, baseName)
    return(_logFile)
def createLogFile(dirName, baseName):
    log = "%s/%s.log" % (dirName, baseName)
    return(log)
def createBackupDir(output):
    _backupDir = createSubDir(output, "pydpiper-backups")
    return(_backupDir)
def createOutputFileName(argArray):
    outFileName = [] #clear out any arguments from previous call    
    for a in argArray:
        outFileName.append(str(a))
    return("".join(outFileName))
def createOutputAndLogFiles(output_base, log_base, fileType, argArray=None):
    if argArray:
        outArray = [output_base, "_", "_".join(argArray), fileType]
        logArray = [log_base, "_", "_".join(argArray), ".log"] 
    else:
        outArray = [output_base, fileType]
        logArray = [log_base, ".log"]
    outFile = createOutputFileName(outArray)
    logFile = createOutputFileName(logArray)
    return (outFile, logFile)
def makedirsIgnoreExisting(dirname):
    """os.makedirs which fails silently if dir exists"""
    try:
        newDir = abspath(dirname)
        if not isdir(newDir):
            makedirs(newDir)
        return(newDir)
    except:
        print "Could not create directory " + str(dirname)
        raise
