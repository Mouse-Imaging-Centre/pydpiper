#!/usr/bin/env python

from os.path import basename,isdir,splitext, abspath
from os import mkdir,makedirs

"""File handling methods for creating subdirectories/base file names as needed"""

def removeBaseAndExtension(filename):
    """removes path as well as extension from filename"""
    bname = basename(filename)
    root,ext = splitext(bname)
    return(root)
def createSubDir(input_dir, subdir):
    #abspath in makedirsIgnoreExisting handles extra / if appropriate
    _newdir = input_dir + "/" + subdir
    returnDir = makedirsIgnoreExisting(_newdir)
    return (returnDir)
def createLogDir(input_dir):
    _logDir = createSubDir(input_dir, "log")
    return (_logDir)
def logFromFile(logDir, inFile):
    """ creates a log file from an input filename
        First verifies existence of input directory
        Then, takes the input file, strips out any extensions, and returns a 
        filename with the same basename, in the log directory, and 
        with a .log extension"""
    _logDir = makedirsIgnoreExisting(logDir)
    logBase = removeBaseAndExtension(inFile)
    log = createLogFile(_logDir, logBase)
    return(log)
def createBaseName(input_dir, base):
    # assume all / in input_dir accounted for? or add checking
    return (input_dir + "/" + base)
def createLogFile(dirName, baseName):
    log = "%s/%s.log" % (dirName, baseName)
    return(log)
def createBackupDir(output):
    _backupDir = createSubDir(output, "pydpiper-backups")
    return(_backupDir)
def makedirsIgnoreExisting(dirname):
    """os.makedirs which fails if dir exists"""
    try:
        newDir = abspath(dirname)
        if not isdir(newDir):
            makedirs(newDir)
        return(newDir)
    except:
        print "Could not create directory " + str(dirname)
        raise
