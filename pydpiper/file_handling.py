#!/usr/bin/env python

from __future__ import print_function
from os.path import basename, isdir, splitext, abspath, join
import os

"""File handling methods for creating subdirectories/base file names as needed"""

existing_dirs = set()

cached_cwd = os.getcwd()

#def abspath_(path):
#    """Since Pydpiper never calls os.(f)chdir, use cached result of `os.getcwd`"""
#    return os.path.normpath(join(cached_cwd, path))
def removeBaseAndExtension(filename):
    """removes path as well as extension from filename"""
    bname = basename(filename)
    root,ext = splitext(bname)
    return(root)
def createSubDir(input_dir, subdir, ensure_parent_exists=True):
    #abspath in makedirsIgnoreExisting handles extra / if appropriate
    _newdir = join(input_dir, subdir)
    returnDir = makedirsIgnoreExisting(_newdir) if ensure_parent_exists else makedirIgnoreExisting(_newdir)
    return returnDir
def createLogDir(input_dir):
    _logDir = createSubDir(input_dir, "log")
    return (_logDir)
def logFromFile(logDir, inFile):
    """ creates a log file from an input filename
        First verifies existence of input directory
        Then, takes the input file, strips out any extensions, and returns a 
        filename with the same basename, in the log directory, and 
        with a .log extension"""
    #_logDir = makedirsIgnoreExisting(logDir)
    # new assumption: logDir exists (e.g., created by createLogDir)
    _logDir = logDir
    logBase = removeBaseAndExtension(inFile)
    log = createLogFile(_logDir, logBase)
    return(log)
def createBaseName(input_dir, base):
    # assume all / in input_dir accounted for? or add checking
    return join(input_dir, base)
def createLogFile(dirName, baseName):
    log = "%s/%s.log" % (dirName, baseName)
    return(log)
def createBackupDir(output, pipelineName):
    _backupDir = createSubDir(output, pipelineName + "-pydpiper-backups")
    return(_backupDir)
def makedirIgnoreExisting(dirname):
    """os.makedirs which doesn't fail if dir exists"""
    return makeIgnoreExisting(dirname=dirname, f=os.mkdir, test=isdir)
def makedirsIgnoreExisting(dirname):
    """os.makedir which doesn't fail if dir exists"""
    return makeIgnoreExisting(dirname=dirname, f=os.makedirs, test=isdir)
def makeIgnoreExisting(dirname, f, test):
    """Directory operation like `f` but which isn't called on `dirname` if `test` succeeds"""
    #newDir = abspath_(dirname)
    newDir = dirname # FIXME does this still work? need to test ...
    if newDir not in existing_dirs:
        try:
            if not test(newDir):
                f(newDir)
            existing_dirs.add(newDir)
        except:
            print("Could not create directory " + str(dirname))
            raise
    return newDir
