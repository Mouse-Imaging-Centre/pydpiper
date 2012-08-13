#!/usr/bin/env python

import pydpiper_apps.minc_tools.registration_file_handling as rfh
from os.path import abspath

def initializeInputFiles(args, mainDirectory):
    inputs = []
    for iFile in range(len(args)):
        inputPipeFH = rfh.RegistrationPipeFH(abspath(args[iFile]), mainDirectory)
        inputs.append(inputPipeFH)
    return inputs

def isFileHandler(inSource, inTarget=None):
    """Source and target types can be either RegistrationPipeFH or strings
    Regardless of which is chosen, they must both be the same type.
    If this function returns True - both types are fileHandlers. If it returns
    false, both types are strings. If there is a mismatch, the assert statement
    should cause an error to be raised."""
    isFileHandlingClass = True
    assertMsg = 'source and target files must both be same type: RegistrationPipeFH or string'
    if isinstance(inSource, rfh.RegistrationPipeFH):
        if inTarget:
            assert isinstance(inTarget, rfh.RegistrationPipeFH), assertMsg
    else:
        if inTarget:
            assert not isinstance(inTarget, rfh.RegistrationPipeFH), assertMsg
        isFileHandlingClass = False
    return(isFileHandlingClass)
