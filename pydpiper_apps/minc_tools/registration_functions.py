#!/usr/bin/env python

import pydpiper_apps.minc_tools.registration_file_handling as rfh
from os.path import abspath

def initializeInputFiles(args, mainDirectory):
    inputs = []
    for iFile in range(len(args)):
        inputPipeFH = rfh.RegistrationPipeFH(abspath(args[iFile]), mainDirectory)
        inputs.append(inputPipeFH)
    return inputs