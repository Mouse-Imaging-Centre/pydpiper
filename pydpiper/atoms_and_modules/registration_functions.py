#!/usr/bin/env python

from __future__ import print_function
from os.path import splitext, basename
import subprocess
import sys
import re

def checkInputFiles(args):
    # We currently allow any number of input files via setting `nargs=*`,
    # so an application that actually requires
    # input files (not all do, some use a .csv file) should check some are provided.
    if len(args) < 1:
        print("\nError: no input files are provided. Exiting...\n")
        sys.exit(1)
    else:
        # here we should also check that the input files can be read
        issuesWithInputs = False
        for inputF in args:
            mincinfoCmd = subprocess.Popen(["mincinfo", inputF], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False)
            # the following returns anything other than None, the string matched, 
            # and thus indicates that the file could not be read
            if re.search("Unable to open file", mincinfoCmd.stdout.read()):
                print("Error: can not read input file: " + str(inputF))
                issuesWithInputs = True
        if issuesWithInputs:
            print("Error: issues reading input files. Exiting...\n")
            sys.exit(1)
    # lastly we should check that the actual filenames are distinct, because
    # directories are made based on the basename
    seen = set()
    for inputF in args:
        fileBase = splitext(basename(inputF))[0]
        if fileBase in seen:
            print("Error: the following name occurs at least twice in the input file list:\n" + str(fileBase) + "\nPlease provide unique names for all input files. Exiting\n")
            sys.exit(1)
        seen.add(fileBase)
