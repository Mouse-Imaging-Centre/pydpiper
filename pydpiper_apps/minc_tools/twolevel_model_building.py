#!/usr/bin/env python



from pydpiper.application import AbstractApplication
from pydpiper.pipeline import CmdStage, InputFile, OutputFile, LogFile
import pydpiper.file_handling as fh
import pydpiper_apps.minc_tools.registration_functions as rf
import pydpiper_apps.minc_tools.registration_file_handling as rfh
import pydpiper_apps.minc_tools.minc_modules as mm
import pydpiper_apps.minc_tools.NLIN as nl
import pydpiper_apps.minc_tools.minc_atoms as ma
import pydpiper_apps.minc_tools.stats_tools as st
import pydpiper_apps.minc_tools.option_groups as og
import pydpiper_apps.minc_tools.hierarchical_minctracc as hmt
import pydpiper_apps.minc_tools.old_MBM_interface_functions as ombm
import Pyro
from optparse import OptionGroup
from datetime import date
from os.path import abspath, isdir, split, splitext
import logging
import csv
import sys

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

class LongitudinalTwolevelNlin(AbstractApplication):
    def setup_options(self):
        helpString="""
twolevel_model_building

A pydpiper application designed to work with longitudinal data. LSQ12
and nonlinear registration is used to create a consensus average of
every subject. A second level of LSQ12 and nonlinear registrations is 
then used to bring all the consensus averages from each subject into
their own consensus average.

Some assumptions:
* at least two timepoints per subject 
  * future work should be able to extend this to allow single timepoint subjects
* all images must be similar enough to allow registration

The last point is particularly important: the consensus average building process
aligns every image from each subject to every other image from that subject. Early
developmental data or tumour data, where the first image in the series might not be
alignable to the last image in the series, is thus not suited for this approach.

Data is passed to the application through a CSV file. This file has one line per subject,
with each scan per subject listed on the same line and separated by a comma.
"""
        
        # own options go here
        nl.addNlinRegOptionGroup(self.parser)
        rf.addGenRegOptionGroup(self.parser)
        
        # TODO: better usage description (once I've figured out what the usage will be ...)
        self.parser.set_usage("%prog [options] input.csv")
        # set help - note that the format is messed up, something that can be fixed if we upgrade
        # from optparse to argparse.
        self.parser.set_description(helpString) 
        
    def setup_appName(self):
        appName = "twolevel_model_building"
        return appName
    
    def run(self):
        options = self.options
        args = self.args
        
        #TODO: some of this code is copied and pasted from NLIN.py; make it reusable?
        if not options.pipeline_name:
            pipeName = str(date.today()) + "_pipeline"
        else:
            pipeName = options.pipeline_name
            
        # creation of directory tree                
        pipeDir = fh.makedirsIgnoreExisting(options.pipeline_dir)
        nlinDirectory = fh.createSubDir(pipeDir, pipeName + "_nlin")
        processedDirectory = fh.createSubDir(pipeDir, pipeName + "_processed")
        
        # read in files from CSV
        fileList = open(self.args[0], 'rb')
        subjectList = csv.reader(fileList, delimiter=',', skipinitialspace=True)
        subjects = {}
        index = 0
        for subj in subjectList:
            subjects[index] = rf.initializeInputFiles(subj, processedDirectory, maskDir=options.mask_dir)
            index += 1
        
        # testing only part of the code: run one lsq12 followed by an NLIN
        
        # this actually makes little sense, but what the hell
#         blurs = [10,5,5]
#         step = [4,4,4]
#         simplex = [20,20,20]
#         filesToAvg = []
#         LSQ12 = mm.LSQ12(inputFiles[0], inputFiles[1], blurs, step, simplex)
#         rs = ma.mincresample(inputFiles[0], inputFiles[1], likeFile=inputFiles[1])
#         filesToAvg.append(rs.outputFiles[0])
#         self.pipeline.addStage(rs)
#         self.pipeline.addPipeline(LSQ12.p)
#         LSQ12 = mm.LSQ12(inputFiles[1], inputFiles[0], blurs, step, simplex)
#         rs = ma.mincresample(inputFiles[1], inputFiles[0], likeFile=inputFiles[0])
#         filesToAvg.append(rs.outputFiles[0])
#         self.pipeline.addPipeline(LSQ12.p)
#         self.pipeline.addStage(rs)
#         lsq12AvgFile = abspath(processedDirectory) + "/" + "lsq12avg.mnc"
#         avg = ma.mincAverage(filesToAvg, lsq12AvgFile)
#         self.pipeline.addStage(avg)

        # TODO: LSQ6 registration if requested
        # TODO: LSQ12 registration if requested
        
        # for the moment assume that all input files are in LSQ12 space
        index = 0
        firstlevelNlins = [] # stores the per subject NLINs avgs
        ### first level of registrations: register within subject
        for i in range(len(subjects)):
            ### filename munging ###
            # takes the filename of the first file in the list and prepends FIRSTLEVEL- to it
            baseVol = subjects[i][0].getLastBasevol()
            subjBase = "FIRSTLEVEL-" + splitext(split(baseVol)[1])[0]
            # create an NLIN directory inside the main NLIN directory per subject
            firstNlinDirectory = fh.createSubDir(nlinDirectory, subjBase)
            # put the lsq12 averages in the processed directory for now
            lsq12AvgFile = abspath(processedDirectory) + "/" + subjBase + "-lsq12avg.mnc"
            lsq12FH = rfh.RegistrationPipeFH(lsq12AvgFile, basedir=nlinDirectory)
            ### step 1: create an average of all the input files per subject ###
            # TODO: optionally allow LSQ12 or LSQ6 + LSQ12 here rather than assume they come prealigned
            avg = ma.mincAverage(subjects[i], lsq12FH)
            lsq12FH.setLastBasevol(avg.outputFiles[0])
            self.pipeline.addStage(avg)
            ### step 2: run iterative ANTS model building for each subject
            NL = nl.NLINANTS(subjects[i], lsq12FH, firstNlinDirectory, options.nlin_protocol)
            NL.iterate()
            self.pipeline.addPipeline(NL.p)
            # add the last NLIN average to the volumes that will proceed to step 2
            firstlevelNlins.append(NL.nlinAvg[-1])
        ### second level of registrations: register across subjects
        ## start by averaging all the NLINs from the first level; should be replaced by an LSQ12
        lsq12AvgFile = abspath(processedDirectory) + "/firstlevelNlins-lsq12avg.mnc"
        lsq12FH = rfh.RegistrationPipeFH(lsq12AvgFile, basedir=nlinDirectory)
        avg = ma.mincAverage(firstlevelNlins, lsq12FH)
        lsq12FH.setLastBasevol(avg.outputFiles[0])
        self.pipeline.addStage(avg)
        ## run iterative ANTS model building across the per subject averages
        # TODO: allow for a different protocol here.
        NL = nl.NLINANTS(firstlevelNlins, lsq12FH, nlinDirectory, options.nlin_protocol)
        NL.iterate()
        self.pipeline.addPipeline(NL.p)
        # and done! Still to do: create the different stats files and resample to appropriate places.
        
if __name__ == "__main__":
    application = LongitudinalTwolevelNlin()
    application.start()