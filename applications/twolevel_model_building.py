#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import atoms_and_modules.registration_functions as rf
import atoms_and_modules.minc_modules as mm
import atoms_and_modules.minc_parameters as mp
import atoms_and_modules.LSQ12 as lsq12
import atoms_and_modules.NLIN as nl
import atoms_and_modules.minc_atoms as ma
import atoms_and_modules.stats_tools as st
import Pyro
from optparse import OptionGroup
from datetime import date
from os.path import abspath, isdir, split, splitext
import logging
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
        lsq12.addLSQ12OptionGroup(self.parser)
        nl.addNlinRegOptionGroup(self.parser)
        rf.addGenRegOptionGroup(self.parser)
        mp.addRegParamsOptionGroup(self.parser)
        st.addStatsOptions(self.parser)
        
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
        
        # Setup output directories for two-level model building:
        # All first level registrations go into the first level directory. The main
        # first level directory will contain an _LSQ6, _LSQ12, _NLIN and _processed directory
        # for each subject. The second level directory will contain an _lsq6/12/NLIN/processed
        # directory that will be used for registering the consensus averages.         
        (subjectDirs, dirs) = rf.setupTwoLevelDirectories(args[0], self.outputDir, options.pipeline_name, module="ALL")
        
        # read in files from CSV
        subjects = rf.setupSubjectHash(args[0], subjectDirs, self.options.mask_dir)
        
        firstlevelNlins = [] # stores the per subject NLINs avgs
        subjStats = [] #used for storing first level stats, which will have to be resampled later
        ### first level of registrations: register within subject
        for i in range(len(subjects)):
            # Add in LSQ6/12 here. 
            #TODO: Are we handling the masking correctly?  
            baseVol = subjects[i][0].getLastBasevol()
            subjBase = splitext(split(baseVol)[1])[0]
            ### create an average of all the input files per subject + run registration###
            lsq12AvgFile = abspath(subjectDirs[i].lsq12Dir) + "/" + subjBase + "-lsq12avg.mnc"
            nlinObj = nl.initializeAndRunNLIN(subjectDirs[i].lsq12Dir,
                                              subjects[i],
                                              subjectDirs[i].nlinDir,
                                              createAvg=True,
                                              targetAvg=lsq12AvgFile,
                                              targetMask=options.target_mask,
                                              nlin_protocol=options.nlin_protocol,
                                              reg_method=options.reg_method)
        
            self.pipeline.addPipeline(nlinObj.p)
            self.nlinAverages = nlinObj.nlinAverages
            
            # add the last NLIN average to the volumes that will proceed to step 2
            firstlevelNlins.append(self.nlinAverages[-1])
            if options.calc_stats:
                finalNlin = self.nlinAverages[-1]
                tmpStats=[]
                for s in subjects[i]:
                    stats = st.CalcStats(s, finalNlin, options.stats_kernels)
                    self.pipeline.addPipeline(stats.p)
                    tmpStats.append(stats)
                subjStats.append(tmpStats)
        ### second level of registrations: register across subjects
        ## average all the NLINs from the first level, iterative model building across 
        ## per subject averages - do LSQ12/NLIN only 
        if options.input_space == "lsq12":
            #Fix this to get protocol based on one of the input files from step 1!!!
            options.lsq12_protocol = nlinObj.nlinParams
        options.nlin_protocol = nlinObj.nlinParams
        lsq12Nlin = mm.FullIterativeLSQ12Nlin(firstlevelNlins, dirs, options)
        self.pipeline.addPipeline(lsq12Nlin.p)
        finalNlin = lsq12Nlin.nlinFH
        
        if options.calc_stats:
            for s in firstlevelNlins:
                stats = st.CalcStats(s, finalNlin, options.stats_kernels)
                self.pipeline.addPipeline(stats.p)
                # now resample the stats files from the first level registration to the common space
                # created by the second level of registration
                for i in range(len(subjects)):
                    for s in range(len(subjects[i])):
                        # get the last xfm from the second level registrations
                        xfm = firstlevelNlins[i].getLastXfm(nlinObj.nlinAverages[-1])
                        p = mm.resampleToCommon(xfm,
                                                subjects[i][s], 
                                                subjStats[i][s].statsGroup, 
                                                options.stats_kernels, 
                                                nlinObj.initialTarget)
                        self.pipeline.addPipeline(p)
        
if __name__ == "__main__":
    application = LongitudinalTwolevelNlin()
    application.start()
