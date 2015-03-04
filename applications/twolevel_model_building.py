#!/usr/bin/env python

from __future__ import print_function
from pydpiper.application import AbstractApplication
import atoms_and_modules.registration_functions as rf
import atoms_and_modules.minc_modules as mm
import atoms_and_modules.minc_parameters as mp
import atoms_and_modules.LSQ6 as lsq6
import atoms_and_modules.LSQ12 as lsq12
import atoms_and_modules.NLIN as nl
import atoms_and_modules.stats_tools as st
import atoms_and_modules.registration_file_handling as rfh
from os.path import split, splitext, abspath
import sys
import logging

logger = logging.getLogger(__name__)

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
        lsq6.addLSQ6ArgumentGroup(self.parser)
        lsq12.addLSQ12ArgumentGroup(self.parser)
        nl.addNlinRegArgumentGroup(self.parser)
        rf.addGenRegArgumentGroup(self.parser)
        st.addStatsArguments(self.parser)
        
        # set help - note that the format is messed up, something that can be fixed if we upgrade
        # from optparse to argparse.
        self.parser.description = helpString
        
    def setup_appName(self):
        appName = "twolevel_model_building"
        return appName
    
    def run(self):
        options = self.options
        args = self.args
        
        # Setup output directories for two-level model building: 
        (subjectDirs, dirs) = rf.setupTwoLevelDirectories(args[0], self.outputDir, options.pipeline_name, module="ALL")
        
        # read in files from CSV
        subjects = rf.setupSubjectHash(args[0], subjectDirs, options.mask_dir)
        
        #firstlevelNlins stores per subject NLIN avgs, subjStats stores first level stats, to be resampled at the end
        firstlevelNlins = [] 
        subjStats = [] 
        
        ### first level of registrations: register within subject
        for i in range(len(subjects)):   
            baseVol = subjects[i][0].getLastBasevol()
            subjBase = splitext(split(baseVol)[1])[0]
            if options.input_space == "native":
                initModel, targetPipeFH = rf.setInitialTarget(options.init_model, 
                                                              options.lsq6_target, 
                                                              subjectDirs[i].lsq6Dir,
                                                              self.outputDir,
                                                              options.pipeline_name)
                #LSQ6 MODULE, NUC and INORM
                runLSQ6NucInorm = lsq6.LSQ6NUCInorm(subjects[i],
                                                    targetPipeFH,
                                                    initModel, 
                                                    subjectDirs[i].lsq6Dir, 
                                                    options)
                self.pipeline.addPipeline(runLSQ6NucInorm.p)
            if options.input_space=="native" or options.input_space=="lsq6":
                # LSQ12+NLIN (registration starts here or is run after LSQ6)
                if options.input_space == "lsq6":
                    initModel=None
                lsq12Nlin = mm.FullIterativeLSQ12Nlin(subjects[i], 
                                                      subjectDirs[i], 
                                                      options, 
                                                      avgPrefix=subjBase,
                                                      initModel=initModel)
                self.pipeline.addPipeline(lsq12Nlin.p)
                finalNlin = lsq12Nlin.nlinFH
                #If no protocols are specified, use same lsq12 and nlin protocols as for first level registration
                if not options.lsq12_protocol:
                    options.lsq12_protocol = lsq12Nlin.lsq12Params
                if not options.nlin_protocol:
                    options.nlin_protocol = lsq12Nlin.nlinParams
            elif options.input_space=="lsq12":
                #If inputs in lsq12 space, run NLIN only 
                lsq12AvgFile = abspath(subjectDirs[i].lsq12Dir) + "/" + subjBase + "-lsq12avg.mnc"
                nlinObj = nl.initializeAndRunNLIN(subjectDirs[i].lsq12Dir,
                                                  subjects[i],
                                                  subjectDirs[i].nlinDir,
                                                  avgPrefix=subjBase,
                                                  createAvg=True,
                                                  targetAvg=lsq12AvgFile,
                                                  targetMask=options.target_mask,
                                                  nlin_protocol=options.nlin_protocol,
                                                  reg_method=options.reg_method)
        
                self.pipeline.addPipeline(nlinObj.p)
                finalNlin = nlinObj.nlinAverages[-1]
                # If no protocols are specified, get lsq12 based on resolution of one of the existing input files.
                # Use same nlin protocol as the one we ran previously. 
                if not options.lsq12_protocol: 
                    if not options.lsq12_subject_matter:
                        fileRes = rf.returnFinestResolution(subjects[i][0])
                    options.lsq12_protocol = mp.setLSQ12MinctraccParams(fileRes, 
                                                                        subject_matter=options.lsq12_subject_matter)
                if not options.nlin_protocol:
                    options.nlin_protocol = nlinObj.nlinParams
            else:
                print("--input-space can only be native, lsq6 or lsq12. You specified: " + str(options.input_space))
                sys.exit()
            
            # add the last NLIN average to the volumes that will proceed to step 2
            firstlevelNlins.append(finalNlin)
            if options.calc_stats:
                tmpStats=[]
                for s in subjects[i]:
                    stats = st.CalcStats(s, finalNlin, options.stats_kernels)
                    self.pipeline.addPipeline(stats.p)
                    tmpStats.append(stats)
                subjStats.append(tmpStats)
        # second level of registrations: register final averages from first level 
        # TODO: Allow for LSQ6 reg first, or just NLIN. Right now, we allow LSQ12+NLIN only
        firstLevelNlinsNewFH = []
        for nlin in firstlevelNlins:
            nlinFH = rfh.RegistrationPipeFH(nlin.getLastBasevol(), mask=nlin.getMask(), basedir=dirs.processedDir)
            firstLevelNlinsNewFH.append(nlinFH)
        # the following call needs to figure out at what resolution the LSQ12 and NLIN stages
        # are supposed to be run. For this reason (if no subject matter is specified), 
        # we will pass along the initial model
        lsq12Nlin = mm.FullIterativeLSQ12Nlin(firstLevelNlinsNewFH, 
                                              dirs, 
                                              options, 
                                              avgPrefix="second_level",
                                              initModel=initModel)
        self.pipeline.addPipeline(lsq12Nlin.p)
        finalNlin = lsq12Nlin.nlinFH
        initialTarget = lsq12Nlin.initialTarget
        
        if options.calc_stats:
            for s in firstLevelNlinsNewFH:
                stats = st.CalcStats(s, finalNlin, options.stats_kernels)
                self.pipeline.addPipeline(stats.p)
                # now resample the stats files from the first level registration to the common space
                # created by the second level of registration
                for i in range(len(subjects)):
                    for s in range(len(subjects[i])):
                        # get the last xfm from the second level registrations
                        xfm = firstLevelNlinsNewFH[i].getLastXfm(finalNlin)
                        p = mm.resampleToCommon(xfm,
                                                subjects[i][s], 
                                                subjStats[i][s].statsGroup, 
                                                options.stats_kernels, 
                                                initialTarget)
                        self.pipeline.addPipeline(p)
        
if __name__ == "__main__":
    application = LongitudinalTwolevelNlin()
    application.start()
