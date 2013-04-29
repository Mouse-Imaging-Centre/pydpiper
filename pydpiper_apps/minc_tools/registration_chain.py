#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import pydpiper.file_handling as fh
import pydpiper_apps.minc_tools.registration_functions as rf
import pydpiper_apps.minc_tools.registration_file_handling as rfh
import pydpiper_apps.minc_tools.minc_modules as mm
import pydpiper_apps.minc_tools.minc_atoms as ma
import pydpiper_apps.minc_tools.stats_tools as st
import pydpiper_apps.minc_tools.option_groups as og
import pydpiper_apps.minc_tools.old_MBM_interface_functions as ombm
import Pyro
from optparse import OptionGroup
from datetime import date
from os.path import abspath, isdir
import logging
import csv
import sys

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

class RegistrationChain(AbstractApplication):
    def setup_options(self):
        group = OptionGroup(self.parser, "Registration-chain options", 
                        "Options for registering consecutive timepoints of longitudinal data.")
        group.add_option("--avg-time-point", dest="avg_time_point",
                      type="int", default=1,
                      help="Time point averaged prior to this registration to get common nlin space.")
        group.add_option("--input-space", dest="input_space",
                      type="string", default="lsq6", 
                      help="Option to specify space of input-files. Can be lsq6 (default), lsq12 or native.")
        self.parser.add_option_group(group)
        """Add option groups from specific modules"""
        rf.addGenRegOptionGroup(self.parser)
        og.tmpLongitudinalOptionGroup(self.parser)
        st.addStatsOptions(self.parser)
        
        self.parser.set_usage("%prog [options] input.csv") 

    def setup_appName(self):
        appName = "Registration-chain"
        return appName

    def run(self):
        
        """Directory handling etc as in MBM"""
        if not self.options.pipeline_name:
            pipeName = str(date.today()) + "_pipeline"
        else:
            pipeName = self.options.pipeline_name
        
        processedDirectory = fh.createSubDir(self.outputDir, pipeName + "_processed")
        
        """Check that correct registration method was specified"""
        if self.options.reg_method != "minctracc" and self.options.reg_method != "mincANTS":
            logger.error("Incorrect registration method specified: " + self.options.reg_method)
            sys.exit()
        
        """Read in files from csv"""
        fileList = open(self.args[0], 'rb')
        subjectList = csv.reader(fileList, delimiter=',', skipinitialspace=True)
        subjects = {} # One array of images for each subject
        index = 0 
        for subj in subjectList:
            subjects[index] = rf.initializeInputFiles(subj, processedDirectory, self.options.mask_dir)
            index += 1
        
        """Put blurs into array"""
        blurs = []
        for i in self.options.stats_kernels.split(","):
            blurs.append(float(i))
        
        """Create file handler for nlin average from MBM"""
        if self.options.nlin_avg:
            nlinFH = rfh.RegistrationFHBase(abspath(self.options.nlin_avg), processedDirectory)
        else:
            nlinFH = None
        if self.options.mbm_dir and not isdir(abspath(self.options.mbm_dir)):
            logger.error("The --mbm-directory specified does not exist: " + abspath(self.options.mbm_dir))
            sys.exit()
        
        """Take average time point, subtract 1 for proper indexing"""
        avgTime = self.options.avg_time_point - 1
        
        """Get transforms from inputs to final nlin average and vice versa"""
        if self.options.nlin_avg and self.options.mbm_dir:
            xfmsPipe = ombm.getXfms(nlinFH, subjects, self.options.input_space, abspath(self.options.mbm_dir), time=avgTime)
            if len(xfmsPipe.stages) > 0:
                self.pipeline.addPipeline(xfmsPipe)
            
            """Align everything to lsq6 space, with ordering depending on time point"""
            #Disabled now for testing purposes. 
            #if options.lsq6_space:
                #"""lsq6Files from MBM run will be file handlers indexed by subjects[s][time]"""
                #lsq6Files = ombm.getLsq6Files(abspath(options.mbm_dir), subjects, avgTime, processedDirectory)
                #lsq6Pipe = mm.ChainAlignLSQ6(subjects, avgTime, lsq6Files)
                #self.p.addPipeline(lsq6Pipe)
        else:
            logger.info("MBM directory and nlin_average not specified.")
            logger.info("Calculating registration chain only")
        
        """Create a dictionary of statistics. Each subject gets an array of statistics
           indexed by timepoint. The indexing on the subjectStats dictionary should match
           the subjects dictionary"""
        subjectStats = {}
        
        for subj in subjects:
            subjectStats[subj] = {}
            s = subjects[subj]
            count = len(s) 
            for i in range(count - 1):
                # Create new groups
                # MF TODO: Make generalization of registration parameters easier. 
                if self.options.reg_method == "mincANTS":
                    register = mm.LSQ12ANTSNlin(s[i], s[i+1])
                    self.pipeline.addPipeline(register.p)
                elif self.options.reg_method == "minctracc":
                    hm = mm.HierarchicalMinctracc(s[i], s[i+1])
                    self.pipeline.addPipeline(hm.p)
                """Resample s[i] into space of s[i+1]""" 
                if nlinFH:
                    resample = ma.mincresample(s[i], s[i+1], likeFile=nlinFH)
                else:
                    resample = ma.mincresample(s[i], s[i+1], likeFile=s[i])
                self.pipeline.addStage(resample)
                lastXfm = s[i].getLastXfm(s[i+1])
                # not sure if want to use new group or existing
                """Initialize newGroup with initial file as i resampled to i+1 
                   space and setLastXfm to be final xfm from original group"""
                groupName = "time_point_" + str(i) + "_to_" + str(i+1) 
                s[i].newGroup(inputVolume=resample.outputFiles[0], groupName=groupName) 
                s[i].setLastXfm(s[i+1], lastXfm)
                stats = st.CalcChainStats(s[i], s[i+1], blurs)
                stats.calcFullDisplacement()
                stats.calcDetAndLogDet(useFullDisp=True)
                self.pipeline.addPipeline(stats.p)
                subjectStats[subj][i] = stats.statsGroup
        
        """Now that all registration is complete, concat transforms and resample"""
        if self.options.nlin_avg and self.options.mbm_dir:
            car = ombm.concatAndResample(subjects, subjectStats, avgTime, nlinFH, blurs) 
            self.pipeline.addPipeline(car)

if __name__ == "__main__":
    
    application = RegistrationChain()
    application.start()
