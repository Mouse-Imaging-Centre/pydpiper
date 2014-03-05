#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import atoms_and_modules.registration_functions as rf
import atoms_and_modules.registration_file_handling as rfh
import atoms_and_modules.minc_modules as mm
import atoms_and_modules.minc_atoms as ma
import atoms_and_modules.stats_tools as st
import atoms_and_modules.option_groups as og
import atoms_and_modules.minc_parameters as mp
import atoms_and_modules.hierarchical_minctracc as hmt
import atoms_and_modules.old_MBM_interface_functions as ombm
import Pyro
from optparse import OptionGroup
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
        group.add_option("--common-space-name", dest="common_name",
                      type="string", default="common", 
                      help="Option to specify a name for the common space. This is useful for the "
                            "creation of more readable output file names. Default is common.")
        self.parser.add_option_group(group)
        """Add option groups from specific modules"""
        rf.addGenRegOptionGroup(self.parser)
        mp.addLSQ12NLINOptionGroup(self.parser)
        og.tmpLongitudinalOptionGroup(self.parser)
        st.addStatsOptions(self.parser)
        
        self.parser.set_usage("%prog [options] input.csv") 

    def setup_appName(self):
        appName = "Registration-chain"
        return appName

    def run(self):
        
        # Setup output directories for registration chain (_processed only).        
        dirs = rf.setupDirectories(self.outputDir, self.options.pipeline_name, module=None)
        
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
            subjects[index] = rf.initializeInputFiles(subj, dirs.processedDir, self.options.mask_dir)
            index += 1
        
        """Put blurs into array"""
        blurs = []
        for i in self.options.stats_kernels.split(","):
            blurs.append(float(i))
        
        """Create file handler for nlin average from MBM"""
        if self.options.nlin_avg:
            nlinFH = rfh.RegistrationPipeFH(abspath(self.options.nlin_avg), basedir=dirs.processedDir)
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
            
            """ Possible TODO: Align everything to lsq6 space, with ordering depending on time point.
                This functionality was previously partially implemented (ChainAlignLSQ6), but was removed, as it was
                never fully working or tested. Leaving this comment as a place holder in case we want to re-implement. 
            """
        else:
            logger.info("MBM directory and nlin_average not specified.")
            logger.info("Calculating registration chain only")
        
        for subj in subjects:
            s = subjects[subj]
            count = len(s) 
            for i in range(count - 1):
                # Create new groups
                if self.options.reg_method == "mincANTS":
                    register = mm.LSQ12ANTSNlin(s[i], 
                                                s[i+1],
                                                lsq12_protocol=self.options.lsq12_protocol,
                                                nlin_protocol=self.options.nlin_protocol)
                    self.pipeline.addPipeline(register.p)
                elif self.options.reg_method == "minctracc":
                    hm = hmt.HierarchicalMinctracc(s[i], 
                                                   s[i+1], 
                                                   lsq12_protocol=self.options.lsq12_protocol,
                                                   nlin_protocol=self.options.nlin_protocol)
                    self.pipeline.addPipeline(hm.p)
                """Resample s[i] into space of s[i+1]""" 
                if nlinFH:
                    resample = ma.mincresample(s[i], s[i+1], likeFile=nlinFH, argArray=["-sinc"])
                else:
                    resample = ma.mincresample(s[i], s[i+1], likeFile=s[i], argArray=["-sinc"])
                self.pipeline.addStage(resample)
                """Invert transforms for use later in stats"""
                lastXfm = s[i].getLastXfm(s[i+1])
                inverseXfm = s[i+1].getLastXfm(s[i])
                if not inverseXfm:
                    "invert xfm and calculate"
                    xi = ma.xfmInvert(lastXfm, FH=s[i]) 
                    self.pipeline.addStage(xi)
                    s[i+1].addAndSetXfmToUse(s[i], xi.outputFiles[0])
        
        """Now that all registration is complete, calculate stats, concat transforms and resample"""
        car = mm.LongitudinalStatsConcatAndResample(subjects, 
                                                    avgTime, 
                                                    nlinFH, 
                                                    blurs, 
                                                    self.options.common_name) 
        self.pipeline.addPipeline(car.p)

if __name__ == "__main__":
    
    application = RegistrationChain()
    application.start()
