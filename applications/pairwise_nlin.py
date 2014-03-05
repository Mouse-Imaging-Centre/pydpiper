#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import pydpiper.file_handling as fh
import atoms_and_modules.registration_functions as rf
import atoms_and_modules.registration_file_handling as rfh
import atoms_and_modules.minc_modules as mm
import atoms_and_modules.minc_atoms as ma
import atoms_and_modules.stats_tools as st
import atoms_and_modules.option_groups as og
import atoms_and_modules.hierarchical_minctracc as hmt
import atoms_and_modules.old_MBM_interface_functions as ombm
import Pyro
from optparse import OptionGroup
from datetime import date
from os.path import abspath, isdir
import logging
import sys

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

"""NOTE: This application needs a significant overhaul and/or combining with the RegistrationChain
         Application. Until this comment is removed, please consider this class DEPRECATED."""

class PairwiseNonlinear(AbstractApplication):
    def setup_options(self):
        group = OptionGroup(self.parser, "Pairwise non-linear options", 
                        "Options for pairwise non-linear registration of lsq6 or lsq12 aligned brains.")
        group.add_option("--input-space", dest="input_space",
                      type="string", default="lsq6", 
                      help="Option to specify space of input-files. Can be lsq6 (default), lsq12 or native.")
        self.parser.add_option_group(group)
        """Add option groups from specific modules"""
        rf.addGenRegOptionGroup(self.parser)
        og.tmpLongitudinalOptionGroup(self.parser)
        st.addStatsOptions(self.parser)
        
        self.parser.set_usage("%prog [options] input files") 

    def setup_appName(self):
        appName = "Pairwise-nonlinear"
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
        
        """Create file handling classes for each image"""
        inputs = rf.initializeInputFiles(self.args, processedDirectory, self.options.mask_dir)
        
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
        
        """Get transforms from inputs to final nlin average and vice versa as well as lsq6 files"""
        if self.options.nlin_avg and self.options.mbm_dir:
            xfmsPipe = ombm.getXfms(nlinFH, inputs, self.options.input_space, abspath(self.options.mbm_dir))
            if len(xfmsPipe.stages) > 0:
                self.pipeline.addPipeline(xfmsPipe)
        else:
            logger.info("MBM directory and nlin_average not specified.")
            logger.info("Calculating pairwise nlin only without resampling to common space.")
        
        """Create a dictionary of statistics. Each subject gets an array of statistics
           indexed by inputFile."""
        subjectStats = {}
        
        """Register each image with every other image."""
        for inputFH in inputs:
            subjectStats[inputFH] = {}
            for targetFH in inputs:
                if inputFH != targetFH:
                # MF TODO: Make generalization of registration parameters easier. 
                    if self.options.reg_method == "mincANTS":
                        register = mm.LSQ12ANTSNlin(inputFH, targetFH)
                        self.pipeline.addPipeline(register.p)
                    elif self.options.reg_method == "minctracc":
                        hm = hmt.HierarchicalMinctracc(inputFH, targetFH)
                        self.pipeline.addPipeline(hm.p)
                    if nlinFH:
                        resample = ma.mincresample(inputFH, targetFH, likeFile=nlinFH)
                    else:
                        resample = ma.mincresample(inputFH, targetFH, likeFile=inputFH)
                    self.pipeline.addStage(resample)
                    inputFH.setLastBasevol(resample.outputFiles[0])
                    """Calculate statistics"""
                    stats = st.CalcChainStats(inputFH, targetFH, blurs)
                    stats.calcFullDisplacement()
                    stats.calcDetAndLogDet(useFullDisp=True)
                    self.pipeline.addPipeline(stats.p)
                    subjectStats[inputFH][targetFH] = stats.statsGroup
                    """Resample to nlin space from previous build model run, if specified"""
                    if self.options.nlin_avg and self.options.mbm_dir:
                        xfmToNlin = inputFH.getLastXfm(nlinFH, groupIndex=0)
                        res = mm.resampleToCommon(xfmToNlin, inputFH, subjectStats[inputFH][targetFH], blurs, nlinFH)
                        self.pipeline.addPipeline(res)
                    """Reset last base volume to original input before continuing to next pair in loop."""
                    inputFH.setLastBasevol(setToOriginalInput=True)

if __name__ == "__main__":
    
    application = PairwiseNonlinear()
    application.start()
