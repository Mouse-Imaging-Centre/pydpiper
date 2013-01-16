#!/usr/bin/env python

from pydpiper.application import AbstractApplication
from pydpiper.pipeline import Pipeline, CmdStage, InputFile, OutputFile, LogFile
import pydpiper.file_handling as fh
import pydpiper_apps.minc_tools.registration_functions as rf
import pydpiper_apps.minc_tools.registration_file_handling as rfh
import pydpiper_apps.minc_tools.minc_modules as mm
import pydpiper_apps.minc_tools.minc_atoms as ma
import pydpiper_apps.minc_tools.stats_tools as st
import pydpiper_apps.minc_tools.option_groups as og
import pydpiper_apps.minc_tools.old_MBM_interface_functions as ombm
import Pyro
from datetime import date
from os.path import abspath, isdir
from os import walk
import fnmatch
import logging
import csv
import sys

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

class PairwiseNonlinear(AbstractApplication):
    def setup_options(self):
        self.parser.add_option("--registration-method", dest="reg_method",
                      type="string", default="mincANTS",
                      help="Specify whether to use minctracc or mincANTS (default)")
        self.parser.add_option("--lsq6-space", dest="lsq6_space",
                      action="store_true", default=True, 
                      help="If true (default), images have already been aligned in lsq6 space.")
        self.parser.add_option("--lsq12-space", dest="lsq12_space",
                      action="store_true", default=False, 
                      help="If true, images have already been aligned in lsq12 space. Default is false.")
        self.parser.add_option("--mask-dir", dest="mask_dir",
                      type="string", default=None, 
                      help="Directory of masks. If not specified, no masks are used. \
                            If only one mask in directory, same mask used for all scans.")
        """Add option groups from specific modules"""
        og.addMBMGroup(self.parser)
        og.tmpLongitudinalOptionGroup(self.parser)
        st.addStatsOptions(self.parser)
        
        self.parser.set_usage("%prog [options] input files") 

    def setup_backupDir(self):
        """Output directory set here as well. backups subdirectory automatically
        placed here so we don't need to set via the command line"""
        backup_dir = fh.makedirsIgnoreExisting(self.options.pipeline_dir)    
        self.pipeline.setBackupFileLocation(backup_dir)

    def setup_appName(self):
        appName = "Pairwise-nonlinear"
        return appName

    def run(self):
        options = self.options
        args = self.args
        self.reconstructCommand()
        
        """Directory handling etc as in MBM"""
        pipeDir = fh.makedirsIgnoreExisting(options.pipeline_dir)
        if not options.pipeline_name:
            pipeName = str(date.today()) + "_pipeline"
        else:
            pipeName = options.pipeline_name
        
        processedDirectory = fh.createSubDir(pipeDir, pipeName + "_processed")
        
        """Check that correct registration method was specified"""
        if options.reg_method != "minctracc" and options.reg_method != "mincANTS":
            logger.error("Incorrect registration method specified: " + options.reg_method)
            sys.exit()
        
        """Create file handling classes for each image"""
        inputs = rf.initializeInputFiles(args, processedDirectory)
        
        """Put blurs into array"""
        blurs = []
        for i in options.stats_kernels.split(","):
            blurs.append(float(i))
        
        """Create file handler for nlin average from MBM"""
        if options.nlin_avg:
            nlinFH = rfh.RegistrationFHBase(abspath(options.nlin_avg), processedDirectory)
        else:
            nlinFH = None
        if options.mbm_dir and not isdir(abspath(options.mbm_dir)):
            logger.error("The --mbm-directory specified does not exist: " + abspath(options.mbm_dir))
            sys.exit()
        
        """If directory of masks is specified, apply to each file handler.
           Two options:
              1. One mask in directory --> use for all scans. 
              2. Same number of masks as files, with same naming convention. Individual
                 mask for each scan.  
        """
        if options.mask_dir:
            absMaskPath = abspath(options.mask_dir)
            masks = walk(absMaskPath).next()[2]
            numMasks = len(masks)
            numScans = len(inputs)
            if numMasks == 1:
                for inputFH in inputs:
                    inputFH.setMask(absMaskPath + "/" + masks[0])
            elif numMasks == numScans:
                for m in masks:
                    maskBase = fh.removeBaseAndExtension(m).split("_mask")[0]
                    for inputFH in inputs:
                        if fnmatch.fnmatch(inputFH.getLastBasevol(), "*" + maskBase + "*"):
                            inputFH.setMask(absMaskPath + "/" + m)
            else:
                logger.error("Number of masks in directory does not match number of scans, but is greater than 1. Exiting...")
                sys.exit()
        
        """Get transforms from inputs to final nlin average and vice versa as well as lsq6 files"""
        if options.nlin_avg and options.mbm_dir:
            xfmsPipe = ombm.getXfms(nlinFH, inputs, options.lsq6_space, abspath(options.mbm_dir))
            if len(xfmsPipe.stages) > 0:
                self.pipeline.addPipeline(xfmsPipe)
        else:
            logger.info("MBM directory and nlin_average not specified.")
            logger.info("Calculating registration chain only")
        
        """Create a dictionary of statistics. Each subject gets an array of statistics
           indexed by inputFile."""
        subjectStats = {}
        
        """Register each image with every other image."""
        for inputFH in inputs:
            subjectStats[inputFH] = {}
            for targetFH in inputs:
                if inputFH != targetFH:
                # MF TODO: Make generalization of registration parameters easier. 
                    if options.reg_method == "mincANTS":
                        b = 0.056  
                        self.pipeline.addStage(ma.blur(inputFH, b, gradient=True))
                        self.pipeline.addStage(ma.blur(targetFH, b, gradient=True))              
                        self.pipeline.addStage(ma.mincANTS(inputFH, 
                                                           targetFH,
                                                           blur=[-1,b]))
                    elif options.reg_method == "minctracc":
                        hm = mm.HierarchicalMinctracc(inputFH, targetFH)
                        self.pipeline.addPipeline(hm.p)
                    if nlinFH:
                        resample = ma.mincresample(inputFH, targetFH, likeFile=nlinFH)
                    else:
                        resample = ma.mincresample(inputFH, targetFH, likeFile=inputFH)
                    self.pipeline.addStage(resample)
                    inputFH.setLastBasevol(resample.outputFiles[0])
                    """Calculate statistics"""
                    stats = st.CalcChainStats(inputFH, targetFH, blurs)
                    stats.fullStatsCalc()
                    self.pipeline.addPipeline(stats.p)
                    subjectStats[inputFH][targetFH] = stats.statsGroup
                    """Resample to nlin space from previous build model run, if specified"""
                    if options.nlin_avg and options.mbm_dir:
                        xfmToNlin = inputFH.getLastXfm(nlinFH, groupIndex=0)
                        for b in blurs:
                            res = ombm.resampleToCommon(xfmToNlin, inputFH, subjectStats[inputFH][targetFH], b, nlinFH)
                            self.pipeline.addPipeline(res)
                    inputFH.setLastBasevol(inputFH.inputFileName)

if __name__ == "__main__":
    
    application = PairwiseNonlinear()
    application.start()
