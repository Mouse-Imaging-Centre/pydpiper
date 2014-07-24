#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import pydpiper.file_handling as fh
import atoms_and_modules.registration_functions as rf
import atoms_and_modules.registration_file_handling as rfh
import atoms_and_modules.minc_modules as mm
import atoms_and_modules.minc_atoms as ma
import atoms_and_modules.LSQ6 as lsq6
import atoms_and_modules.LSQ12 as lsq12
import atoms_and_modules.NLIN as nlin
import atoms_and_modules.stats_tools as st
import atoms_and_modules.minc_parameters as mp
import Pyro
from optparse import OptionGroup
from os.path import abspath, isdir, isfile
import logging
import sys

logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

def addRegChainOptionGroup(parser):
    """option group for the command line argument parser"""
    group = OptionGroup(parser, "Registration-chain options", 
                        "Options for registering consecutive timepoints of longitudinal data.")
    group.add_option("--avg-time-point", dest="avg_time_point",
                      type="int", default=1,
                      help="Time point averaged prior to this registration to get common nlin space.")
    group.add_option("--common-space-name", dest="common_name",
                      type="string", default="common", 
                      help="Option to specify a name for the common space. This is useful for the "
                            "creation of more readable output file names. Default is common. Note "
                            "that the common space is the one created by an iterative group-wise " 
                            "registration, either within this code or one that was run previously.")
    group.add_option("--run-groupwise", dest="run_groupwise",
                      action="store_true",
                      help="Run an iterative, groupwise registration (MBM) on the specified average "
                           " time point. [Default]")
    group.add_option("--no-run-groupwise", dest="run_groupwise",
                      action="store_false", 
                      help="If specified, do not run a groupwise registration on the specified "
                            "average time point [Opposite of --run-groupwise.]. If an iterative "
                            "group-wise registration was run previously, the average and transforms "
                            "from that registration can be accessed using the --MBM-directory and "
                            "--nlin-average options. (See below.)")
    group.add_option("--MBM-directory", dest="mbm_dir",
                      type="string", default=None, 
                      help="_processed directory from MBM used to average specified time point ")
    group.add_option("--nlin-average", dest="nlin_avg",
                      type="string", default=None, 
                      help="Final nlin average from MBM run.")
    parser.set_defaults(run_groupwise=True)
    parser.add_option_group(group)

class RegistrationChain(AbstractApplication):
    def setup_options(self):
        """Add option groups from specific modules"""
        addRegChainOptionGroup(self.parser)
        rf.addGenRegOptionGroup(self.parser)
        lsq6.addLSQ6OptionGroup(self.parser)
        lsq12.addLSQ12OptionGroup(self.parser)
        nlin.addNlinRegOptionGroup(self.parser)
        st.addStatsOptions(self.parser)
        
        self.parser.set_usage("%prog [options] input.csv") 

    def setup_appName(self):
        appName = "Registration-chain"
        return appName

    def run(self):
        
        #Setup output directories for registration chain (_processed only)       
        dirs = rf.setupDirectories(self.outputDir, self.options.pipeline_name, module="ALL")
        
        #Check that correct registration method was specified
        if self.options.reg_method != "minctracc" and self.options.reg_method != "mincANTS":
            logger.error("Incorrect registration method specified: " + self.options.reg_method)
            sys.exit()
        
        #Take average time point, subtract 1 for proper indexing
        avgTime = self.options.avg_time_point - 1
        
        #Read in files from csv
        subjects = rf.setupSubjectHash(self.args[0], dirs, self.options.mask_dir)
        
        # If input = native space then do LSQ6 first on all files.
        if self.options.input_space == "native":
            initModel, targetPipeFH = rf.setInitialTarget(self.options.init_model, 
                                                          self.options.lsq6_target, 
                                                          dirs.lsq6Dir,
                                                          self.outputDir)
            #LSQ6 MODULE, NUC and INORM
            inputFiles = []
            for subj in subjects:
                for i in range(len(subjects[subj])):
                    inputFiles.append(subjects[subj][i])
            runLSQ6NucInorm = lsq6.LSQ6NUCInorm(inputFiles,
                                                targetPipeFH,
                                                initModel, 
                                                dirs.lsq6Dir, 
                                                self.options)
            self.pipeline.addPipeline(runLSQ6NucInorm.p)
        
        elif self.options.input_space == "lsq6":
            initModel = None
        else:
            print """Only native and lsq6 are allowed as input_space options for the registration chain. You specified: """ + str(self.options.input_space)
            print "Exiting..."
            sys.exit()
            
        #Get current group index for use later, when chain is run. 
        #Value will be different for input_space == native vs LSQ6
        currGroupIndex = rf.getCurrIndexForInputs(subjects)
        
        #If requested, run iterative groupwise registration for all subjects at the specified
        #common timepoint, otherwise look to see if files are specified from a previous run.
        if self.options.run_groupwise:
            inputs = []
            for s in subjects:
                inputs.append(subjects[s][avgTime])
            #Run full LSQ12 and NLIN modules.
            lsq12Nlin = mm.FullIterativeLSQ12Nlin(inputs, 
                                                  dirs, 
                                                  self.options, 
                                                  avgPrefix=self.options.common_name, 
                                                  initModel=initModel)
            self.pipeline.addPipeline(lsq12Nlin.p)
            nlinFH = lsq12Nlin.nlinFH
            
            #Set lastBasevol and group to lsq6 space, using currGroupIndex set above.
            for i in inputs:
                i.currentGroupIndex = currGroupIndex
        else: 
            if self.options.mbm_dir and self.options.nlin_avg:
                if (not isdir(self.options.mbm_dir)) or (not isfile(self.options.nlin_avg)):
                    logger.error("The specified MBM-directory or nlin-average do not exist.") 
                    logger.error("Specified MBM-directory: " + abspath(self.options.mbm_dir))
                    logger.error("Specified nlin average: " + abspath(self.options.nlin_avg))
                    sys.exit()
                # create file handler for nlin average
                nlinFH = rfh.RegistrationPipeFH(abspath(self.options.nlin_avg), basedir=dirs.processedDir)
                # Get transforms from subjects at avg time point to final nlin average and vice versa 
                rf.getXfms(nlinFH, subjects, "lsq6", abspath(self.options.mbm_dir), time=avgTime)
            else:
                logger.info("MBM directory and nlin_average not specified.")
                logger.info("Calculating registration chain only") 
                nlinFH = None
        
        for subj in subjects:
            s = subjects[subj]
            count = len(s) 
            for i in range(count - 1):
                s[i].newGroup(groupName="chain")
                if self.options.reg_method == "mincANTS":
                    register = mm.LSQ12ANTSNlin(s[i], 
                                                s[i+1],
                                                lsq12_protocol=self.options.lsq12_protocol,
                                                nlin_protocol=self.options.nlin_protocol)
                    self.pipeline.addPipeline(register.p)
                elif self.options.reg_method == "minctracc":
                    hm = mm.HierarchicalMinctracc(s[i], 
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
                                                    self.options.stats_kernels, 
                                                    self.options.common_name) 
        self.pipeline.addPipeline(car.p)

if __name__ == "__main__":
    
    application = RegistrationChain()
    application.start()
