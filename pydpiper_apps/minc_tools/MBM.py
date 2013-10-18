#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import pydpiper.file_handling as fh
import pydpiper_apps.minc_tools.registration_functions as rf
import pydpiper_apps.minc_tools.registration_file_handling as rfh
import pydpiper_apps.minc_tools.minc_modules as mm
import pydpiper_apps.minc_tools.LSQ6 as lsq6
import pydpiper_apps.minc_tools.LSQ12 as lsq12
import pydpiper_apps.minc_tools.NLIN as nlin
import pydpiper_apps.minc_tools.stats_tools as st
import Pyro
import os
from optparse import OptionGroup
from datetime import date
import logging


logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

def addMBMGroup(parser):
    group = OptionGroup(parser, "MBM options", 
                        "Options for MICe-build-model.")
    parser.add_option_group(group)

class MBMApplication(AbstractApplication):
    def setup_options(self):
        """Add option groups from specific modules"""
        rf.addGenRegOptionGroup(self.parser)
        addMBMGroup(self.parser)
        lsq6.addLSQ6OptionGroup(self.parser)
        lsq12.addLSQ12OptionGroup(self.parser)
        nlin.addNlinRegOptionGroup(self.parser)
        st.addStatsOptions(self.parser)
        
        self.parser.set_usage("%prog [options] input files") 

    def setup_appName(self):
        appName = "MICe-build-model"
        return appName

    def run(self):
        options = self.options
        args = self.args
        
        #Setup pipeline name
        if not options.pipeline_name:
            pipeName = str(date.today()) + "_pipeline"
        else:
            pipeName = options.pipeline_name
        # TODO: The lsq6, 12 and nlin directory creation will be created in
        # appropriate modules. self.outputDir is set in AbstractApplication before
        # run() is called.  
        lsq6Directory = fh.createSubDir(self.outputDir, pipeName + "_lsq6")
        lsq12Directory = fh.createSubDir(self.outputDir, pipeName + "_lsq12")
        nlinDirectory = fh.createSubDir(self.outputDir, pipeName + "_nlin")
        processedDirectory = fh.createSubDir(self.outputDir, pipeName + "_processed")
        inputFiles = rf.initializeInputFiles(args, processedDirectory, maskDir=options.mask_dir)
        
        # TODO: Write this as a function in LSQ6 and call from there. 
        initModel = None
        if(options.target != None):
            targetPipeFH = rfh.RegistrationPipeFH(os.path.abspath(options.target), basedir=lsq6Directory)
        else: # options.init_model != None  
            initModel = rf.setupInitModel(options.init_model, self.outputDir)
            if (initModel[1] != None):
                # we have a target in "native" space 
                targetPipeFH = initModel[1]
            else:
                # we will use the target in "standard" space
                targetPipeFH = initModel[0]
        
        #TODO: May want to pre-mask since minc_compression is turned on and we can save disk space. 
        
        #TODO: Add a commmand line option to specify a resolution for running registration. 
        # After resampling all files, create new or re-set init model for LSQ6 at this new resolution.
        # Code below is example, will need modification.
#        filesToResample = [initModel[0]]
#        if initModel[1]:
#            filesToResample.append(initModel[1])
#        for i in inputFiles:
#            filesToResample.append(i)
#        
#        #NOTE: Test function, this will eventually be called from LSQ6
#        # and resolution will NOT be hardcoded. Because obviously. 
#        resolution = 0.056
#        resPipe = mm.SetResolution(filesToResample, resolution)
#        if len(resPipe.p.stages) > 0:
#            # Only add to pipeline if resampling is needed
#            self.pipeline.addPipeline(resPipe.p)
            
        #LSQ6 MODULE
        lsq6module = None
        """
            Option 1) run a simple lsq6: the input files are assumed to be in the
            same space and roughly in the same orientation.
        """
        if(options.lsq6_method == "lsq6_simple"):
            lsq6module =  lsq6.LSQ6HierarchicalMinctracc(inputFiles,
                                                    targetPipeFH,
                                                    initial_model     = initModel,
                                                    lsq6OutputDir     = lsq6Directory,
                                                    initial_transform = "identity",
                                                    lsq6_protocol     = options.lsq6_protocol)
        """
            Option 2) run an lsq6 registration where the centre of the input files
            is estimated.  Orientation is assumed to be similar, space is not.
        """
        if(options.lsq6_method == "lsq6_centre_estimation"):
            lsq6module =  lsq6.LSQ6HierarchicalMinctracc(inputFiles,
                                                    targetPipeFH,
                                                    initial_model     = initModel,
                                                    lsq6OutputDir     = lsq6Directory,
                                                    initial_transform = "estimate",
                                                    lsq6_protocol     = options.lsq6_protocol)
        """
            Option 3) run a brute force rotational minctracc.  Input files can be
            in any random orientation and space.
        """
        if(options.lsq6_method == "lsq6_large_rotations"):
            lsq6module = lsq6.LSQ6RotationalMinctracc(inputFiles,
                                                 targetPipeFH,
                                                 initial_model = initModel,
                                                 lsq6OutputDir = lsq6Directory,
                                                 large_rotation_parameters = options.large_rotation_parameters)
        
        # after the correct module has been set, get the transformation and
        # deal with resampling and potential model building
        lsq6module.createLSQ6Transformation()
        lsq6module.finalize()
        self.pipeline.addPipeline(lsq6module.p)
        
        #TODO: NUC and INORMALIZE HERE. 
        
        #LSQ12 MODULE
        lsq12module = lsq12.FullLSQ12(inputFiles, 
                                      lsq12Directory, 
                                      likeFile=targetPipeFH, 
                                      maxPairs=None, 
                                      lsq12_protocol=options.lsq12_protocol)
        lsq12module.iterate()
        self.pipeline.addPipeline(lsq12module.p)
        
        #TODO: Additional NUC step here. This will impact both the lsq6 and lsq12 modules. 
        # May want to not do resampling and averaging by default. TBD. 
        
        #NLIN MODULE - Need to handle minctracc case also
        nlinModule = nlin.NLINANTS(inputFiles, 
                                   lsq12module.lsq12AvgFH, 
                                   nlinDirectory, 
                                   options.nlin_protocol)
        nlinModule.iterate()
        self.pipeline.addPipeline(nlinModule.p)
        
        #STATS MODULE
        if options.calc_stats:
            """Get blurs from command line option and put into array"""
            blurs = []
            for i in options.stats_kernels.split(","):
                blurs.append(float(i))
            """Choose final average from array of nlin averages"""
            numGens = len(nlinModule.nlinAverages)
            finalNlin = nlinModule.nlinAverages[numGens-1]
            """For each input file, calculate statistics from finalNlin to input"""
            for inputFH in inputFiles:
                stats = st.CalcStats(inputFH, 
                                     finalNlin, 
                                     blurs, 
                                     inputArray=inputFiles,
                                     scalingFactor=lsq12module.lsq12AvgXfms[inputFH])
                stats.fullStatsCalc()
                self.pipeline.addPipeline(stats.p)
        
if __name__ == "__main__":
    
    application = MBMApplication()
    application.start()
    