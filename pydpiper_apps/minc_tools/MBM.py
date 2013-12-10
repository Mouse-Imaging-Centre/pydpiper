#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import pydpiper_apps.minc_tools.registration_functions as rf
import pydpiper_apps.minc_tools.registration_file_handling as rfh
import pydpiper_apps.minc_tools.LSQ6 as lsq6
import pydpiper_apps.minc_tools.LSQ12 as lsq12
import pydpiper_apps.minc_tools.NLIN as nlin
import pydpiper_apps.minc_tools.stats_tools as st
import Pyro
import os
from optparse import OptionGroup
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
        
        # Setup output directories for different registration modules.        
        dirs = rf.setupDirectories(self.outputDir, options.pipeline_name, module="ALL")
        inputFiles = rf.initializeInputFiles(args, dirs.processedDir, maskDir=options.mask_dir)
        
        # TODO: Write this as a function in LSQ6 and call from there. 
        initModel = None
        if(options.target != None):
            targetPipeFH = rfh.RegistrationPipeFH(os.path.abspath(options.target), basedir=dirs.lsq6Dir)
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
        lsq6module = lsq6.getLSQ6Module(inputFiles, 
                                        targetPipeFH, 
                                        lsq6Directory = dirs.lsq6Dir, 
                                        initialTransform = options.lsq6_method, 
                                        initModel = initModel, 
                                        lsq6Protocol = options.lsq6_protocol, 
                                        largeRotationParameters = options.large_rotation_parameters)
        # after the correct module has been set, get the transformation and
        # deal with resampling and potential model building
        lsq6module.createLSQ6Transformation()
        lsq6module.finalize()
        self.pipeline.addPipeline(lsq6module.p)
        
        # NUC 
        nucorrection = lsq6.NonUniformityCorrection(inputFiles,
                                                    initial_model = initModel,
                                                    resampleNUCtoLSQ6 = False)
        nucorrection.finalize()
        self.pipeline.addPipeline(nucorrection.p)
        
        # INORMALIZE
        intensity_normalization = lsq6.IntensityNormalization(inputFiles,
                                                              initial_model = initModel,
                                                              resampleINORMtoLSQ6 = True)
        self.pipeline.addPipeline(intensity_normalization.p)
        
        #LSQ12 MODULE
        lsq12module = lsq12.FullLSQ12(inputFiles, 
                                      dirs.lsq12Dir, 
                                      likeFile=targetPipeFH, 
                                      maxPairs=None, 
                                      lsq12_protocol=options.lsq12_protocol)
        lsq12module.iterate()
        self.pipeline.addPipeline(lsq12module.p)
        
        #TODO: Additional NUC step here. This will impact both the lsq6 and lsq12 modules. 
        # May want to not do resampling and averaging by default. TBD. 
        
        #NLIN MODULE - Register with minctracc or mincANTS based on options.reg_method
        nlinModule = nlin.initNLINModule(inputFiles, 
                                         lsq12module.lsq12AvgFH, 
                                         dirs.nlinDir, 
                                         options.nlin_protocol, 
                                         options.reg_method)
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
    