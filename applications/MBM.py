#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import atoms_and_modules.registration_functions as rf
import atoms_and_modules.registration_file_handling as rfh
import atoms_and_modules.LSQ6 as lsq6
import atoms_and_modules.LSQ12 as lsq12
import atoms_and_modules.NLIN as nlin
import atoms_and_modules.minc_parameters as mp
import atoms_and_modules.stats_tools as st
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
        addMBMGroup(self.parser)
        rf.addGenRegOptionGroup(self.parser)
        lsq6.addLSQ6OptionGroup(self.parser)
        lsq12.addLSQ12OptionGroup(self.parser)
        nlin.addNlinRegOptionGroup(self.parser)
        mp.addRegParamsOptionGroup(self.parser)
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
        if(options.lsq6_target != None):
            targetPipeFH = rfh.RegistrationPipeFH(os.path.abspath(options.lsq6_target), basedir=dirs.lsq6Dir)
        else: # options.init_model != None  
            initModel = rf.setupInitModel(options.init_model, self.outputDir)
            if (initModel[1] != None):
                # we have a target in "native" space 
                targetPipeFH = initModel[1]
            else:
                # we will use the target in "standard" space
                targetPipeFH = initModel[0]
            
        #LSQ6 MODULE
        lsq6module = lsq6.getLSQ6Module(inputFiles, 
                                        targetPipeFH, 
                                        lsq6Directory = dirs.lsq6Dir, 
                                        initialTransform = options.lsq6_method, 
                                        initModel = initModel, 
                                        lsq6Protocol = options.lsq6_protocol, 
                                        largeRotationParameters = options.large_rotation_parameters,
                                        largeRotationRange      = options.large_rotation_range,
                                        largeRotationInterval   = options.large_rotation_interval)
        # after the correct module has been set, get the transformation and
        # deal with resampling and potential model building
        lsq6module.createLSQ6Transformation()
        lsq6module.finalize()
        self.pipeline.addPipeline(lsq6module.p)
        
        # NUC 
        if options.nuc:
            nucorrection = lsq6.NonUniformityCorrection(inputFiles, 
                                                        initial_model=initModel,
                                                        resampleNUCtoLSQ6=False)
            nucorrection.finalize()
            self.pipeline.addPipeline(nucorrection.p)
        
        #INORMALIZE
        if options.inormalize:
            intensity_normalization = lsq6.IntensityNormalization(inputFiles,
                                                                  initial_model=initModel,
                                                                  resampleINORMtoLSQ6=True)
            self.pipeline.addPipeline(intensity_normalization.p)
        
        # LSQ12 MODULE
        # We need to specify a likeFile/space when all files are resampled
        # at the end of LSQ12. If one is not specified, use standard space. 
        if options.lsq12_likeFile == None:
            targetPipeFH = initModel[0]
        else:
            targetPipeFH = rfh.RegistrationFHBase(os.path.abspath(options.lsq12_likeFile), 
                                                  basedir=dirs.lsq12Dir)
        lsq12module = lsq12.FullLSQ12(inputFiles, 
                                      dirs.lsq12Dir, 
                                      likeFile=targetPipeFH, 
                                      maxPairs=None, 
                                      lsq12_protocol=options.lsq12_protocol,
                                      subject_matter=options.lsq12_subject_matter)
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
            #Choose final average from array of nlin averages
            numGens = len(nlinModule.nlinAverages)
            finalNlin = nlinModule.nlinAverages[numGens-1]
            # For each input file, calculate statistics from final average (finalNlin) 
            # to the inputFH space where all linear differences have been accounted for (LSQ12). 
            # The additionalXfm specified for each inputFH is the transform from the lsq6 to lsq12 
            # space for that scan. This encapsulates linear differences and is necessary for
            # some of the calculations in CalcStats.  
            for inputFH in inputFiles:
                stats = st.CalcStats(inputFH, 
                                     finalNlin, 
                                     options.stats_kernels,
                                     additionalXfm=lsq12module.lsq12AvgXfms[inputFH])
                self.pipeline.addPipeline(stats.p)
        
if __name__ == "__main__":
    
    application = MBMApplication()
    application.start()
    