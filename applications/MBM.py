#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import atoms_and_modules.registration_functions as rf
import atoms_and_modules.registration_file_handling as rfh
import atoms_and_modules.LSQ6 as lsq6
import atoms_and_modules.LSQ12 as lsq12
import atoms_and_modules.NLIN as nlin
import atoms_and_modules.minc_parameters as mp
import atoms_and_modules.stats_tools as st
import os
import logging


logger = logging.getLogger(__name__)

def addMBMGroup(parser):
    group = parser.add_argument_group("MBM options", "Options for MICe-build-model.")

class MBMApplication(AbstractApplication):
    def setup_options(self):
        """Add option groups from specific modules"""
        addMBMGroup(self.parser)
        rf.addGenRegArgumentGroup(self.parser)
        lsq6.addLSQ6ArgumentGroup(self.parser)
        lsq12.addLSQ12ArgumentGroup(self.parser)
        nlin.addNlinRegArgumentGroup(self.parser)
        st.addStatsArgumentss(self.parser)
        
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
        
        #Setup init model and inital target. Function also exists if no target was specified.
        initModel, targetPipeFH = rf.setInitialTarget(options.init_model, 
                                                      options.lsq6_target, 
                                                      dirs.lsq6Dir,
                                                      self.outputDir)
            
        #LSQ6 MODULE, NUC and INORM
        runLSQ6NucInorm = lsq6.LSQ6NUCInorm(inputFiles,
                                            targetPipeFH,
                                            initModel, 
                                            dirs.lsq6Dir, 
                                            options)
        self.pipeline.addPipeline(runLSQ6NucInorm.p)
        
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
        
        #Target mask for registration--I HATE this hack, as is noted in check-in and
        #as a github issue. 
        if lsq12module.lsq12AvgFH.getMask()== None:
            if initModel[0]:
                lsq12module.lsq12AvgFH.setMask(initModel[0].getMask())
        
        #NLIN MODULE - Register with minctracc or mincANTS based on options.reg_method
        nlinObj = nlin.initializeAndRunNLIN(dirs.lsq12Dir,
                                            inputFiles,
                                            dirs.nlinDir,
                                            avgPrefix=options.pipeline_name,
                                            createAvg=False,
                                            targetAvg=lsq12module.lsq12AvgFH,
                                            nlin_protocol=options.nlin_protocol,
                                            reg_method=options.reg_method)
        
        self.pipeline.addPipeline(nlinObj.p)
        self.nlinAverages = nlinObj.nlinAverages
        
        #STATS MODULE
        if options.calc_stats:
            #Choose final average from array of nlin averages
            finalNlin = self.nlinAverages[-1]
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
    
