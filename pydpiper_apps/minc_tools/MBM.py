#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import pydpiper.file_handling as fh
import pydpiper_apps.minc_tools.registration_functions as rf
import pydpiper_apps.minc_tools.minc_modules as mm
import pydpiper_apps.minc_tools.option_groups as og
import Pyro
from datetime import date
import logging


logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

class MBMApplication(AbstractApplication):
    def setup_options(self):
        self.parser.add_option("--init-model", dest="init_model",
                      type="string", default=None,
                      help="Name of file to register towards. If unspecified, bootstrap.")
        
        """Add option groups from specific modules"""
        og.addMBMGroup(self.parser)
        
        self.parser.set_usage("%prog [options] input files") 

    def setup_backupDir(self):
        """Output directory set here as well. backups subdirectory automatically
        placed here so we don't need to set via the command line"""
        backup_dir = fh.makedirsIgnoreExisting(self.options.pipeline_dir)    
        self.pipeline.setBackupFileLocation(backup_dir)

    def setup_appName(self):
        appName = "MICe-build-model"
        return appName

    def run(self):
        options = self.options
        args = self.args
        self.reconstructCommand()
        
        # Make main pipeline directories
        pipeDir = fh.makedirsIgnoreExisting(options.pipeline_dir)
        if not options.pipeline_name:
            pipeName = str(date.today()) + "_pipeline"
        else:
            pipeName = options.pipeline_name
        
        processedDirectory = fh.createSubDir(pipeDir, pipeName + "_processed")
        inputFiles = rf.initializeInputFiles(args, processedDirectory)
        
        if options.init_model:
            """setupInitModel returns a tuple containing:
               (standardFH, nativeFH, native_to_standard.xfm)
               First value must exist, others may be None 
            """
            initModel = rf.setupInitModel(options.init_model, pipeDir)
        else:
            """"Bootstrap using the first image in inputFiles
                Note: This will be a full FH class, not the base, 
                as above
            """
            initModel = (inputFiles[0], None, None)
        
        #Pre-masking here if flagged? 
        
        filesToResample = [initModel[0]]
        if initModel[1]:
            filesToResample.append(initModel[1])
        for i in inputFiles:
            filesToResample.append(i)
        
        #NOTE: Test function, this will eventually be called from LSQ6
        # and resolution will NOT be hardcoded. Because obviously. 
        resolution = 0.056
        resPipe = mm.SetResolution(filesToResample, resolution)
        if len(resPipe.p.stages) > 0:
            # Only add to pipeline if resampling is needed
            self.pipeline.addPipeline(resPipe.p)
            
        
if __name__ == "__main__":
    
    application = MBMApplication()
    application.start()
    