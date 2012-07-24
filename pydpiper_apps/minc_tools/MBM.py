#!/usr/bin/env python

from pydpiper.application import AbstractApplication
import pydpiper.file_handling as fh
import pydpiper_apps.minc_tools.registration_functions as rf
import Pyro
from datetime import date, datetime
import logging


logger = logging.getLogger(__name__)

Pyro.config.PYRO_MOBILE_CODE=1 

class MBMApplication(AbstractApplication):
    def setup_options(self):
        self.parser.add_option("--pipeline-name", dest="pipeline_name",
                      type="string", default=None,
                      help="Name of pipeline and prefix for models.")
        self.parser.add_option("--pipeline-dir", dest="pipeline_dir",
                      type="string", default=".",
                      help="Directory for placing pipeline results. Default is current.")
        
        self.parser.set_usage("%prog [options] input files") 

    def setup_backupDir(self):
        """Output directory set here as well. backups subdirectory automatically
        placed here so we don't need to set via the command line"""
        backup_dir = fh.makedirsIgnoreExisting(self.options.pipeline_dir)    
        self.pipeline.setBackupFileLocation(backup_dir)

    def setup_appName(self):
        appName = "MICe-build-model"
        return appName
    
    def setup_logger(self):
        FORMAT = '%(asctime)-15s %(name)s %(levelname)s: %(message)s'
        now = datetime.now()  
        FILENAME = str(self.appName) + "-" + now.strftime("%Y%m%d-%H%M%S") + ".log"
        logging.basicConfig(filename=FILENAME, format=FORMAT, level=logging.DEBUG)

    def run(self):
        options = self.options
        args = self.args
        self.reconstructCommand()
        
        # Make main pipeline directories
        pipeDir = fh.makedirsIgnoreExisting(options.pipeline_dir)
        if not options.pipeline_name:
            pipeName = str(date.today()) + "_pipeline"
            print pipeName
        else:
            pipeName = options.pipeline_name
        
        processedDirectory = fh.createSubDir(pipeDir, pipeName + "_processed")
        inputFiles = rf.initializeInputFiles(args, processedDirectory)
        
if __name__ == "__main__":
    
    application = MBMApplication()
    application.start()