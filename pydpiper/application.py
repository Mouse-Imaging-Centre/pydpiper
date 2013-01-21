from optparse import OptionParser
from pydpiper.pipeline import Pipeline, pipelineDaemon
from pydpiper.queueing import runOnQueueingSystem
from pydpiper.file_handling import makedirsIgnoreExisting
from datetime import datetime
import Pyro
import logging
import networkx as nx
import sys

logger = logging.getLogger(__name__)

class AbstractApplication(object):
    """Framework class for writing applications for PydPiper. 
    
       This class defines the default behaviour for accepting common command-line options, and executing the application
       under various queueing systems. 
       
       Subclasses should extend the following methods:
           setup_appName()
           setup_logger() [optional, default method is defined here]
           setup_options()
           run()
    
       Usage: 
          class MyApplication(AbstractApplication):
                ... 
           
          if __name__ == "__main__":
              application = ConcreteApplication()
              application.start()
    """
    def __init__(self):
        Pyro.config.PYRO_MOBILE_CODE=1 
        self.parser = OptionParser()
    
    def _setup_options(self):
            # PydPiper options
        self.parser.add_option("--uri-file", dest="urifile",
                          type="string", default=None,
                          help="Location for uri file if NameServer is not used. If not specified, default is current working directory.")
        self.parser.add_option("--use-ns", dest="use_ns",
                          action="store_true",
                          help="Use the Pyro NameServer to store object locations")
        self.parser.add_option("--create-graph", dest="create_graph",
                          action="store_true",
                          help="Create a .dot file with graphical representation of pipeline relationships")
        self.parser.add_option("--num-executors", dest="num_exec", 
                          type="int", default=0, 
                          help="Launch executors automatically without having to run pipeline_excutor.py independently.")
        self.parser.add_option("--time", dest="time", 
                          type="string", default="2:00:00:00", 
                          help="Wall time to request for each executor in the format dd:hh:mm:ss")
        self.parser.add_option("--proc", dest="proc", 
                          type="int", default=8,
                          help="Number of processes per executor. Default is 8. Also sets max value for processor use per executor.")
        self.parser.add_option("--mem", dest="mem", 
                          type="float", default=16,
                          help="Total amount of requested memory. Default is 16G.")
        self.parser.add_option("--ppn", dest="ppn", 
                          type="int", default=8,
                          help="Number of processes per node. Default is 8. Used when --queue=pbs")
        self.parser.add_option("--queue", dest="queue", 
                          type="string", default=None,
                          help="Use specified queueing system to submit jobs. Default is None.")
        self.parser.add_option("--sge-queue-opts", dest="sge_queue_opts", 
                          type="string", default=None,
                          help="For --queue=sge, allows you to specify different queues. If not specified, default is used.")
        self.parser.add_option("--restart", dest="restart", 
                          action="store_true",
                          help="Restart pipeline using backup files.")
        self.parser.add_option("--output-dir", dest="output_directory",
                          type="string", default=".",
                          help="Directory where output data and backups will be saved.")   
    
    def _setup_pipeline(self):
        self.pipeline = Pipeline()
        self.setup_directories()
        
    def setup_directories(self):
        """Output and backup directories setup here."""
        self.outputDir = makedirsIgnoreExisting(self.options.output_directory)
        self.pipeline.setBackupFileLocation(self.outputDir)
    
    def reconstructCommand(self):    
        reconstruct = ""
        for i in range(len(sys.argv)):
            reconstruct += sys.argv[i] + " "
        logger.info("Command is: " + reconstruct)
        
    def start(self):
        self._setup_options()
        self.setup_options()
        
        self.options, self.args = self.parser.parse_args()        
        self._setup_pipeline()
        
        self.appName = self.setup_appName()
        self.setup_logger()
        
        if self.options.queue=="pbs":
            roq = runOnQueueingSystem(self.options, sys.argv)
            roq.createPbsScripts()
            return 
        
        if self.options.restart:
            logger.info("Restarting pipeline from pickled files.")
            self.pipeline.restart()
            self.pipeline.initialize()
            self.pipeline.printStages(self.appName)
        else:
            self.reconstructCommand()
            self.run()
            self.pipeline.initialize()
            self.pipeline.printStages(self.appName)
                            
        if self.options.create_graph:
            logger.debug("Writing dot file...")
            nx.write_dot(self.pipeline.G, "labeled-tree.dot")
            logger.debug("Done.")
                
        #pipelineDaemon runs pipeline, launches Pyro client/server and executors (if specified)
        # if use_ns is specified, Pyro NameServer must be started. 
        logger.info("Starting pipeline daemon...")
        pipelineDaemon(self.pipeline, self.options, sys.argv[0])
        logger.info("Server has stopped.  Quitting...")

    def setup_appName(self):
        """sets the name of the application"""
        pass

    def setup_logger(self):
        """sets logging info specific to application"""
        FORMAT = '%(asctime)-15s %(name)s %(levelname)s: %(message)s'
        now = datetime.now()  
        FILENAME = str(self.appName) + "-" + now.strftime("%Y%m%d-%H%M%S%f") + ".log"
        logging.basicConfig(filename=FILENAME, format=FORMAT, level=logging.DEBUG)

    def setup_options(self):
        """Set up the self.options option parser with options this application needs."""
        pass
    
    def run(self):
        """Run this application.
        
           """
        pass
